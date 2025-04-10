import torch
from .mpcg_searcher import MPCGACSearcher
from torch import nn, optim
import numpy as np


#%% MODEL
class Actor(nn.Module):
    def __init__(self, state_dim=6, hidden_dim=256, z_dim=100, dtype=torch.float64):
        super(Actor, self).__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.residual_part = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.ReLU(),
        )
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, z_dim)
        self.fc_log_var = nn.Linear(hidden_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid()
        )
        self.to(dtype)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, state, z):
        encoded_state = self.state_encoder(state)
        encoded_z = self.z_encoder(z).unsqueeze(0).expand_as(encoded_state)
        encoded = torch.cat((encoded_state, encoded_z), dim=-1)
        encoded = self.shared_encoder(encoded)
        mean = self.fc_mean(encoded)
        log_var = self.fc_log_var(encoded)
        z_sample = self.reparameterize(mean, log_var)
        decoded = self.decoder(z_sample)
        output = decoded + self.residual_part(encoded_state)
        return output


class Critic(nn.Module):
    def __init__(self, state_dim=6, hidden_dim=256, z_dim=100, dtype=torch.float64):
        super(Critic, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, z_dim)
        self.fc_log_var = nn.Linear(hidden_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.to(dtype)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, state):
        encoded = self.encoder(state)
        mean = self.fc_mean(encoded)
        log_var = self.fc_log_var(encoded)
        z_sample = self.reparameterize(mean, log_var)
        decoded_value = self.decoder(z_sample)
        output_value = decoded_value
        return output_value


#%% MPCGAC
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import time
def extract(num, shape, device):
    num = torch.tensor([num] * shape[0]).to(device)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    num = num.reshape(*reshape)
    return num

global_fixed_noise = None

def get_fixed_noise(args, dataloader, device):
    with torch.no_grad():
        for source_data, _, _, _ in dataloader:
            source_data = source_data.to(device, non_blocking=True)
            if args.input_channels == 3:
                source_data = source_data.squeeze(1)
            noises = []
            for _ in range(3):
                noises.append(torch.randn_like(source_data).to(device))
            fixed_noise = torch.stack(noises)
            fixed_noise = torch.mean(fixed_noise, dim=0).to(device)
            return fixed_noise

def topological_sort(graph):
    visited = set()
    stack = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in graph[node][1]:
            if neighbor != -1:
                dfs(neighbor)
        stack.append(node)

    for node in graph:
        if node not in visited:
            dfs(node)
    return stack

def get_cell_inputs(args, cell_input, mpcg_result, source, device):
    inputs_x0 = []
    inputs_xt = []
    xt_noise = torch.randn_like(source).to(device)
    if args.sample_fixed:
        xt_noise = global_fixed_noise
    for input_id in cell_input:
        if input_id == -1:
            inputs_x0.append(source)
            inputs_xt.append(xt_noise)
        else:
            inputs_group = mpcg_result.get(input_id)
            if inputs_group is None:
                inputs_x0.append(source)
                inputs_xt.append(xt_noise)
                print("[Error]: input not found about input_id ({})".format(input_id))
            else:
                inputs_x0.append(inputs_group[0])
                inputs_xt.append(inputs_group[1])
    if inputs_x0:
        input_x0 = torch.stack(inputs_x0)
        input_x0 = torch.mean(input_x0, dim=0).to(device)
        input_xt = torch.stack(inputs_xt)
        input_xt = torch.mean(input_xt, dim=0).to(device)
    else:
        input_x0 = source
        input_xt = xt_noise
    return input_x0, input_xt

def get_cell_pulses(x_0, x_t, intensity, capacity, pulses_type, device):
    if x_t is None or pulses_type == "noise_pulses":
        x_t = torch.randn_like(x_0)
    h_t = (extract((1. - intensity) * capacity, x_0.shape, device) * x_0
           + extract(intensity * capacity, x_0.shape, device) * x_t)
    return h_t

def sample_by_mpcg(args, mpcg, model, source, device):
    sample_sequence = topological_sort(mpcg)
    mpcg_result = {key: None for key in mpcg}
    with torch.no_grad():
        for mpcg_id in sample_sequence:
            cell_info = mpcg[mpcg_id] # id : [time_layer, [input_ids], pulses, capacity]
            input_x0, input_xt = get_cell_inputs(args, cell_info[1], mpcg_result, source, device)
            if mpcg_id == 0.0:
                mpcg_result[0.0] = [get_cell_pulses(input_x0, input_xt, cell_info[2], cell_info[3], args.pulses_type, device), input_xt]
                continue
            cell_time = torch.full((input_x0.size(0),), cell_info[0], dtype=torch.int64).to(device)
            cell_latent = torch.randn(input_x0.size(0), args.z_emb_dim, device=device)
            if args.sample_fixed:
                cell_latent = torch.zeros(input_x0.size(0), args.z_emb_dim, device=device)
            h_t = get_cell_pulses(input_x0, input_xt, cell_info[2], cell_info[3], args.pulses_type, device)
            output_x0, _ = model(torch.cat((h_t, source),axis=1), cell_time, cell_latent)
            output_xt = h_t
            mpcg_result[mpcg_id] = [output_x0, output_xt]
    result = mpcg_result.get(0.0)
    if result is None:
        assert "[Error]: The result is None. "
    return result[0]

def evaluate_samples(args, real_data, fake_sample):
    to_range_0_1 = lambda x: (x + 1.) / 2.
    real_data = real_data.cpu().numpy()
    fake_sample = fake_sample.cpu().numpy()
    psnr_list = []
    ssim_list = []
    mae_list = []
    for i in range(real_data.shape[0]):
        real_data_i = real_data[i]
        fake_sample_i = fake_sample[i]
        real_data_i = to_range_0_1(real_data_i)
        real_data_i = real_data_i / real_data_i.max()
        fake_sample_i = to_range_0_1(fake_sample_i)
        fake_sample_i = fake_sample_i / fake_sample_i.max()
        psnr_val = psnr(real_data_i, fake_sample_i, data_range=real_data_i.max() - real_data_i.min())
        mae_val = np.mean(np.abs(real_data_i - fake_sample_i))
        if args.input_channels == 1:
            ssim_val = ssim(real_data_i[0], fake_sample_i[0], data_range=real_data_i.max() - real_data_i.min())
        elif args.input_channels == 3:
            real_data_i = np.squeeze(real_data_i).transpose(1, 2, 0)
            fake_sample_i = np.squeeze(fake_sample_i).transpose(1, 2, 0)
            ssim_val = ssim(real_data_i, fake_sample_i, channel_axis=-1, data_range=real_data_i.max() - real_data_i.min())
        else:
            raise ValueError("Unsupported number of input channels")
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        mae_list.append(mae_val)
    return psnr_list, ssim_list, mae_list

def mpcg_to_tensor(mpcg, dtype=torch.float64, device=torch.device("cuda:0")):
    pulses = []
    capacity = []
    for key in sorted(mpcg.keys()):
        if key >= 1.0 or key <= 0.0:
            continue
        values = mpcg[key]
        pulses.append(values[2])
        capacity.append(values[3])
    all_tensors = torch.tensor(pulses+capacity, dtype=dtype).to(device)
    return all_tensors

def tensor_to_mpcg(mpcg, tensor):
    all_lists = tensor.cpu().detach().tolist()
    mid = len(all_lists) // 2

    pulses = all_lists[:mid]
    capacity = all_lists[mid:]
    count = 0
    for key in sorted(mpcg.keys()):
        if key >= 1.0 or key <= 0.0:
            continue
        mpcg[key][2] = pulses[count]
        mpcg[key][3] = capacity[count]
        count += 1
    return mpcg


def sample_one(args, dataloader, model, mpcg, device):
    PSNR = []
    SSIM = []
    MAE = []
    with torch.no_grad():
        for source_data, target_data, _, _ in dataloader:
            target_data = target_data.to(device, non_blocking=True)
            source_data = source_data.to(device, non_blocking=True)
            if args.input_channels == 3:
                target_data = target_data.squeeze(1)
                source_data = source_data.squeeze(1)
            fake_sample = sample_by_mpcg(args, mpcg, model, source_data, device)
            psnr_list, ssim_list, mae_list = evaluate_samples(args, target_data, fake_sample)
            PSNR.extend(psnr_list)
            SSIM.extend(ssim_list)
            MAE.extend(mae_list)
    v = sum(PSNR) / len(PSNR) + 30 * sum(SSIM) / len(SSIM)
    return v

class MPCGACTrainer:
    def __init__(self, args, mpcg_dim=6, hidden_dim=256, z_dim=100, env_step=30000,
                       mpcg_lr=1e-4, mpcg_lrf=1e-5, batch_size=16,
                 mpcg=None, model=None, dataloader=None, device=torch.device('cuda:0')):
        self.args = args
        self.device = device
        self.mpcg_dim = mpcg_dim
        self.hidden_dim = hidden_dim
        self.dtype = torch.float64
        self.z_dim = z_dim
        self.actor = Actor(state_dim=mpcg_dim, hidden_dim=hidden_dim, z_dim=z_dim, dtype=self.dtype).to(self.device)
        self.critic = Critic(state_dim=mpcg_dim, hidden_dim=hidden_dim, z_dim=z_dim, dtype=self.dtype).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=mpcg_lr, betas=(self.args.beta1, self.args.beta2))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=mpcg_lr, betas=(self.args.beta1, self.args.beta2))
        self.env_steps = env_step
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, self.env_steps, eta_min=mpcg_lrf)
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, self.env_steps, eta_min=mpcg_lrf)
        self.lambda_reg = 0.1
        self.lambda_vm = 1
        self.batch_size = batch_size
        self.mpcg = mpcg
        self.model = model
        self.dataloader = dataloader
        self.opt_state = None
        self.opt_mpcg = None
        self.opt_v = -1000.
        self.org_v = -1000.
        self.args.C = 60
        self.searcher = MPCGACSearcher(self.args, self.mpcg, self.model, self.dataloader)
        self.buffer = self.searcher.get_buffer()

        self.change_global_noise()

    def change_global_noise(self):
        if self.args.sample_fixed:
            global global_fixed_noise
            global_fixed_noise = get_fixed_noise(self.args, self.dataloader, self.device)

    def update_networks(self):
        if len(self.searcher.buffer) < self.batch_size:
            return
        for p in self.critic.parameters():
            p.requires_grad = True
        self.critic_optimizer.zero_grad()
        states, vs = self.searcher.buffer.sample_all(self.batch_size)
        states = torch.stack(states)
        vs = torch.tensor(vs, dtype=self.dtype, device=self.device).unsqueeze(1) / self.args.C
        vs_predict = self.critic(states)
        critic_loss = nn.MSELoss()(vs_predict, vs)
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        for p in self.critic.parameters():
            p.requires_grad = False
        self.actor_optimizer.zero_grad()
        latent_z = torch.randn(self.z_dim, dtype=self.dtype, device=self.device)
        new_states = self.actor(states, latent_z)
        new_vs_predict = self.critic(new_states)
        actor_loss = (-self.lambda_vm * new_vs_predict.mean()
                      + self.lambda_reg * torch.norm(new_states - states, p=2))
        if torch.isnan(actor_loss).any() or torch.isinf(actor_loss).any():
            actor_loss = torch.tensor(0.0, device=self.device)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        return actor_loss.item(), critic_loss.item()

    def validate(self):
        with torch.no_grad():
            states, vs = self.searcher.buffer.sample_elite(1)
            states = torch.stack(states)
            latent_z = torch.randn(self.z_dim, dtype=self.dtype, device=self.device)
            new_states = self.actor(states, latent_z)
            mpcg_test = tensor_to_mpcg(self.mpcg, new_states[0])
            v = sample_one(self.args, self.dataloader, self.model, mpcg_test, self.device)
            return mpcg_test, v, new_states[0]

    def get_result(self):
        mpcg_test, v_test, _ = self.validate()
        print("Actor Opt V")
        print("=" * 50)
        print("\033[92m" + str(mpcg_test) + "\033[0m")
        print("Actor Opt V: " + str(v_test))
        print("=" * 50 + "\n")
        print(" ")
        print("Truth Opt V")
        print("=" * 50)
        print("\033[93m" + str(self.searcher.opt_mpcg) + "\033[0m")
        print("Truth Opt V: " + str(self.searcher.opt_v))
        print("=" * 50 + "\n")
        return mpcg_test, v_test, self.searcher.opt_mpcg, self.opt_v

    def init_mpcg(self):
        if self.args.mpcg_init_search:
            self.opt_mpcg = self.searcher.ternary_search_init_mpcg(search_round=1)
        else:
            self.searcher.prepare_data()
        self.searcher.start_synchronize_load_data_in_background(data_size=12000)

    def train(self):
        self.init_mpcg()

        for step in tqdm(range(self.env_steps), desc="Training MPCG", ncols=100):
            a_loss, c_loss = self.update_networks()

            if step % int(self.args.mpcg_log_time / 5) == 0 and step != 0:
                time.sleep(self.args.mpcg_sleep_time)

            if step % self.args.mpcg_log_time == 0 and step != 0:
                mpcg_test, v_test, mpcg_tensor_test = self.validate()

                if v_test > self.searcher.opt_v:
                    self.searcher.buffer.push(mpcg_tensor_test, v_test)

                mpcg_test = '\n'.join([f'{key}: {value}' for key, value in mpcg_test.items()])
                tqdm.write(f"\n{'-' * 50}\n"
                           f"Step: {step}\n"
                           f"Actor Loss: {a_loss:.4f} | Critic Loss: {c_loss:.4f}\n"
                           f"Org V: {self.searcher.org_v:.4f} | Truth Opt V: {self.searcher.opt_v:.4f} | Actor Opt V: {v_test:.4f}\n"
                           f"Buffer Size: {self.searcher.buffer.size()}\n"
                           f"{'-' * 50}\n"
                           f"Actor Opt MPCG:"
                           f"{str(mpcg_test)}\n")
                self.change_global_noise()


            if step % (self.env_steps / 5) == 0 and step != 0:
                self.get_result()
