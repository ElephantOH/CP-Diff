import argparse
import copy

import torch
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Subset
from tqdm import tqdm
import config
from backbones.double_flow_model import DoubleFlowModel
from backbones.NSCNpp.ncsnpp_generator_adagn import NCSNpp
from dataset import GetDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
        
#%% Diffusion coefficients
def extract(num, shape, device):
    num = torch.tensor([num] * shape[0]).to(device)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    num = num.reshape(*reshape)
    return num

def load_checkpoint(checkpoint_dir, mapping_network, name_of_network, epoch, device = 'cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)
    ckpt = torch.load(checkpoint_file, map_location=device)
    for key in list(ckpt.keys()):
         ckpt[key[7:]] = ckpt.pop(key)
    mapping_network.load_state_dict(ckpt)
    mapping_network.eval()

global_fixed_noise = None

def get_fixed_noise(dataloader, device):
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

def evaluate_samples(real_data, fake_sample):
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

def get_mpcg_dim(mpcg):
    mpcg_dim = 0
    hidden_dim = 256
    for key in sorted(mpcg.keys()):
        if key >= 1.0 or key <= 0.0:
            continue
        mpcg_dim += 2
    if mpcg_dim > 12:
        hidden_dim = 384
    return mpcg_dim, hidden_dim

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

def get_cell_inputs(cell_input, mpcg_result, source, device):
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
    if inputs_x0 and len(inputs_x0) != 1:
        input_x0 = torch.stack(inputs_x0)
        input_x0 = torch.mean(input_x0, dim=0).to(device)
        input_xt = torch.stack(inputs_xt)
        input_xt = torch.mean(input_xt, dim=0).to(device)
    elif inputs_xt and len(inputs_x0) == 1:
        input_x0 = inputs_x0[0]
        input_xt = inputs_xt[0]
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

def sample_by_mpcg(mpcg, model, source, device):
    sample_sequence = topological_sort(mpcg)
    mpcg_result = {key: None for key in mpcg}
    with torch.no_grad():
        for mpcg_id in sample_sequence:
            cell_info = mpcg[mpcg_id] # id : [time_layer, [input_ids], intensity, capacity]
            input_x0, input_xt = get_cell_inputs(cell_info[1], mpcg_result, source, device)
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

def sample_one(dataloader, model, mpcg, device):
    PSNR = []
    SSIM = []
    MAE = []
    progress_bar = tqdm(dataloader, desc="Processing", colour="green")
    with torch.no_grad():
        for iteration, (source_data, target_data, _, _) in enumerate(progress_bar):
            target_data = target_data.to(device, non_blocking=True)
            source_data = source_data.to(device, non_blocking=True)
            if args.input_channels == 3:
                target_data = target_data.squeeze(1)
                source_data = source_data.squeeze(1)
            fake_sample = sample_by_mpcg(mpcg, model, source_data, device)
            psnr_list, ssim_list, mae_list = evaluate_samples(target_data, fake_sample)
            PSNR.extend(psnr_list)
            SSIM.extend(ssim_list)
            MAE.extend(mae_list)
            progress_bar.set_postfix(PSNR=sum(PSNR) / len(PSNR), SSIM=sum(SSIM) / len(SSIM), MAE=sum(MAE) / len(MAE))
            # tqdm.write(f"[{iteration}/{len(dataloader)}] PSNR: {psnr_list[0]}, SSIM: {ssim_list[0]}, MAE: {mae_list[0]}")
    vv = sum(PSNR) / len(PSNR) + 30 * sum(SSIM) / len(SSIM)
    return vv

def sample_only_one(dataloader, model, mpcg, times, device):
    PSNR = []
    SSIM = []
    MAE = []
    FAKE = []

    for i in range(times):
        print(f"[{i}|{times}]")
        with torch.no_grad():
            for iteration, (source_data, target_data, _, _) in enumerate(dataloader):
                target_data = target_data.to(device, non_blocking=True)
                source_data = source_data.to(device, non_blocking=True)
                if args.input_channels == 3:
                    target_data = target_data.squeeze(1)
                    source_data = source_data.squeeze(1)
                fake_sample = sample_by_mpcg(mpcg, model, source_data, device)
                psnr_list, ssim_list, mae_list = evaluate_samples(target_data, fake_sample)
                PSNR.append(psnr_list[0])
                SSIM.append(ssim_list[0])
                MAE.append(mae_list[0])
                FAKE.append(fake_sample)

    def save_image(img, names, input_channels):
        file_path = '{}.png'.format(names)
        if input_channels == 1:
            to_range_0_1 = lambda x: (x + 1.) / 2.
            img = to_range_0_1(img)
            torchvision.utils.save_image(img, file_path)
        elif input_channels == 3:
            img = img.permute(1, 2, 0).cpu().numpy()
            img = (img * 127.5 + 127.5).astype(np.uint8)[..., [2, 1, 0]]
            image = Image.fromarray(img)
            image.save(file_path)

    return PSNR, SSIM, MAE

def refresh_cell_dependent(mpcg):
    for key, value in mpcg.items():
        if round(key, 2) >= 1.0:
            continue

        layer = round(int(key * 10) / 10, 2)
        dep = []
        dep_count = 0
        for key1, value1 in mpcg.items():
            if round(layer + 0.1 , 2) <= key1 and round(layer + 0.2 , 2) > key1:
                dep.append(key1)
                dep_count += 1

        if dep_count == 0:
            for key1, value1 in mpcg.items():
                if round(1.00, 2) <= key1:
                    dep.append(key1)
                    dep_count += 1

        mpcg[key][1] = dep
    return mpcg

def modify_cell_by_copy(mpcg, layer_id, op_id):
    if op_id == 1:
        original_key = layer_id
        if original_key not in mpcg:
            print("[warning]: key not found!")
            return mpcg
        new_key = round(original_key + 0.01, 2)
        while new_key in mpcg:
            new_key = round(new_key + 0.01, 2)
        mpcg[new_key] = copy.deepcopy(mpcg[original_key])
    elif op_id == -1:
        original_key = layer_id
        upper = round(original_key + 0.1, 2)
        candidates = []
        for k in mpcg:
            if original_key < k < upper:
                k_rounded = round(k, 2)
                second_decimal = int(round(k_rounded * 100)) % 10
                if second_decimal != 0:
                    candidates.append(k)
        if candidates:
            max_key = max(candidates)
            del mpcg[max_key]
    else:
        return mpcg
    return mpcg

def modify_cell_by_layer(mpcg, layer_num=[1, 1, 1, 1, 1], max_layer_num=5):
    modify_mpcg = copy.deepcopy(mpcg)
    for i in range(len(layer_num)):
        if layer_num[i] < 1 or layer_num[i] > max_layer_num or i == 0:
            continue
        elif layer_num[i] > 1:
            if i == len(layer_num) - 1:
                layer_id = 1.00
            else:
                layer_id = round(i / 10, 2)
            for j in range(layer_num[i]-1):
                modify_mpcg = modify_cell_by_copy(modify_mpcg, layer_id, 1)
    modify_mpcg = refresh_cell_dependent(modify_mpcg)
    return modify_mpcg


#%% MAIN FUNCTION
def train_mpcg(args):
    torch.manual_seed(42)
    torch.cuda.set_device(args.gpu_chose)
    device = torch.device('cuda:{}'.format(args.gpu_chose))
    args.phase = "train_mpcg"
    args.sample_fixed = True
    train_dataset = GetDataset(args.phase, args.input_path, args.source, args.target, dim=args.input_channels, normed=args.normed)
    mpcg_dataset_indices = np.random.choice(len(train_dataset), args.mpcg_dataset_num, replace=False)
    mpcg_dataset = Subset(train_dataset, mpcg_dataset_indices)
    mpcg_dataloader = torch.utils.data.DataLoader(mpcg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.use_model_name == "dfm":
        model = DoubleFlowModel(args).to(device)
    else:
        model = NCSNpp(args).to(device)

    mpcg_name = "mpcg_" + args.mpcg_name
    mpcg = getattr(args, mpcg_name)

    args.use_model_name = "DFM"
    checkpoint_file = args.checkpoint_path + "/{}_{}.pth"
    load_checkpoint(checkpoint_file, model, '{}_{}'.format(args.network_type, args.use_model_name), epoch=str(args.which_epoch), device=device)

    # get_nr(mpcg_dataloader, model, 500, device)
    # exit(0)
    # get_ablation_on_mpcg_framework(mpcg_dataloader, model, mpcg)
    # exit(0)
    # get_photo(mpcg_dataloader, model, 1000, device)
    # exit(0)
    #######################################################
    from backbones.mpcg_actor_critic import mpcgACTrainer
    mpcg_dim, hidden_dim = get_mpcg_dim(mpcg)
    mpcgac = mpcgACTrainer(args, mpcg_dim=mpcg_dim, hidden_dim=hidden_dim, env_step=args.mpcg_env_step,
                         mpcg=mpcg, model=model, dataloader=mpcg_dataloader, device=device,
                         mpcg_lr=1e-4, mpcg_lrf=1e-5, batch_size=args.mpcg_batch_size)
    mpcgac.train()
    a_mpcg, a_v, t_mpcg, t_v = mpcgac.get_result()
    #######################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser('cell diffusion parameters')
    parser = config.load_config(parser)
    args = parser.parse_args()

    if args.network_type == 'normal':
        print("Using normal network configuration.")
    elif args.network_type == 'large':
        print("Using large network configuration.")
        args.num_channels_dae = 128
        args.num_res_blocks = 3
    elif args.network_type == 'max':
        print("Using max network configuration.")
        args.num_channels_dae = 128
        args.num_res_blocks = 4
        args.ch_mult = [1, 1, 2, 2, 4, 8]
    else:
        print(f"Unknown network type: {args.network_type}")
    
    train_mpcg(args)
    
