import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import time
import threading
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

from backbones.cpn_replayer import ReplayBuffer


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

def get_cell_inputs(args, cell_input, cpn_result, source, device):
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
            inputs_group = cpn_result.get(input_id)
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

def sample_by_cpn(args, cpn, model, source, device):
    sample_sequence = topological_sort(cpn)
    cpn_result = {key: None for key in cpn}
    with torch.no_grad():
        for cpn_id in sample_sequence:
            cell_info = cpn[cpn_id] # id : [time_layer, [input_ids], pulses, capacity]
            input_x0, input_xt = get_cell_inputs(args, cell_info[1], cpn_result, source, device)
            if cpn_id == 0.0:
                cpn_result[0.0] = [get_cell_pulses(input_x0, input_xt, cell_info[2], cell_info[3], args.pulses_type, device), input_xt]
                continue
            cell_time = torch.full((input_x0.size(0),), cell_info[0], dtype=torch.int64).to(device)
            cell_latent = torch.randn(input_x0.size(0), args.z_emb_dim, device=device)
            if args.sample_fixed:
                cell_latent = torch.zeros(input_x0.size(0), args.z_emb_dim, device=device)
            h_t = get_cell_pulses(input_x0, input_xt, cell_info[2], cell_info[3], args.pulses_type, device)
            output_x0, _ = model(torch.cat((h_t, source),axis=1), cell_time, cell_latent)
            output_xt = h_t
            cpn_result[cpn_id] = [output_x0, output_xt]
    result = cpn_result.get(0.0)
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

def cpn_to_tensor(cpn, dtype=torch.float64, device=torch.device("cuda:0")):
    pulses = []
    capacity = []
    for key in sorted(cpn.keys()):
        if key >= 1.0 or key <= 0.0:
            continue
        values = cpn[key]
        pulses.append(values[2])
        capacity.append(values[3])
    all_tensors = torch.tensor(pulses+capacity, dtype=dtype).to(device)
    return all_tensors

def tensor_to_cpn(cpn, tensor):
    all_lists = tensor.cpu().detach().tolist()
    mid = len(all_lists) // 2

    pulses = all_lists[:mid]
    capacity = all_lists[mid:]
    count = 0
    for key in sorted(cpn.keys()):
        if key >= 1.0 or key <= 0.0:
            continue
        cpn[key][2] = pulses[count]
        cpn[key][3] = capacity[count]
        count += 1
    return cpn

def sample_one(args, dataloader, model, cpn, device):
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
            fake_sample = sample_by_cpn(args, cpn, model, source_data, device)
            psnr_list, ssim_list, mae_list = evaluate_samples(args, target_data, fake_sample)
            PSNR.extend(psnr_list)
            SSIM.extend(ssim_list)
            MAE.extend(mae_list)
    vv = sum(PSNR) / len(PSNR) + 30 * sum(SSIM) / len(SSIM)
    return vv

def print_cpn(cpn):
    cpn = "{" + ',\n'.join([f'{key}: {value}' for key, value in cpn.items()]) + "}"
    print("\033[93m" + str(cpn) + "\033[0m")

class CPNACSearcher:
    def __init__(self, args, cpn=None, model=None, dataloader=None, device=torch.device('cuda:0')):
        self.args = args
        self.device = device
        self.dtype = torch.float64
        self.cpn = cpn
        self.model = model
        self.dataloader = dataloader
        self.buffer = None
        self.opt_state = None
        self.opt_cpn = None
        self.change_global_noise()
        self.org_v = sample_one(self.args, self.dataloader, self.model, self.cpn, self.device)
        self.opt_v = self.org_v

    def change_global_noise(self):
        if self.args.sample_fixed:
            global global_fixed_noise
            global_fixed_noise = get_fixed_noise(self.args, self.dataloader, self.device)

    def get_buffer(self, sample_capacity=60, max_capacity=120):
        self.buffer = ReplayBuffer(sample_capacity, max_capacity)
        return self.buffer

    def change_cpn_layer(self, cpn, key, item, value):
        for k in sorted(cpn.keys()):
            if round(key, 1) <= k < round(key + 0.1, 1):
                cpn[k][item] = round(value, 4)
        return cpn

    def ternary_search_layer_in_range(self, key, item, item_min=-0.5, item_max=3., eps=5e-3):
        start_time = time.time()
        left = item_min
        right = item_max
        best_cpn = self.cpn
        best_v = sample_one(self.args, self.dataloader, self.model, best_cpn, self.device)
        results = []
        items = []

        def sample_and_restore_data(cpn, key, item, new_value):
            new_cpn = self.change_cpn_layer(cpn, key, item, new_value)
            new_v = sample_one(self.args, self.dataloader, self.model, new_cpn, self.device)
            results.append(new_v)
            items.append(new_value)
            cpn_tensors = cpn_to_tensor(cpn=new_cpn, device=self.device)
            self.buffer.push(cpn_tensors, new_v)
            return new_v

        while right - left > eps:
            tmp_cpn = best_cpn
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            mid1_v = sample_and_restore_data(tmp_cpn, key, item, mid1)
            mid2_v = sample_and_restore_data(tmp_cpn, key, item, mid2)
            if mid1_v < mid2_v:
                left = mid1
            else:
                right = mid2
        best_item = (left + right) / 2.
        tmp_cpn = self.change_cpn_layer(best_cpn, key, item, best_item)
        tmp_v = sample_one(self.args, self.dataloader, self.model, tmp_cpn, self.device)
        if tmp_v < best_v:
            best_cpn = self.cpn
            best_item = self.cpn[key][item]
        else:
            best_cpn = tmp_cpn
            best_v = tmp_v

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("[Searcher]: Find the key:{}, index:{}, buffer_size:{}, best_item:{:.4f}, org_v:{:.4f}, best_v:{:.4f}, times:{:.2f} minutes".format(
            key, item, self.buffer.size(), best_item, self.org_v, best_v, elapsed_time / 60
        ))
        return best_cpn, best_v, results, items

    def ternary_search_init_cpn(self, search_round=2):
        cpn_tensors = cpn_to_tensor(cpn=self.cpn, device=self.device)
        v = sample_one(self.args, self.dataloader, self.model, self.cpn, self.device)
        self.org_v = v
        self.buffer.push(cpn_tensors, v)
        search_low = -0.5
        search_high = 3.
        for r in range(search_round):
            for n in range(10, -1, -1):
                lower = n / 10.0
                upper = round(lower + 0.1, 1)
                for key in self.cpn:
                    if int(key * 100) % 10 != 0:
                        continue
                    if lower <= key < upper:
                        if r != 0:
                            search_low = self.cpn[float(round(lower, 2))][2] - 0.3
                            search_high = self.cpn[float(round(lower, 2))][2] + 0.3
                        best_cpn, best_v, rst, itm = self.ternary_search_layer_in_range(float(round(lower, 2)), 2, search_low, search_high)
                        self.cpn = best_cpn
                        self.draw_result(itm, rst, "cpn_intensity_at_{:.2f}".format(lower), "intensity")

                        if r != 0:
                            search_low = self.cpn[float(round(lower, 2))][3] - 0.3
                            search_high = self.cpn[float(round(lower, 2))][3] + 0.3
                        best_cpn, best_v, rst, itm = self.ternary_search_layer_in_range(float(round(lower, 2)), 3, search_low, search_high)
                        self.cpn = best_cpn
                        self.draw_result(itm, rst, "cpn_capacity_at_{:.2f}".format(lower), "capacity")

            print("Init CPN:\n")
            self.opt_cpn = self.cpn
            print_cpn(self.cpn)
        return self.cpn

    def draw_result(self, items, results, title_name, item_name, plots_path="plot"):
        os.makedirs(plots_path, exist_ok=True)
        timestamp = datetime.now().strftime("_%H%M%S")
        base_filename = os.path.join(plots_path, f"{title_name}.png")
        combined = list(zip(items, results))
        combined_sorted = sorted(combined, key=lambda x: x[0])
        sorted_items, sorted_results = zip(*combined_sorted)
        items = list(sorted_items)
        results = list(sorted_results)
        if os.path.exists(base_filename):
            filename = os.path.join(plots_path, f"{title_name}{timestamp}.png")
        else:
            filename = base_filename
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(items, results, marker='o', markersize=8, linewidth=2,
                        color='#2b7bba', markerfacecolor='#ff7f0e', markeredgewidth=1, alpha=0.8)
        plt.title(title_name, fontsize=16, pad=20)
        plt.xlabel(item_name, fontsize=12, labelpad=10)
        plt.ylabel("Results", fontsize=12, labelpad=10)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        max_index = results.index(max(results))
        max_value = results[max_index]
        max_item = items[max_index]
        plt.axvline(x=max_item, color='red', linestyle='--', linewidth=1)
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


    def prepare_data(self, data_size=100):
        cpn_tensors = cpn_to_tensor(cpn=self.cpn, device=self.device)
        n_v = sample_one(self.args, self.dataloader, self.model, self.cpn, self.device)
        self.org_v = n_v
        self.buffer.push(cpn_tensors, n_v)
        for _ in tqdm(range(data_size), desc="Preparing data..."):
            states, vs = self.buffer.sample_elite(1)
            cpn_tensors = states[0] + torch.normal(0.0, 0.08, size=states[0].size(), device=self.device)
            cpn_tensors = torch.clamp(cpn_tensors, 0.0, 1.5)
            cpn_noise = tensor_to_cpn(self.cpn, cpn_tensors)
            n_v = sample_one(self.args, self.dataloader, self.model, cpn_noise, self.device)
            if n_v > self.opt_v:
                self.opt_v = n_v
                self.opt_cpn = cpn_noise
                self.opt_state = cpn_tensors
            self.buffer.push(cpn_tensors, n_v)

    def synchronize_load_data(self, args, cpn, dataloader, model, data_size, device):
        for _ in range(data_size):
            states, vs = self.buffer.sample_elite(1)
            cpn_tensors = states[0] + torch.normal(0.0, 0.05, size=states[0].size(), device=device)
            cpn_tensors = torch.clamp(cpn_tensors, 0.0, 1.0)
            cpn_noise = tensor_to_cpn(cpn, cpn_tensors)
            v = sample_one(args, dataloader, model, cpn_noise, device)
            if v > self.opt_v:
                self.opt_v = v
                self.opt_cpn = cpn_noise
                self.opt_state = cpn_tensors
            self.buffer.push(cpn_tensors, v)

    def start_synchronize_load_data_in_background(self, data_size=14000):
        try:
            thread = threading.Thread(target=self.synchronize_load_data,
                                      args=(self.args, self.cpn, self.dataloader, self.model, data_size, self.device))
            thread.daemon = True
            thread.start()
        except Exception as e:
            print(f"Error starting thread: {e}")


