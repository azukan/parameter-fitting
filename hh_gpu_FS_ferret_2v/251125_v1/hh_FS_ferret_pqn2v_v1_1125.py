'''

'''

import numpy as np
import torch
import time
from collections import defaultdict
from typing import Tuple
import random
from openbox import space as spc
from openbox import Optimizer

# 设置随机种子
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f'seed = {SEED}')

neuron_data_1 = {0: {0: {'spike_time': [],
   'spike_amp': [],
   'spike_num': 0,
   'spike_reset': [],
   'width': []}},
 1: {0: {'spike_time': [],
   'spike_amp': [],
   'spike_num': 0,
   'spike_reset': [],
   'width': []}},
 2: {0: {'spike_time': [],
   'spike_amp': [],
   'spike_num': 0,
   'spike_reset': [],
   'width': []}},
 3: {0: {'spike_time': [],
   'spike_amp': [],
   'spike_num': 0,
   'spike_reset': [],
   'width': []}},
 4: {0: {'spike_time': [],
   'spike_amp': [],
   'spike_num': 0,
   'spike_reset': [],
   'width': []}},
 5: {0: {'spike_time': [0.011099999770522118],
   'spike_amp': [4.878310680389404],
   'spike_num': 1,
   'spike_reset': [-7.889039993286133],
   'width': [10]}},
 6: {0: {'spike_time': [0.007449999917298555],
   'spike_amp': [4.977084159851074],
   'spike_num': 1,
   'spike_reset': [-7.880059719085693],
   'width': [11]}},
 7: {0: {'spike_time': [0.006899999920278788],
   'spike_amp': [4.856441020965576],
   'spike_num': 1,
   'spike_reset': [-7.875856876373291],
   'width': [10]}},
 8: {0: {'spike_time': [0.0066999997943639755],
   'spike_amp': [4.865834712982178],
   'spike_num': 1,
   'spike_reset': [-7.8732500076293945],
   'width': [10]}},
 9: {0: {'spike_time': [0.0064999996684491634],
   'spike_amp': [4.885478496551514],
   'spike_num': 1,
   'spike_reset': [-7.871454238891602],
   'width': [10]}}
                 }

neuron_data_2 = {0: {0: {'spike_time': [],
   'spike_amp': [],
   'spike_num': 0,
   'spike_reset': [],
   'width': []}},
 1: {0: {'spike_time': [0.04794999957084656,
    0.10159999877214432,
    0.15524999797344208,
    0.20889998972415924,
    0.2625499963760376,
    0.31619998812675476,
    0.3698499798774719,
    0.4234499931335449,
    0.4770999848842621],
   'spike_amp': [4.9302825927734375,
    4.956626892089844,
    4.953152656555176,
    4.926117897033691,
    4.891073703765869,
    4.853866100311279,
    4.817699909210205,
    4.864701271057129,
    4.928016662597656],
   'spike_num': 9,
   'spike_reset': [-7.786050319671631,
    -7.785863399505615,
    -7.785791873931885,
    -7.786075592041016,
    -7.786284446716309,
    -7.786388874053955,
    -7.786365509033203,
    -7.786255359649658,
    -7.786060333251953],
   'width': [10, 10, 10, 10, 10, 10, 10, 10, 10]}},
 2: {0: {'spike_time': [0.02669999934732914,
    0.05869999900460243,
    0.09064999967813492,
    0.12264999747276306,
    0.1546500027179718,
    0.18664999306201935,
    0.2186499983072281,
    0.25064998865127563,
    0.2826499938964844],
   'spike_amp': [4.864821910858154,
    4.835944175720215,
    4.820933818817139,
    4.8881378173828125,
    4.9344868659973145,
    4.959140777587891,
    4.961730480194092,
    4.945840835571289,
    4.918920040130615],
   'spike_num': 9,
   'spike_reset': [-7.774840354919434,
    -7.774787425994873,
    -7.774669647216797,
    -7.774484157562256,
    -7.77426290512085,
    -7.774324893951416,
    -7.774431228637695,
    -7.774578094482422,
    -7.774723529815674],
   'width': [10, 10, 10, 10, 10, 10, 10, 10, 10]}},
 3: {0: {'spike_time': [0.01979999989271164,
    0.04454999789595604,
    0.0693499967455864,
    0.09414999932050705,
    0.11889999359846115,
    0.1437000036239624,
    0.16849999129772186,
    0.19325000047683716,
    0.21804998815059662],
   'spike_amp': [4.84985876083374,
    4.937242031097412,
    4.951138973236084,
    4.852441787719727,
    4.936573028564453,
    4.95162296295166,
    4.85310173034668,
    4.935263156890869,
    4.952304840087891],
   'spike_num': 9,
   'spike_reset': [-7.763227939605713,
    -7.763071537017822,
    -7.763154029846191,
    -7.763182163238525,
    -7.763072967529297,
    -7.763153076171875,
    -7.763185977935791,
    -7.763071537017822,
    -7.763151168823242],
   'width': [10, 10, 10, 10, 10, 10, 10, 10, 10]}},
 4: {0: {'spike_time': [0.006799999624490738,
    0.0170499999076128,
    0.027249999344348907,
    0.03749999776482582,
    0.04769999906420708,
    0.05794999748468399,
    0.06814999878406525,
    0.07840000092983246,
    0.08859999477863312,
    0.09884999692440033],
   'spike_amp': [5.0011515617370605,
    4.954675674438477,
    4.843955993652344,
    4.94590950012207,
    4.879786491394043,
    4.934409141540527,
    4.908371448516846,
    4.920933723449707,
    4.930314064025879,
    4.90631628036499],
   'spike_num': 10,
   'spike_reset': [-7.655707836151123,
    -7.649722099304199,
    -7.649397850036621,
    -7.649727821350098,
    -7.649230003356934,
    -7.649791240692139,
    -7.649198055267334,
    -7.649837493896484,
    -7.6492600440979,
    -7.8396477699279785],
   'width': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}}}


@torch.jit.script
def euler_2v_gpu_scripted(
        V: torch.Tensor,
        N: torch.Tensor,
        P: torch.Tensor,
        I_true: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 将 I_true 扩展最后一个维度: (T, protocols, 1)
    I_true = I_true.unsqueeze(-1)

    # 从 P 中提取参数，每个参数 reshape 为 (1, num_particles)
    a_fn, a_fp, b_fn, c_fn, a_gn, a_gp, b_gn, c_gn, r_g, phi, tau, k = [
        P[:, i].view(1, -1) for i in range(P.shape[1])
    ]

    dt: float = 0.00005

    b_fp = a_fn * b_fn / a_fp
    c_fp = a_fn * b_fn * b_fn + c_fn - a_fp * b_fp * b_fp
    b_gp = r_g - a_gn * (r_g - b_gn) / a_gp
    c_gp = a_gn * (r_g - b_gn) * (r_g - b_gn) - a_gp * (r_g - b_gp) * (r_g - b_gp) + c_gn

    active_mask = torch.ones((V.size(1), V.size(2)), dtype=torch.bool, device=V.device)
    T = V.size(0)

    for t in range(T - 1):
        v = V[t]
        n = N[t]
        I_stim = I_true[t] * k * 10.0

        fv = torch.where(v < 0.0, a_fn * (v - b_fn) * (v - b_fn) + c_fn,
                         a_fp * (v - b_fp) * (v - b_fp) + c_fp)
        gv = torch.where(v < r_g, a_gn * (v - b_gn) * (v - b_gn) + c_gn,
                         a_gp * (v - b_gp) * (v - b_gp) + c_gp)

        dv = phi * (fv - n + 0 + I_stim) / tau * dt
        dn = (gv - n) / tau * dt

        new_v = torch.where(active_mask, v + dv, v)
        new_n = torch.where(active_mask, n + dn, n)

        V[t + 1] = new_v
        N[t + 1] = new_n

        exceeded_mask = (new_v.abs() >= 200.0) | (new_n.abs() >= 200.0)
        active_mask = active_mask & (~exceeded_mask)

    return V, N


def compute_thresholds(Vt):
    """
    根据膜电位数据计算阈值。
    Vt: [time_steps, protocols, neurons]
    返回：thresholds: [protocols, neurons]
    """
    Vmax = torch.max(Vt, dim=0).values
    Vmin = torch.min(Vt, dim=0).values
    thresholds = Vmax - (Vmax - Vmin) / 3
    thresholds = torch.clamp(thresholds, min=0.5)
    return thresholds


def detect_boundaries(over_threshold):
    """
    检测从低到高（start）和从高到低（end）的阈值过渡点。
    over_threshold: [time_steps, protocols, neurons] 的布尔张量
    返回：starts_idx, ends_idx 分别为起始和终止点的非零索引元组 (time, protocol, neuron)
    """
    starts = (over_threshold[1:] & ~over_threshold[:-1])
    ends = (~over_threshold[1:] & over_threshold[:-1])
    starts_idx = torch.nonzero(starts, as_tuple=True)
    ends_idx = torch.nonzero(ends, as_tuple=True)
    return starts_idx, ends_idx


@torch.jit.script
def detect_peaks_scripted(Vt, thresholds):
    """
    编译后的峰值检测函数。
    参数：
      Vt: (time_steps, protocols, neurons) 的电位数据张量
      thresholds: (protocols, neurons) 的阈值张量
    返回：
      peaks: (time_steps, protocols, neurons) 的布尔张量，标记峰值位置
    """
    potential = Vt[1:-1] > thresholds.unsqueeze(0)
    ascending = Vt[1:-1] >= Vt[:-2]
    descending = Vt[1:-1] > Vt[2:]
    potential_peaks = ascending & descending & potential

    peaks_prev = torch.zeros_like(potential_peaks, dtype=torch.bool)
    below_threshold_state = torch.ones(Vt.size(1), Vt.size(2), dtype=torch.bool, device=Vt.device)

    for t in range(potential_peaks.size(0)):
        below_threshold_state = below_threshold_state | (Vt[t + 1] < thresholds)
        peaks_prev[t] = potential_peaks[t] & below_threshold_state
        below_threshold_state = below_threshold_state & (~peaks_prev[t])

    peaks = torch.cat((torch.zeros(1, Vt.size(1), Vt.size(2), dtype=torch.bool, device=Vt.device), peaks_prev), dim=0)
    return peaks


def detect_resets(Vt, thresholds):
    """
    检测重置值（spike resets）
    Vt: [time_steps, protocols, neurons]
    thresholds: [protocols, neurons]
    返回：resets: [time_steps, protocols, neurons] 的布尔张量，表示重置值位置
    """
    below_threshold = Vt[1:-1] < thresholds.unsqueeze(0)
    lower_than_previous = Vt[1:-1] <= Vt[:-2]
    lower_than_next = Vt[1:-1] < Vt[2:]
    potential_resets = below_threshold & lower_than_previous & lower_than_next
    resets_prev = torch.zeros_like(potential_resets, dtype=torch.bool)
    above_threshold_state = torch.zeros(Vt.shape[1:], dtype=torch.bool, device=Vt.device)
    for t in range(potential_resets.shape[0]):
        above_threshold_state |= (Vt[t + 1] > thresholds)
        resets_prev[t] = potential_resets[t] & above_threshold_state
        above_threshold_state &= ~resets_prev[t]
    resets = torch.cat(
        (torch.zeros((1, *resets_prev.shape[1:]), dtype=resets_prev.dtype, device=resets_prev.device), resets_prev),
        dim=0)
    return resets


def map_events(Vt, events_mask):
    """
    将事件（peaks或resets）映射到 Vt 上，获得对应的电位值。
    Vt: [time_steps, protocols, neurons]
    events_mask: [time_steps, protocols, neurons] 的布尔张量
    返回：nonzero_indices, event_values
    """
    mapped_values = Vt[:-1] * events_mask[:]
    indices = torch.nonzero(mapped_values, as_tuple=True)
    return indices, mapped_values[indices]


#########################################
# 以下部分尽量减少 CPU–GPU 数据传输
#########################################

def build_result_structure(num_protocols, num_neurons):
    result = defaultdict(lambda: defaultdict(lambda: {
        "spike_time": [],
        "spike_amp": [],
        "spike_num": 0,
        "spike_reset": [],
        "width": []
    }))
    for p in range(num_protocols):
        for n in range(num_neurons):
            _ = result[p][n]
    return result


def fill_spike_data(result, times, protocols, neurons, amplitudes):
    # times, protocols, neurons, amplitudes 均为 GPU 张量
    pn_pairs = torch.stack((protocols, neurons), dim=1)  # [N, 2]
    unique_pairs, inverse_indices, counts = torch.unique(pn_pairs, dim=0, return_inverse=True, return_counts=True)
    sorted_inv, sort_idx = torch.sort(inverse_indices, stable=True)
    times_sorted = times[sort_idx]
    amps_sorted = amplitudes[sort_idx]
    cum_counts = torch.cat((torch.tensor([0], device=counts.device, dtype=counts.dtype), counts.cumsum(0)))
    # 一次性将小张量转为 CPU 数组
    unique_pairs_cpu = unique_pairs.cpu().numpy()
    counts_cpu = counts.cpu().numpy()
    cum_counts_cpu = cum_counts.cpu().numpy()
    times_sorted_cpu = times_sorted.cpu().numpy()
    amps_sorted_cpu = amps_sorted.cpu().numpy()
    max_spikes = 20
    for i, (p, n) in enumerate(unique_pairs_cpu):
        start_idx = cum_counts_cpu[i]
        end_idx = cum_counts_cpu[i + 1]
        group_size = end_idx - start_idx
        limit = min(group_size, max_spikes)
        if limit > 0:
            selected_times = times_sorted_cpu[start_idx:start_idx + limit].tolist()
            selected_amps = amps_sorted_cpu[start_idx:start_idx + limit].tolist()
            result[int(p)][int(n)]["spike_time"].extend(selected_times)
            result[int(p)][int(n)]["spike_amp"].extend(selected_amps)
            result[int(p)][int(n)]["spike_num"] += int(group_size)


def fill_reset_data(result, protocols_reset, neurons_reset, reset_value):
    pn_pairs = torch.stack((protocols_reset, neurons_reset), dim=1)  # [N, 2]
    unique_pairs, inverse_indices, counts = torch.unique(pn_pairs, dim=0, return_inverse=True, return_counts=True)
    sorted_inv, sort_idx = torch.sort(inverse_indices, stable=True)
    reset_sorted = reset_value[sort_idx]
    cum_counts = torch.cat((torch.tensor([0], device=counts.device, dtype=counts.dtype), counts.cumsum(0)))
    unique_pairs_cpu = unique_pairs.cpu().numpy()
    cum_counts_cpu = cum_counts.cpu().numpy()
    reset_sorted_cpu = reset_sorted.cpu().numpy()
    max_records = 20
    for i, (p, n) in enumerate(unique_pairs_cpu):
        start_idx = cum_counts_cpu[i]
        end_idx = cum_counts_cpu[i + 1]
        group_size = end_idx - start_idx
        limit = min(group_size, max_records)
        if limit > 0:
            selected_resets = reset_sorted_cpu[start_idx:start_idx + limit].tolist()
            result[int(p)][int(n)]["spike_reset"].extend(selected_resets)


def fill_width_data(result, starts_idx, ends_idx):
    # starts_idx, ends_idx 均为 GPU 张量，先转换为 CPU 数组一次性获取
    s_t = starts_idx[0].cpu().numpy()
    s_p = starts_idx[1].cpu().numpy()
    s_n = starts_idx[2].cpu().numpy()
    e_t = ends_idx[0].cpu().numpy()
    e_p = ends_idx[1].cpu().numpy()
    e_n = ends_idx[2].cpu().numpy()
    # 构造 numpy 数组
    starts_data = np.stack((s_p, s_n, s_t), axis=1)  # [Ns, 3]
    ends_data = np.stack((e_p, e_n, e_t), axis=1)  # [Ne, 3]
    # 对 starts_data 进行 lex 排序：先按 p，再按 n，最后按 t
    starts_data = starts_data[np.lexsort((starts_data[:, 2], starts_data[:, 1], starts_data[:, 0]))]
    ends_data = ends_data[np.lexsort((ends_data[:, 2], ends_data[:, 1], ends_data[:, 0]))]
    # 分组统计：利用 numpy 的 unique
    unique_s_pn, s_indices, s_counts = np.unique(starts_data[:, :2], axis=0, return_index=True, return_counts=True)
    unique_e_pn, e_indices, e_counts = np.unique(ends_data[:, :2], axis=0, return_index=True, return_counts=True)
    s_cum = np.concatenate(([0], np.cumsum(s_counts)))
    e_cum = np.concatenate(([0], np.cumsum(e_counts)))
    e_map = {}
    unique_e_pn_list = unique_e_pn.tolist()
    for i, pair in enumerate(unique_e_pn_list):
        e_map[tuple(pair)] = (e_cum[i], e_cum[i + 1])
    max_starts = 80
    max_ends = 80
    max_width = 20
    for i, pair in enumerate(unique_s_pn.tolist()):
        p, n = pair
        s_start = s_cum[i]
        s_end = s_cum[i + 1]
        s_group = starts_data[s_start:s_end]
        s_group = s_group[:max_starts]
        s_times = s_group[:, 2]
        if (p, n) not in e_map:
            continue
        e_start, e_end = e_map[(p, n)]
        e_group = ends_data[e_start:e_end]
        e_group = e_group[:max_ends]
        e_times = e_group[:, 2]
        idxs = np.searchsorted(e_times, s_times, side='right')
        matched_widths = []
        for st, idx in zip(s_times, idxs):
            if idx < len(e_times):
                width = e_times[idx] - st
                matched_widths.append(width)
                if len(matched_widths) >= max_width:
                    break
        if len(matched_widths) > 0:
            result[int(p)][int(n)]["width"].extend(matched_widths)


def convert_to_regular_dict(result):
    return {
        p: {
            n: {
                "spike_time": data["spike_time"],
                "spike_amp": data["spike_amp"],
                "spike_num": data["spike_num"],
                "spike_reset": data["spike_reset"],
                "width": data["width"]
            }
            for n, data in neuron_data.items()
        }
        for p, neuron_data in result.items()
    }


def spike_detect(V_init, dt=0.00005):
    Vt = V_init[:, :, :]
    thresholds = compute_thresholds(Vt)
    over_threshold = Vt > thresholds.unsqueeze(0)
    starts_idx, ends_idx = detect_boundaries(over_threshold)
    peaks = detect_peaks_scripted(Vt, thresholds)
    resets = detect_resets(Vt, thresholds)
    (peak_times_idx, peak_protocols_idx, peak_neurons_idx), peak_values = map_events(Vt, peaks)
    (reset_times_idx, reset_protocols_idx, reset_neurons_idx), reset_values = map_events(Vt, resets)
    times = peak_times_idx.float() * dt
    _, num_protocols, num_neurons = Vt.shape
    result = build_result_structure(num_protocols, num_neurons)
    fill_spike_data(result, times, peak_protocols_idx, peak_neurons_idx, peak_values)
    fill_reset_data(result, reset_protocols_idx, reset_neurons_idx, reset_values)
    fill_width_data(result, starts_idx, ends_idx)
    result = convert_to_regular_dict(result)
    return result


def spike_num_detect(V_init):
    Vt = V_init[:, :, :]
    thresholds = compute_thresholds(Vt)
    peaks = detect_peaks_scripted(Vt, thresholds)
    (peak_times_idx, peak_protocols_idx, peak_neurons_idx), peak_values = map_events(Vt, peaks)
    _, num_protocols, num_neurons = Vt.shape
    pn_pairs = torch.stack((peak_protocols_idx, peak_neurons_idx), dim=1)
    unique_pairs, inverse_indices, counts = torch.unique(pn_pairs, dim=0, return_inverse=True, return_counts=True)
    spike_counts = torch.zeros((num_protocols, num_neurons), dtype=torch.int64, device='cuda')
    spike_counts[unique_pairs[:, 0], unique_pairs[:, 1]] = counts
    return spike_counts


# a_fn, a_fp, b_fn, c_fn,
# a_gn, a_gp, b_gn, c_gn,
# r_g, phi, tau, k
bounds = np.array([
    (0.1, 4),
    (-4, -1),
    (-5, 5),
    (-5, 5),

    (0.1, 4),
    (1, 15),
    (-5, 5),
    (-5, 5),

    (-5, 5),
    (0.1, 10),
    (0.0001, 0.1),
    (0.1, 10)
])


class ParticleSwarmOptimizer:
    def __init__(self, func, bounds, num_particles, max_iter, w=0.5, c1=1.5, c2=1.5, velocity_limit=None):
        self.func = func
        self.bounds = torch.tensor(bounds, dtype=torch.float32, device='cuda')
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        if velocity_limit is None:
            self.velocity_limit = None
        else:
            if isinstance(velocity_limit, (int, float)):
                range_per_dim = self.bounds[:, 1] - self.bounds[:, 0]
                self.velocity_limit = velocity_limit * range_per_dim
            else:
                raise ValueError("velocity_limit应为标量或None")

    def optimize(self):
        start_time = time.time()
        num_params = self.bounds.shape[0]
        self.positions = (self.bounds[:, 1] - self.bounds[:, 0]) * torch.rand((self.num_particles, num_params),
                                                                              device='cuda') + self.bounds[:, 0]
        self.velocities = torch.zeros((self.num_particles, num_params), device='cuda')
        self.personal_best_positions = self.positions.clone()
        self.personal_best_scores = torch.full((self.num_particles,), float('inf'), device='cuda')
        self.global_best_position = torch.zeros(num_params, device='cuda')
        self.global_best_score = float('inf')

        for iter in range(self.max_iter):
            scores = self.func(self.positions)
            scores = torch.where(torch.isnan(scores), torch.tensor(float('inf'), device='cuda'), scores)
            better_mask = scores < self.personal_best_scores
            self.personal_best_scores[better_mask] = scores[better_mask]
            self.personal_best_positions[better_mask] = self.positions[better_mask]
            min_score, min_idx = torch.min(scores, dim=0)
            if min_score < self.global_best_score:
                self.global_best_score = min_score
                self.global_best_position = self.positions[min_idx].clone()

            if (iter + 1) == 100:
                if self.global_best_score.item() > 20.0:
                    elapsed_time = time.time() - start_time
                    print(
                        f"[Early stop] Iter {iter + 1}/{self.max_iter}: best={self.global_best_score.item():.6f}, Time: {elapsed_time:.2f} s")
                    return self.global_best_position.cpu().numpy(), self.global_best_score.cpu()
            if (iter + 1) == 200:
                if self.global_best_score.item() > 10.0:
                    elapsed_time = time.time() - start_time
                    print(
                        f"[Early stop] Iter {iter + 1}/{self.max_iter}: best={self.global_best_score.item():.6f}, Time: {elapsed_time:.2f} s")
                    return self.global_best_position.cpu().numpy(), self.global_best_score.cpu()
            r1 = torch.rand(self.positions.shape, device='cuda')
            r2 = torch.rand(self.positions.shape, device='cuda')
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.personal_best_positions - self.positions) +
                               self.c2 * r2 * (self.global_best_position - self.positions))
            if self.velocity_limit is not None:
                self.velocities = torch.clamp(self.velocities, -self.velocity_limit, self.velocity_limit)
            self.positions += self.velocities
            self.positions = torch.clamp(self.positions, self.bounds[:, 0], self.bounds[:, 1])
            # 每 10 次迭代打印一次，减少同步
            if iter % 20 == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Iter {iter}/{self.max_iter}, Best Score: {self.global_best_score:.6f}, Time: {elapsed_time:.2f} s")
        return self.global_best_position.cpu().numpy(), self.global_best_score.cpu()


# 主程序入口
start_time = time.time()

I_hh1 = torch.zeros((2000, 10), device='cuda', dtype=torch.float32)
for i in range(10):
    I_hh1[100:120, i] = 0.086 + i * 0.0172

I_hh2 = torch.zeros((10000, 5), device='cuda', dtype=torch.float32)
I_hh2[:, 0] = 0.01
I_hh2[:, 1] = 0.016
I_hh2[:6000, 2] = 0.018
I_hh2[:4500, 3] = 0.02
I_hh2[:2000, 4] = 0.04

time_steps = 10000
num_neurons = 5000
I_true_zero = torch.zeros((20000, 1), device='cuda', dtype=torch.float32)


# I_hh1, I_hh2 已在 GPU 上

def objective(params_batch):
    # 在积分和状态更新中使用 AMP 混合精度加速
    with torch.amp.autocast('cuda'):
        V_zero = torch.zeros((20000, 1, num_neurons), device='cuda', dtype=torch.float32)
        N_zero = torch.zeros((20000, 1, num_neurons), device='cuda', dtype=torch.float32)
        V_init_1 = torch.zeros((2000, 10, num_neurons), device='cuda', dtype=torch.float32)
        N_init_1 = torch.zeros((2000, 10, num_neurons), device='cuda', dtype=torch.float32)
        V_init_2 = torch.zeros((time_steps, 5, num_neurons), device='cuda', dtype=torch.float32)
        N_init_2 = torch.zeros((time_steps, 5, num_neurons), device='cuda', dtype=torch.float32)

        V_zero_out, N_zero_out = euler_2v_gpu_scripted(V_zero, N_zero, params_batch, I_true_zero)
        V_init_1[0] = V_zero_out[-1, 0, :].unsqueeze(0).repeat(10, 1)
        N_init_1[0] = N_zero_out[-1, 0, :].unsqueeze(0).repeat(10, 1)
        V_init_2[0] = V_zero_out[-1, 0, :].unsqueeze(0).repeat(5, 1)
        N_init_2[0] = N_zero_out[-1, 0, :].unsqueeze(0).repeat(5, 1)

        V_short, _ = euler_2v_gpu_scripted(V_init_1, N_init_1, params_batch, I_hh1)
        V_long, _ = euler_2v_gpu_scripted(V_init_2, N_init_2, params_batch, I_hh2)

        short_results = spike_detect(V_short)
        fit_results = spike_detect(V_long)

    neuron_errors = {}
    for protocol, neuron_data_per_protocol in neuron_data_2.items():
        reference = neuron_data_per_protocol[0]
        fit_data_per_protocol = fit_results[protocol]
        for neuron_idx, fit_data in fit_data_per_protocol.items():
            if reference["spike_num"] == 0:
                protocol_error = fit_data["spike_num"]
            else:
                error_time = error_func(reference["spike_time"], fit_data["spike_time"])
                error_amp = error_func(reference["spike_amp"], fit_data["spike_amp"])
                error_num = torch.tensor(abs((reference["spike_num"] - fit_data["spike_num"]) / reference["spike_num"]),
                                         dtype=torch.float32, device="cuda")
                error_reset = error_func(reference["spike_reset"], fit_data["spike_reset"])
                error_width = error_func(reference["width"], fit_data["width"])
                protocol_error = (error_time + error_width + error_num) * 0.9 + (error_amp + error_reset) * 0.1
            if neuron_idx not in neuron_errors:
                neuron_errors[neuron_idx] = protocol_error
            else:
                neuron_errors[neuron_idx] += protocol_error

    for protocol, neuron_data_per_protocol in neuron_data_1.items():
        reference = neuron_data_per_protocol[0]
        fit_data_per_protocol = short_results[protocol]
        for neuron_idx, fit_data in fit_data_per_protocol.items():
            protocol_error = 0
            if reference["spike_num"] == 0:
                if fit_data["spike_num"] > 0:
                    protocol_error = protocol_error + 5 - protocol
            else:
                if fit_data["spike_num"] == 0:
                    protocol_error = protocol_error + protocol - 4
                else:
                    error_time = error_func(reference["spike_time"], fit_data["spike_time"])
                    error_amp = error_func(reference["spike_amp"], fit_data["spike_amp"])
                    error_width = error_func(reference["width"], fit_data["width"])
                    protocol_error += (error_time + error_width) * 0.9 + error_amp * 0.1
            if neuron_idx not in neuron_errors:
                neuron_errors[neuron_idx] = protocol_error
            else:
                neuron_errors[neuron_idx] += protocol_error

    var_last_1000 = V_zero_out[-1000:, 0, :].var(dim=0)
    delta = (V_zero_out[-1, 0, :] - V_zero_out[-500, 0, :]).abs()
    spike_count = spike_num_detect(V_zero_out)
    for neuron_idx in neuron_errors.keys():
        if var_last_1000[neuron_idx] > 1:
            neuron_errors[neuron_idx] = torch.tensor(float('inf'), device='cuda')
        if delta[neuron_idx] > 1e-4:
            neuron_errors[neuron_idx] = torch.tensor(float('inf'), device='cuda')
        if spike_count[0][neuron_idx] >= 2:
            neuron_errors[neuron_idx] = torch.tensor(float('inf'), device='cuda')
    error_all = torch.stack([neuron_errors[neuron_idx] for neuron_idx in sorted(neuron_errors.keys())])
    return error_all


def error_func(true, test):
    if test == []:
        errors = float('inf')
    else:
        true = np.array(true)
        test = np.array(test)
        true_length = len(true)
        test_length = len(test)
        min_len = min(true_length, test_length)
        truncated_true = true[:min_len]
        truncated_test = test[:min_len]
        err = np.sum(np.abs((truncated_true - truncated_test) / truncated_true))
        length_penalty = abs(test_length - true_length)
        err += length_penalty
        errors = err
    return errors


def get_configspace():
    space = spc.Space()
    weight = spc.Real("weight", 0.1, 2, q=0.1)
    cl = spc.Real("cl", 0.5, 2, q=0.1)
    cg = spc.Real("cg", 0.5, 2, q=0.1)
    speed = spc.Real("speed", 0.1, 1, q=0.1)

    space.add_variables([weight, cl, cg, speed])
    return space


def objective_function(config: spc.Configuration):
    start_time = time.time()
    w = config['weight']
    c1 = config['cl']
    c2 = config['cg']
    speed_limit = config['speed']
    pso = ParticleSwarmOptimizer(
        func=objective,  # 目标函数
        bounds=bounds,
        num_particles=num_neurons,  # 粒子数量
        max_iter=300,  # 最大迭代次数
        w=w, c1=c1, c2=c2, velocity_limit=speed_limit  # 超参数
    )
    best_position, best_score = pso.optimize()
    print("score：", best_score)
    elapsed_time = time.time() - start_time  # Calculate the elapsed time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    # 输出结果
    print(np.array2string(best_position, separator=','))

    return dict(objectives=[best_score])


opt_task = Optimizer(
    objective_function=objective_function,
    config_space=get_configspace(),
    num_objectives=1,  # 一个目标
    num_constraints=0,  # 没有约束
    sample_strategy='bo',  # 贝叶斯优化
    max_runs=100,
    initial_runs=5,  # 初始值
    advisor_type='default',
    surrogate_type='prf',
    acq_type='ei',
    acq_optimizer_type='local_random',
    init_strategy='sobol',  # 初始值随机
    visualization='basic',
    auto_open_html=True,
    task_id='task',
)

history = opt_task.run()
print(opt_task.get_history())
