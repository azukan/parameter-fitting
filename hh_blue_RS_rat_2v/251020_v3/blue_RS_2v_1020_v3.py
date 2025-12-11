import numpy as np
import bluepyopt as bpop
import bluepyopt.deapext.optimisations as bpop_deap
from typing import List, Tuple
import time

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
 5: {0: {'spike_time': [0.0608999989926815],
   'spike_amp': [4.4764018058776855],
   'spike_num': 1,
   'spike_reset': [],
   'width': [15]}},
 6: {0: {'spike_time': [0.055299997329711914],
   'spike_amp': [4.520143508911133],
   'spike_num': 1,
   'spike_reset': [],
   'width': [16]}},
 7: {0: {'spike_time': [0.054099999368190765],
   'spike_amp': [4.558402061462402],
   'spike_num': 1,
   'spike_reset': [],
   'width': [15]}},
 8: {0: {'spike_time': [0.05364999920129776],
   'spike_amp': [4.56689453125],
   'spike_num': 1,
   'spike_reset': [],
   'width': [16]}},
 9: {0: {'spike_time': [0.053349997848272324],
   'spike_amp': [4.586643218994141],
   'spike_num': 1,
   'spike_reset': [],
   'width': [16]}}
                 }

neuron_data_2 = {0: {0: {'spike_time': [0.12469999492168427,
                                        0.19325000047683716,
                                        0.2617499828338623,
                                        0.3303000032901764,
                                        0.39879998564720154,
                                        0.4672999978065491],
                         'spike_amp': [4.468343257904053,
                                       4.479752540588379,
                                       4.478189945220947,
                                       4.472619533538818,
                                       4.481297969818115,
                                       4.471827507019043],
                         'spike_num': 6,
                         'spike_reset': [-5.8715081214904785,
                                         -5.871493816375732,
                                         -5.871493816375732,
                                         -5.871505260467529,
                                         -5.871490001678467,
                                         -5.871506214141846],
                         'width': [16, 15, 16, 16, 15, 16]}},
                 1: {0: {'spike_time': [0.08609999716281891,
                                        0.13324999809265137,
                                        0.18039999902248383,
                                        0.22749999165534973,
                                        0.2746500074863434,
                                        0.32179999351501465],
                         'spike_amp': [4.479356288909912,
                                       4.484089374542236,
                                       4.473029613494873,
                                       4.481793403625488,
                                       4.483037948608398,
                                       4.468901634216309],
                         'spike_num': 6,
                         'spike_reset': [-5.856196403503418,
                                         -5.8561859130859375,
                                         -5.856186389923096,
                                         -5.856194496154785,
                                         -5.8561859130859375,
                                         -5.85618782043457],
                         'width': [16, 15, 16, 16, 15, 16]}},
                 2: {0: {'spike_time': [0.06644999980926514,
                                        0.10314999520778656,
                                        0.13989999890327454,
                                        0.17659999430179596,
                                        0.21329998970031738,
                                        0.25005000829696655,
                                        0.2867499887943268],
                         'spike_amp': [4.484732627868652,
                                       4.485476970672607,
                                       4.476609706878662,
                                       4.487361907958984,
                                       4.479365348815918,
                                       4.483635425567627,
                                       4.486279487609863],
                         'spike_num': 7,
                         'spike_reset': [-5.84191370010376,
                                         -5.84193754196167,
                                         -5.841935634613037,
                                         -5.841919422149658,
                                         -5.841945648193359,
                                         -5.841912269592285,
                                         -5.841934680938721],
                         'width': [16, 16, 16, 15, 16, 16, 15]}},
                 3: {0: {'spike_time': [0.04619999974966049,
                                        0.07249999791383743,
                                        0.09879999607801437,
                                        0.1251000016927719,
                                        0.1513499915599823,
                                        0.17764998972415924,
                                        0.20559999346733093],
                         'spike_amp': [4.4858198165893555,
                                       4.4910054206848145,
                                       4.491335391998291,
                                       4.483737945556641,
                                       4.483402729034424,
                                       4.491405487060547,
                                       4.469889163970947],
                         'spike_num': 7,
                         'spike_reset': [-5.815540313720703,
                                         -5.815410137176514,
                                         -5.815375804901123,
                                         -5.815414905548096,
                                         -5.815429210662842,
                                         -5.81540584564209],
                         'width': [16, 16, 15, 16, 16, 15, 16]}},
                 4: {0: {'spike_time': [0.032249998301267624,
                                        0.05144999921321869,
                                        0.0705999955534935,
                                        0.08980000019073486,
                                        0.10894999653100967,
                                        0.12815000116825104,
                                        0.14729999005794525],
                         'spike_amp': [4.49575662612915,
                                       4.477965354919434,
                                       4.491093158721924,
                                       4.483866214752197,
                                       4.488966941833496,
                                       4.4881086349487305,
                                       4.48484992980957],
                         'spike_num': 7,
                         'spike_reset': [-5.779344081878662,
                                         -5.778434753417969,
                                         -5.7783966064453125,
                                         -5.778412342071533,
                                         -5.778416633605957,
                                         -5.778388500213623],
                         'width': [16, 16, 15, 16, 16, 16, 16]}}
                 }


def pqn_2v(V, N, P, I_true, dt=0.00005):
    a_fn, a_fp, b_fn, c_fn, a_gn, a_gp, b_gn, c_gn, r_g, phi, tau, k = P
    b_fp = a_fn * b_fn / a_fp
    c_fp = -a_fn * a_fn * b_fn * b_fn / a_fp + a_fn * b_fn * b_fn + c_fn
    b_gp = a_gn * (b_gn - r_g) / a_gp + r_g
    c_gp = -a_gn * a_gn * (b_gn - r_g) * (b_gn - r_g) / a_gp + a_gn * (b_gn - r_g) * (b_gn - r_g) + c_gn

    diff = tau / dt
    I_true = I_true * k * 10.0

    for i in range(V.shape[0] - 1):
        v = V[i]
        n = N[i]
        I_stim = I_true[i]

        if v < 0:
            fv = a_fn * (v - b_fn) * (v - b_fn) + c_fn
        else:
            fv = a_fp * (v - b_fp) * (v - b_fp) + c_fp
        if v < r_g:
            fn = a_gn * (v - b_gn) * (v - b_gn) + c_gn
        else:
            fn = a_gp * (v - b_gp) * (v - b_gp) + c_gp

        dn = (fn - n) / diff  # calculate dn
        dv = phi * (fv - n + I_stim) / diff  # calculate dv

        V[i + 1] = v + dv
        N[i + 1] = n + dn
        # 如果状态超出了一个范围（-100 到 100），则将其限制在该范围内。
        if V[i + 1] > 100.0:
            V[i + 1] = 100.0
        elif V[i + 1] < -100.0:
            V[i + 1] = -100.0
        if N[i + 1] > 100.0:
            N[i + 1] = 100.0
        elif N[i + 1] < -100.0:
            N[i + 1] = -100.0

    return V, N


def compute_threshold(Vt):
    Vmax = max(Vt)
    Vmin = min(Vt)
    thre = Vmax - (Vmax - Vmin) / 3
    thre = max(thre, 0.5)
    return thre


def spike_detect(Vt, thre, t=0.00005):
    spike = []
    reset_time = []
    num = 0
    flag = True
    for i in range(len(Vt) - 1):
        if flag and Vt[i] > thre and Vt[i + 1] < Vt[i]:
            spike.append(i)
            num += 1
            flag = False
        elif not flag and Vt[i] < 0 and Vt[i + 1] > Vt[i]:
            reset_time.append(i)
            flag = True
    peaks = Vt[spike]
    reset = Vt[reset_time]
    spike_time = [x * t for x in spike]

    return spike_time, num, peaks, reset


def spike_num_detect(Vt):
    thre = compute_threshold(Vt)
    num = spike_detect(Vt, thre)[1]
    return num


def width_detect(V, threshold):
    b = []
    flag = True
    for i in range(len(V) - 1):
        if flag and (V[i] >= threshold):
            b.append(i)
            flag = False
        elif (not flag) and (V[i] <= threshold):
            b.append(i)
            flag = True

    thre_num = len(b)
    if thre_num >= 4:
        avg_width = np.array(b[1::2]) - np.array(b[:-1:2])

    elif thre_num == 2:
        avg_width = np.array([b[1] - b[0]])
    else:
        avg_width = np.array([])

    return thre_num, avg_width


def target_detect(Vt):
    threshold = compute_threshold(Vt)
    spike, num, peaks, reset = spike_detect(Vt, threshold)
    thre_num, average_width = width_detect(Vt, threshold)
    return num, spike, peaks, reset, average_width


def error_func(true, test):
    if len(test) == 0:
        errors = 1e9
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


I_hh1 = np.zeros((2000, 10))
for i in range(10):
    I_hh1[1000:1060, i] = 0.05 + i * 0.025

I_hh2 = np.zeros((10000, 5))

I_hh2[:, 0] = 0.006
I_hh2[:7000, 1] = 0.008
I_hh2[:6000, 2] = 0.01
I_hh2[:4000, 3] = 0.014
I_hh2[:3000, 4] = 0.02

I_zero = np.zeros((20000,))


def objective_function(param):
    V_zero = np.zeros((20000,))
    N_zero = np.zeros((20000,))
    V_s = np.zeros((10000, 5))
    N_s = np.zeros((10000, 5))
    V_p = np.zeros((2000, 10))
    N_p = np.zeros((2000, 10))

    V_zero, N_zero = pqn_2v(V_zero, N_zero, param, I_zero)
    V_s[0, :] = V_zero[-1]
    V_p[0, :] = V_zero[-1]
    N_s[0, :] = N_zero[-1]
    N_p[0, :] = N_zero[-1]

    neuron_error = 0
    for i in range(5):
        V_s[:, i], N_s[:, i] = pqn_2v(V_s[:, i], N_s[:, i], param, I_hh2[:, i])
        num, spike, peaks, reset, average_width = target_detect(V_s[:, i])
        truth = neuron_data_2[i][0]

        error_time = error_func(truth['spike_time'], spike)
        error_num = abs((truth['spike_num'] - num) / truth['spike_num'])
        error_amp = error_func(truth['spike_amp'], peaks)
        error_reset = error_func(truth['spike_reset'], reset)
        error_width = error_func(truth['width'], average_width)
        neuron_error += (error_time + error_width + error_num) * 0.9 + (error_amp + error_reset) * 0.1

    for j in range(10):
        V_p[:, j], _ = pqn_2v(V_p[:, j], N_p[:, j], param, I_hh1[:, j])
        num, spike, peaks, _, average_width = target_detect(V_p[:, j])
        truth = neuron_data_1[j][0]
        if truth['spike_num'] == 0:
            if num > 0:
                neuron_error = neuron_error + 5 - j
        else:
            if num == 0:
                neuron_error = neuron_error - 4 + j
            else:
                error_time = error_func(truth['spike_time'], spike)
                error_amp = error_func(truth['spike_amp'], peaks)
                error_width = error_func(truth['width'], average_width)
                neuron_error += (error_time + error_width) * 0.9 + error_amp * 0.1

    var_last_1000 = V_zero[-1000:].var()
    delta = np.abs(V_zero[-1] - V_zero[-500])
    spike_count = spike_num_detect(V_zero)
    if var_last_1000 > 1:
        neuron_error = 1e9
    if delta > 1e-4:
        neuron_error = 1e9
    if spike_count >= 2:
        neuron_error = 1e9

    return neuron_error,V_s,V_p


class PQNEvaluator:
    """
    将你的 objective_function(param) 封装成 BluePyOpt/DEAPOptimisation 可用的 Evaluator。
    关键要求：
      - 提供 self.params: 带 lower_bound/upper_bound 的参数列表
      - 提供 self.objectives: 目标名列表（长度 = 目标数）
      - 提供 self.param_names: 便于日志展示（可选）
      - 提供 init_simulator_and_evaluate_with_lists(x): 返回长度==len(objectives) 的 tuple/list
    """

    def __init__(
            self,
            param_bounds: List[Tuple[float, float]],
            param_names: List[str],
            shared_data: dict
    ):
        """
        Args:
            param_bounds: 每个参数的 (low, high) 上下界，顺序需与 objective_function 的 P 解包顺序一致
            param_names: 参数名列表（同顺序）
            shared_data: 打包传入 objective_function 所需的全局变量（如 I_zero/I_hh1/I_hh2/...）
                         例如: {
                           "I_zero": I_zero,
                           "I_hh1": I_hh1,
                           "I_hh2": I_hh2,
                           "neuron_data_1": neuron_data_1,
                           "neuron_data_2": neuron_data_2,
                           "error_func": error_func,
                           "target_detect": target_detect,
                           "spike_num_detect": spike_num_detect
                         }
        """
        assert len(param_bounds) == len(param_names), \
            "param_bounds 与 param_names 长度必须一致"

        # ---- 1) BluePyOpt 参数对象，用于提供 lower_bound / upper_bound ----
        self.params = [
            bpop.parameters.Parameter(name=n, bounds=b)
            for n, b in zip(param_names, param_bounds)
        ]

        # ---- 2) 目标定义（你这里是单目标：neuron_error，且是最小化）----
        self.objectives = ["neuron_error"]
        self.param_names = list(param_names)

        # ---- 3) 把你的全局依赖存起来，避免在 objective_function 里用全局变量 ----
        self.shared = shared_data

        # 可选：保存最近一次轨迹，方便 debug/可视化
        self.last_traces = None  # dict 存 (V_s, V_p, error)

    # ========== 供 DEAPOptimisation.setup_deap 注册的评估函数 ==========
    def init_simulator_and_evaluate_with_lists(self, x: List[float]):
        """
        x: 个体（参数列表/数组）
        return: (neuron_error, )  —— 注意返回 tuple/list，长度==len(self.objectives)
        """
        # 将共享数据注入到当前命名空间，让 objective_function 能访问到
        # （也可以改写 objective_function，使之显式接收这些数据；这里不改你的函数签名）
        globals_backup = {}
        try:
            # 把 shared 中的键作为“临时全局变量”注入
            for k, v in self.shared.items():
                if k in globals():
                    globals_backup[k] = globals()[k]
                globals()[k] = v

            # 调用你现有的目标函数
            neuron_error, V_s, V_p = objective_function(np.array(x, dtype=float))

            # 记录一下，便于之后可视化/调试
            self.last_traces = {
                "V_s": V_s,
                "V_p": V_p,
                "error": float(neuron_error),
                "x": list(map(float, x))
            }

            # 返回单目标 tuple（DEAP 需要 tuple/list）
            return (float(neuron_error),)

        finally:
            # 恢复被覆盖的全局变量，避免污染
            for k in self.shared.keys():
                if k in globals_backup:
                    globals()[k] = globals_backup[k]
                else:
                    try:
                        del globals()[k]
                    except KeyError:
                        pass


PARAM_NAMES = [
    "a_fn", "a_fp", "b_fn", "c_fn",
    "a_gn", "a_gp", "b_gn", "c_gn",
    "r_g", "phi", "tau", "k"
]
PARAM_BOUNDS = [
    (0.1, 4),
    (-15, -1),
    (-5, 5),
    (-10, 10),

    (0.1, 4),
    (1, 15),
    (-5, 5),
    (-10, 10),

    (-5, 5),
    (0.1, 10),
    (0.001, 1),
    (0.1, 12)
]

start_time = time.time()

# ======== 4. 打包共享数据 ========
shared_data = dict(
    I_zero=I_zero,
    I_hh1=I_hh1,
    I_hh2=I_hh2,
    neuron_data_1=neuron_data_1,
    neuron_data_2=neuron_data_2,
    error_func=error_func,
    target_detect=target_detect,
    spike_num_detect=spike_num_detect
)

# ======== 5. 构造 Evaluator ========
evaluator = PQNEvaluator(
    param_bounds=PARAM_BOUNDS,
    param_names=PARAM_NAMES,
    shared_data=shared_data
)

# ======== 6. 构造优化器（bluepyopt 内置类） ========
opt = bpop_deap.DEAPOptimisation(
    evaluator=evaluator,
    seed=50,                # 随机种子
    offspring_size=5000,      # 每代样本数
    eta=20.0,               # SBX/Polynomial参数
    cxpb=0.7,               # 交叉概率
    mutpb=0.3,              # 变异概率
    selector_name='NSGA2'    # 或 'NSGA2'
)

# ======== 7. 启动优化 ========
pop, hof, log, history = opt.run(
    max_ngen=200,            # 迭代代数
    cp_filename='checkpoint.pkl',
    cp_frequency=5,
    continue_cp=False
)

# ======== 8. 输出结果 ========
best = hof[0]
best_error = best.fitness.values[0]
print("Best error:", best_error)
print("Best param:", best)
print(f'Total time = {time.time() - start_time}')

for gen, record in enumerate(log):
    print(f"Iter: {gen+1} , Best = {record['min']:.6f}")
