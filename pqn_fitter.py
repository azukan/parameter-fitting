"""
## Usage

- mkdir data runs
- tensorboard --logdir runs --reload_interval 5 &
- python pqn_fitter

So you can see the generated plots over time (refresh the page, as tensorboard does not refresh for you)
"""

import torch
from torch.func import jacrev # type: ignore
from torch.utils.tensorboard import SummaryWriter # type: ignore
import numpy as np
import pickle
import matplotlib.pyplot as plt
from line_profiler import profile
from pathlib import Path

dt = 0.00005

## Pospischill's model
@torch.jit.script
def euler_hh_scripted(Vars: torch.Tensor, I: torch.Tensor, dt: float=dt) -> torch.Tensor:
    """
    编译后的 Hodgkin-Huxley 模型积分函数。
    V: (time_steps, 5, num_protocols, num_neurons)
    I: (time_steps, num_neurons)
    """
    # NOTE (Enrico): actually num_neurons and num_protocols are flipped
    E_Na = 0.05
    g_Na = 560.0
    g_Kd = 60.0
    E_Kd = -0.09
    E_K = -0.09
    g_M = 0.75
    tau_max = 0.608
    g_Leak = 0.205
    E_Leak = -0.0703
    C_membrane = 0.02448
    vt = -0.0562

    # I 的形状：(time_steps, num_neurons) -> (time_steps, 1, num_neurons)
    I_expanded = I.unsqueeze(1)
    T = Vars.size(0)
    for i in range(T - 1):
        # 取出当前时间步各状态变量，形状：(num_neurons, num_protocols)
        v = Vars[i, 0, :, :]
        m = Vars[i, 1, :, :]
        h = Vars[i, 2, :, :]
        n = Vars[i, 3, :, :]
        p = Vars[i, 4, :, :]

        # 计算各通道电流
        I_Leak = g_Leak * (E_Leak - v)
        I_Na = g_Na * (m ** 3) * h * (E_Na - v)
        I_Kd = g_Kd * (n ** 4) * (E_Kd - v)
        I_M = g_M * p * (E_K - v)

        # 计算 alpha/beta 参数
        alpham = (-0.32) * (v - vt - 0.013) / (torch.exp(-(v - vt - 0.013) / 0.004) - 1) * 1e6
        betam  = 0.28  * (v - vt - 0.040) / (torch.exp((v - vt - 0.040) / 0.005) - 1) * 1e6
        alphah = 0.128 * torch.exp(-(v - vt - 0.017) / 0.018) * 1e3
        betah  = 4.0 / (1 + torch.exp(-(v - vt - 0.040) / 0.005)) * 1e3
        alphan = (-0.032) * (v - vt - 0.015) / (torch.exp(-(v - vt - 0.015) / 0.005) - 1) * 1e6
        betan  = 0.5 * torch.exp(-(v - vt - 0.010) / 0.040) * 1e3
        p_inf  = 1.0 / (1.0 + torch.exp(-(v + 0.035) / 0.010))
        tau_hh = tau_max / (3.3 * torch.exp((v + 0.035) / 0.020) + torch.exp(-(v + 0.035) / 0.020))

        # 修改此处：调整注入电流张量的形状与 v 一致
        current_i = I_expanded[i].transpose(0, 1).expand(-1, Vars.size(3))
        dv = (1.0 / C_membrane) * (I_Leak + I_Na + I_Kd + I_M + current_i) * dt
        dm = (1 - torch.exp(-(alpham + betam) * dt)) * (alpham / (alpham + betam) - m)
        dh = (1 - torch.exp(-(alphah + betah) * dt)) * (alphah / (alphah + betah) - h)
        dn = (1 - torch.exp(-(alphan + betan) * dt)) * (alphan / (alphan + betan) - n)
        dp = (p_inf - p) / tau_hh * dt

        # 更新下一时间步状态
        Vars[i + 1, 0, :, :] = v + dv
        Vars[i + 1, 1, :, :] = m + dm
        Vars[i + 1, 2, :, :] = h + dh
        Vars[i + 1, 3, :, :] = n + dn
        Vars[i + 1, 4, :, :] = p + dp

    Vars = Vars * 100.0
    return Vars


## 3 varibles PQN model 

def single_step_3v(I_stim: torch.Tensor, state: torch.Tensor, params: torch.Tensor, state_mask: torch.Tensor, dt: float=0.00005):

    # Extract parameters from P and reshape each parameter to (1, num_particles)

    # I_stim is (N,), state is (3*N flattened to have the same var type contiguous), params is (19,)
    s = state.view(3, -1)
    v = s[0]
    n = s[1]
    q = s[2]

    a_fn, a_fp, b_fn, c_fn, a_gn, a_gp, b_gn, c_gn, r_g, I_0, phi, tau, a_hn, a_hp, b_hn, c_hn, r_h, epsilion, k = params #[params[i] for i in range(19)]
    
    b_fp = a_fn * b_fn / a_fp
    c_fp = a_fn * b_fn * b_fn + c_fn - a_fp * b_fp * b_fp
    b_gp = r_g - a_gn * (r_g - b_gn) / a_gp
    c_gp = a_gn * (r_g - b_gn) * (r_g - b_gn) - a_gp * (r_g - b_gp) * (r_g - b_gp) + c_gn
    b_hp = r_h - a_hn * (r_h - b_hn) / a_hp
    c_hp = a_hn * (r_h - b_hn) * (r_h - b_hn) + c_hn - a_hp * (r_h - b_hp) * (r_h - b_hp)

    I_stim = I_stim * k * 10.0
    
    # TODO (@azukan): these operations break autograd and make training impossible for these parameters.
    # Replace with torch.sigmoid or torch.relu
    # E.g.
    #
    # sigmoid_slope = 50.0  # controls sharpness of transition
    # smooth_indicator = torch.sigmoid(sigmoid_slope * (r_g - v))  # ~ 1 if v < r_g, ~ 0 otherwise

    # gv = smooth_indicator * (a_gn * (v - b_gn)**2 + c_gn) + \
    #      (1 - smooth_indicator) * (a_gp * (v - b_gp)**2 + c_gp)

    fv = torch.where(v < 0.0, a_fn * (v - b_fn) * (v - b_fn) + c_fn,
                     a_fp * (v - b_fp) * (v - b_fp) + c_fp)
    gv = torch.where(v < r_g, a_gn * (v - b_gn) * (v - b_gn) + c_gn,
                     a_gp * (v - b_gp) * (v - b_gp) + c_gp)
    hv = torch.where(v < r_h, a_hn * (v - b_hn) * (v - b_hn) + c_hn,
                     a_hp * (v - b_hp) * (v - b_hp) + c_hp)
    
    dv = phi * (fv - n - q + I_0 + I_stim) / tau * dt
    dn = (gv - n) / tau * dt
    dq = (hv - q) * epsilion / tau * dt
    
    
    new_v = v + dv*state_mask
    new_n = n + dn*state_mask
    new_q = q + dq*state_mask

    # TODO (@azukan): Similarly, these should go away. Relu would work here.
    exceeded_mask = (new_v.abs() >= 5.0) | (new_n.abs() >= 100.0) | (new_q.abs() >= 100.0)
    state_mask = state_mask * (~exceeded_mask)
    
    return (new_v, new_n, new_q), state_mask


## Bounds for PQN's parameters
# TODO (@azukan): You can also cheat and provide results obtained from previous fits.
bounds = np.array([
    (0.1, 4),
    (-15, -1),
    (-5, 5),
    (-10, 10),
    (0.1, 4),
    (1, 15),
    (-5, 5),
    (-10, 10),
    (-5, 5),
    (-10, 10),
    (0.1, 10),
    (0.001, 1),
    (-2, 3),
    (1, 15),
    (-10, 10),
    (-10, 10),
    (-5, 10),
    (0.001, 0.05),
    (0.01, 10),
])

def synthetize():
    ## generate reference data
    ## I_hh1,V_hh1: pulse current input
    ## I_hh2, V_hh2: sustained current input
    TS_exp1 = 2000
    TS_exp2 = 10000
    NProt_exp1 = 10
    NProt_exp2 = 5
    NNeurons = 1 # I think it is unused?

    HH_VarNum = 5

    # The initial conditions for HH
    initial_values = torch.tensor([-7.03e-02, 6.538e-04, 9.998e-1, 3.0244e-03, 0.0], device='cuda')

    V_hh1_init = torch.zeros((TS_exp1, HH_VarNum, NProt_exp1, NNeurons), device='cuda', dtype=torch.float32)

    for i in range(10):
        V_hh1_init[0, :, i, 0] = initial_values

    # Small current burst, at increasing peaks per protocol.
    I_hh1 = torch.zeros((TS_exp1, NProt_exp1), device='cuda', dtype=torch.float32)
    for i in range(10):
        I_hh1[1000:1100, i] = 0.04 + i * 0.01
    V_hh1 = euler_hh_scripted(V_hh1_init, I_hh1)

    # Long constant current.
    V_hh2_init = torch.zeros((TS_exp2, HH_VarNum, NProt_exp2, NNeurons), device='cuda', dtype=torch.float32)
    for i in range(5):
        V_hh2_init[0, :, i, 0] = initial_values
    I_hh2 = torch.zeros((10000, 5), device='cuda', dtype=torch.float32)
    I_hh2[100:, 0] = 0.0125
    I_hh2[100:7000, 1] = 0.015
    I_hh2[100:5000, 2] = 0.02
    I_hh2[100:3200, 3] = 0.025
    I_hh2[100:2000, 4] = 0.04
    V_hh2 = euler_hh_scripted(V_hh2_init, I_hh2)

    with open("data/hh1.pt", "wb") as f:
        torch.save(((I_hh1, V_hh1)), f)

    with open("data/hh2.pt", "wb") as f:
        torch.save(((I_hh2, V_hh2)), f)

def assimil_4dvar(Istim: torch.Tensor, Vmem: torch.Tensor, obs_noise=1e-5, tol=1e-4, lr=0.1):
    Vmem = Vmem.squeeze()
    Istim = Istim.squeeze()
    bounds_t = torch.tensor(bounds).cuda()
    initial_params = bounds_t[:, 1] - bounds_t[:, 0]*torch.rand(len(bounds_t)).cuda()+bounds_t[:, 0]
    N = Vmem.size(1)
    initial_voltage = Vmem[0, :]
    initial_internal_params = torch.randn(3-1, N).cuda() * 0.001
    initial_vars = torch.cat([initial_voltage, initial_internal_params.ravel()])
    torch.autograd.set_detect_anomaly(True)
    #initial_I = Istim[0, :]

    vars_size = len(initial_vars)
    param_size = len(initial_params)
    total_state_size = vars_size + param_size 

    assert param_size == 19
    assert vars_size == 10*3, f"N_hidden is {vars_size} which is not 30"

    #x0 = (initial_vars, initial_params)
    #x0_catted = torch.cat(x0).float()
    x0_catted = torch.cat([initial_vars, initial_params]).float().detach().requires_grad_()

    def from_x0():
        # separate into state and params
        return x0_catted[:vars_size], x0_catted[vars_size:]

    # Observation and noise matrices
    H = torch.zeros(N, vars_size).cuda()
    H[:, :N] = torch.eye(N).cuda()
    Rinv = torch.eye(N).cuda() * 1/obs_noise

    torch._dynamo.config.compiled_autograd = True

    @torch.compile
    def wrapped(state, params, I, state_mask):
        new_s, state_mask = single_step_3v(I, state, params, state_mask)#x0[:vars_size], x0[vars_size:], state_mask)
        to_derive = torch.cat(new_s)

        return (to_derive, (to_derive, state_mask))


    @profile
    def trace_grads():
        T = Istim.size(0)
        state, params = from_x0()

        # Note: Pytorch complains that it is better to copy the tensor via .clone().detach().requires_grad_(True)

        #Istim = Istim.clone().detach().requires_grad_(False)
        #state = state.clone().detach().requires_grad_(True)
        #params = params.clone().detach().requires_grad_(True)

        # Adjoint method
        lms_vars = torch.zeros(T, vars_size).cuda()
        lms_param = torch.zeros(param_size).cuda()

        state_mask = torch.ones((N,), dtype=torch.bool).cuda()


        jac_fun = jacrev(wrapped, argnums=(0, 1), has_aux=True) # type: ignore

        #Js = torch.empty(T, total_state_size, total_state_size).cuda()
        Js_states = torch.empty(T, vars_size, vars_size).cuda() 
        Js_params = torch.empty(T, vars_size, param_size).cuda() 
        states = torch.empty(T, vars_size).cuda()

        for t in range(T):
            J, s = jac_fun(state, params, Istim[t], state_mask)
            new_state, state_mask = s
            states[t] = state
            Js_states[t] = J[0].float()
            Js_params[t] = J[1].float()
            state = new_state

        #Js = torch.stack(Js)
        #states = torch.stack(states)

        with torch.no_grad():
            for t in range(T-2, -1, -1):
                # GradJ(x0) = B0^{-1} + lambda(0)
                # lambda[i] = lambda[i+1] + (dL/dx - dF/dx @ lambda[i+1])*dt + (y[t] - Obs(x[t]))
                lms_vars[t] = lms_vars[t+1] + dt*(Js_states[t].T @ lms_vars[t+1] + H.T @ Rinv @ (Vmem[t] - H[:, :vars_size]@states[t]))
                lms_param += Js_params[t].T @ lms_vars[t+1]


            return lms_vars[0], dt*lms_param, states#[:len(initial_vars)], lms[0][len(initial_vars):]

    optimizer = torch.optim.SGD((x0_catted,), lr=lr)
    loss = torch.inf
    grad_norm = torch.inf

    writer = SummaryWriter()
    
    n_epoch = 0
    for i in range(10):
        optimizer.zero_grad()

        *grad, states = trace_grads()
        x0_catted.grad = torch.cat(grad)
        grad_norm = torch.norm(x0_catted.grad).item()

        optimizer.step()

        with torch.no_grad():
            x0_catted[0:N] = initial_voltage[0] # Small trick
            x0_catted[vars_size:] = torch.clip(from_x0()[1], bounds_t[:, 0], bounds_t[:, 1])
            observable = states @ H.T
            loss = torch.sum(torch.pow((observable - Vmem), 2)).item()
            print(f"[{n_epoch}]: {loss:.3e}, grad_norm={grad_norm}")

        fig, ax = plt.subplots()
        ax.plot(Vmem.cpu().numpy(), label="HH")
        ax.plot(observable.cpu().numpy(), label="PQN")
        ax.set_ylim((-7, +5))
        ax.legend()
        writer.add_figure(f"PQN Simulation", fig, n_epoch)
        plt.close(fig)

        n_epoch += 1

        if n_epoch % 10 == 0:
            with open(f"data/checkpoint{n_epoch}.pickle", "wb") as f:
                pickle.dump(x0_catted, f)


    with open(f"data/pqn_final.pickle", "wb") as f:
        pickle.dump(x0_catted, f)

    writer.close()

    return x0_catted

if __name__ == "__main__":
    if not (Path("data/hh1.pt").exists() and Path("data/hh2.pt").exists()):
        synthetize()

    with open("data/hh1.pt", "rb") as f:
        I_hh1, V_hh1 = torch.load(f, map_location="cuda")

    #T = I_hh1.size(0)
    T = 200
    TStartFrom = 1000
    #plt.plot(V_hh1[:T, 0, :, 0].cpu().numpy())
    #plt.show()

    assimil_4dvar(I_hh1[TStartFrom:T+TStartFrom], V_hh1[TStartFrom:T+TStartFrom, 0, :, 0])
