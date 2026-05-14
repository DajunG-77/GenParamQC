import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.circuit.library import MSGate
from qiskit.circuit.library import RXXGate


class QiskitCircuitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phi, qc_param, param_list):
        bind_dict = {param: val.item() for param, val in zip(param_list, phi)}
        bound_qc = qc_param.assign_parameters(bind_dict)
        U = Operator(bound_qc).data
        U_torch = torch.complex(
            torch.tensor(U.real, dtype=phi.dtype, device=phi.device),
            torch.tensor(U.imag, dtype=phi.dtype, device=phi.device),
        )
        if U_torch.requires_grad:
            print("U_torch requires grad")
        ctx.qc_param = qc_param
        ctx.param_list = param_list
        ctx.phi = phi.clone()
        ctx.eps = 1e-3
        return U_torch

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: 损失函数关于输出 U 的梯度
        返回: 关于输入参数 phi 的梯度，其他输入不需要梯度故返回 None
        """
        phi = ctx.phi
        eps = ctx.eps
        n_params = phi.numel()
        grad_phi = torch.zeros_like(phi)

        # 对每个参数进行有限差分求导
        for i in range(n_params):
            # 对第 i 个参数，构造正、负扰动
            phi_plus = phi.clone()
            phi_minus = phi.clone()
            phi_plus[i] += eps
            phi_minus[i] -= eps

            # 分别绑定正、负扰动后的参数
            bind_dict_plus = {
                param: val.item() for param, val in zip(ctx.param_list, phi_plus)
            }
            bind_dict_minus = {
                param: val.item() for param, val in zip(ctx.param_list, phi_minus)
            }
            qc_plus = ctx.qc_param.assign_parameters(bind_dict_plus)
            qc_minus = ctx.qc_param.assign_parameters(bind_dict_minus)

            U_plus = Operator(qc_plus).data
            U_minus = Operator(qc_minus).data

            U_plus_torch = torch.complex(
                torch.tensor(U_plus.real, dtype=phi.dtype, device=phi.device),
                torch.tensor(U_plus.imag, dtype=phi.dtype, device=phi.device),
            )
            U_minus_torch = torch.complex(
                torch.tensor(U_minus.real, dtype=phi.dtype, device=phi.device),
                torch.tensor(U_minus.imag, dtype=phi.dtype, device=phi.device),
            )

            # 有限差分计算 dU/dphi_i ≈ (U_plus - U_minus) / (2 * eps)
            dU_dphi = (U_plus_torch - U_minus_torch) / (2 * eps)
            # 利用链式法则计算梯度：grad_phi[i] = Re(trace( dU_dphi^† * grad_output ))
            grad_i = torch.real(torch.sum(torch.conj(dU_dphi) * grad_output))
            grad_phi[i] = grad_i

        # 只有 phi 是需要梯度的，其余 qc_param 和 param_list 均不需要
        return grad_phi, None, None


def extract_or_parameterize(qc):
    qc_new = QuantumCircuit(qc.num_qubits)
    param_list = []
    init_vals = []

    for inst, qargs, _ in qc.data:
        if inst.name == "ms":
            theta_val = inst.params[0]
            param = Parameter(f"ms_{len(param_list)}")
            num_qubits = len(qargs)
            new_inst = MSGate(num_qubits, param)
            qc_new.append(new_inst, qargs)
            param_list.append(param)
            init_vals.append(theta_val)
        elif inst.name.lower() == "rxx":
            theta_val = float(inst.params[0])
            param = Parameter(f"rxx_{len(param_list)}")
            new_inst = RXXGate(param)
            qc_new.append(new_inst, qargs)
            param_list.append(param)
            init_vals.append(theta_val)

        elif inst.name in ["rx", "ry"]:
            theta_val = inst.params[0]
            param = Parameter(f"{inst.name}_{len(param_list)}")
            new_inst = inst.__class__(param, label=inst.label)
            qc_new.append(new_inst, qargs)
            param_list.append(param)
            init_vals.append(theta_val)
        else:
            qc_new.append(inst, qargs)

    return qc_new, param_list, init_vals


def optimize_qc_parameters(qc, U_target, steps=100, lr=0.1, verbose=True):
    qc_param, param_list, init_vals = extract_or_parameterize(qc)
    n_params = len(param_list)

    if n_params == 0:
        return None, qc, None
        # raise ValueError("Not valid")

    phi = torch.nn.Parameter(
        torch.tensor(init_vals, dtype=torch.float64), requires_grad=True
    )
    optimizer = torch.optim.SGD([phi], lr=lr)

    d = U_target.shape[-1]
    for step in range(steps):
        optimizer.zero_grad()
        U_sim = QiskitCircuitFunction.apply(phi, qc_param, param_list)

        # loss = 0.5 * torch.linalg.norm(U_sim - U_target, ord="fro") ** 2
        s = torch.trace(U_target.conj().T @ U_sim)
        loss = 1.0 - (torch.abs(s) ** 2) / (d * d)        # ∈ [0,1]

        loss.backward()
        optimizer.step()

        if verbose and (step % 10 == 0 or step == steps - 1):
            values_str = ", ".join([f"{v.item():.4f}" for v in phi])
            print(f"Step {step:3d}: Loss = {loss.item():.6f}, Params = [{values_str}]")

    best_bind = {param: phi[idx].item() for idx, param in enumerate(param_list)}
    qc_optimized = qc_param.assign_parameters(best_bind)

    return qc_param, qc_optimized, best_bind
