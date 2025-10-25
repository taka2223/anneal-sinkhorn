"""
使用 jax.lax.scan 实现的 Annealed Sinkhorn
算法逻辑与 annealed_sinkhorn_corrected.py 完全相同
"""
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple


def logsumexp_stable(x, axis, keepdims=False):
    """数值稳定的log-sum-exp"""
    max_x = jnp.max(x, axis=axis, keepdims=True)
    result = max_x + jnp.log(
        jnp.sum(jnp.exp(x - max_x), axis=axis, keepdims=True))
    if not keepdims:
        result = jnp.squeeze(result, axis=axis)
    return result


class ScanState(NamedTuple):
    """Scan的状态"""
    u: jnp.ndarray  # (m,) 对偶变量u
    v: jnp.ndarray  # (n,) 对偶变量v
    beta: float  # 当前温度
    t: int  # 当前迭代次数


class ScanOutput(NamedTuple):
    """每步输出"""
    u: jnp.ndarray
    v: jnp.ndarray
    beta: float
    plan: jnp.ndarray


@partial(jax.jit, static_argnames=['debiased', 'num_iters'])
def solve_annealed_sinkhorn_scan(cost_matrix: jnp.ndarray,
                                 p: jnp.ndarray,
                                 q: jnp.ndarray,
                                 beta0: float,
                                 kappa: float,
                                 num_iters: int,
                                 debiased: bool = False):
    """
    使用 jax.lax.scan 的 Annealed Sinkhorn
    
    关键点：
    1. 算法逻辑与循环版本完全相同
    2. 显式传递所有参数，避免闭包
    3. 清晰的状态传递
    """
    m, n = cost_matrix.shape
    log_p = jnp.log(p)
    log_q = jnp.log(q)

    def step_fn(state: ScanState, _):
        """单步迭代函数"""
        u_prev, v_prev, beta_prev, t = state

        # ============ 更新 u ============
        # temp1[i,j] = v[j] + log(q[j]) - β*c[i,j]
        temp1 = v_prev[None, :] + log_q[None, :] - beta_prev * cost_matrix

        if debiased and t > 1:
            # Debiased修正
            # 注意：这里t是当前迭代编号（从1开始）
            t_float = jnp.array(t, dtype=jnp.float32)
            correction_coef = (t_float**kappa -
                               (t_float - 1)**kappa) / (t_float**kappa)
            temp1 = temp1 - correction_coef * u_prev[:, None]

        # u[i] = -log(Σ_j exp(temp1[i,j]))
        u_new = -logsumexp_stable(temp1, axis=1, keepdims=False)

        # ============ 更新 β ============
        # 关键：在u和v之间更新！
        beta_new = beta0 * ((t + 1.0)**kappa)

        # ============ 更新 v ============
        # temp2[i,j] = u[i] + log(p[i]) - β*c[i,j]
        temp2 = u_new[:, None] + log_p[:, None] - beta_new * cost_matrix

        # v[j] = -log(Σ_i exp(temp2[i,j]))
        v_new = -logsumexp_stable(temp2, axis=0, keepdims=False)

        # ============ 计算传输计划 ============
        log_plan = (u_new[:, None] + log_p[:, None] + v_new[None, :] +
                    log_q[None, :] - beta_new * cost_matrix)
        plan = jnp.exp(log_plan)

        # 下一个状态
        next_state = ScanState(u=u_new, v=v_new, beta=beta_new, t=t + 1)

        # 输出当前步的结果
        output = ScanOutput(u=u_new, v=v_new, beta=beta_new, plan=plan)

        return next_state, output

    # 初始状态
    initial_state = ScanState(
        u=jnp.zeros(m),
        v=jnp.zeros(n),
        beta=beta0,
        t=1  # 从1开始计数
    )

    # 运行scan
    final_state, outputs = jax.lax.scan(step_fn,
                                        initial_state,
                                        xs=None,
                                        length=num_iters)

    return outputs


def project_to_transport_set(p: jnp.ndarray, q: jnp.ndarray,
                             pi: jnp.ndarray) -> jnp.ndarray:
    """投影到传输约束集"""
    pi = pi / jnp.sum(pi)

    row_sums = pi.sum(axis=1)
    a = jnp.minimum(1.0, jnp.where(row_sums > 0, p / row_sums, 1.0))
    pi_prime = a[:, None] * pi

    col_sums = pi_prime.sum(axis=0)
    b = jnp.minimum(1.0, jnp.where(col_sums > 0, q / col_sums, 1.0))
    pi_double_prime = pi_prime * b[None, :]

    delta_p = jnp.maximum(p - pi_double_prime.sum(axis=1), 0)
    delta_q = jnp.maximum(q - pi_double_prime.sum(axis=0), 0)

    correction = jnp.outer(delta_p, delta_q) / (jnp.sum(delta_p) + 1e-10)

    return pi_double_prime + correction
