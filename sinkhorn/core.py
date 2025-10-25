"""
核心Sinkhorn算法实现
支持：标准Sinkhorn、Annealed Sinkhorn、Debiased Annealed Sinkhorn
"""
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Callable, Optional
from enum import Enum


class AlgorithmType(Enum):
    """算法类型"""
    STANDARD = "standard"  # 固定beta的标准Sinkhorn
    ANNEALED = "annealed"  # 退火Sinkhorn
    DEBIASED = "debiased"  # 去偏退火Sinkhorn


def logsumexp_stable(x: jnp.ndarray,
                     axis: int,
                     keepdims: bool = False) -> jnp.ndarray:
    """数值稳定的log-sum-exp"""
    max_x = jnp.max(x, axis=axis, keepdims=True)
    result = max_x + jnp.log(
        jnp.sum(jnp.exp(x - max_x), axis=axis, keepdims=True))
    if not keepdims:
        result = jnp.squeeze(result, axis=axis)
    return result


class SinkhornState(NamedTuple):
    """Sinkhorn迭代状态"""
    u: jnp.ndarray  # (m,) 对偶变量u，满足 a = exp(u) * p
    v: jnp.ndarray  # (n,) 对偶变量v，满足 b = exp(v) * q
    beta: float  # 当前逆温度
    iteration: int  # 当前迭代次数


class SinkhornOutput(NamedTuple):
    """算法输出"""
    u: jnp.ndarray  # (T, m) u的历史
    v: jnp.ndarray  # (T, n) v的历史
    beta: jnp.ndarray  # (T,) beta的历史
    transport_plans: jnp.ndarray  # (T, m, n) 传输计划的历史
    converged: bool  # 是否收敛
    num_iterations: int  # 实际迭代次数


class SinkhornConfig(NamedTuple):
    """Sinkhorn算法配置"""
    algorithm_type: AlgorithmType
    num_iterations: int
    beta_schedule: Callable[[int], float]  # t -> beta_t
    debiasing: bool = False
    tolerance: Optional[float] = None  # 收敛容差（可选）
    log_frequency: int = 0  # 日志频率，0表示不记录


@partial(jax.jit, static_argnames=['config'])
def solve_sinkhorn(cost_matrix: jnp.ndarray, p: jnp.ndarray, q: jnp.ndarray,
                   config: SinkhornConfig) -> SinkhornOutput:
    """
    通用Sinkhorn求解器
    
    参数：
        cost_matrix: (m, n) 代价矩阵
        p: (m,) 源分布
        q: (n,) 目标分布
        config: 算法配置
    
    返回：
        SinkhornOutput 包含完整的求解历史
    
    示例：
        >>> from sinkhorn.schedules import polynomial_schedule
        >>> config = SinkhornConfig(
        ...     algorithm_type=AlgorithmType.ANNEALED,
        ...     num_iterations=200,
        ...     beta_schedule=polynomial_schedule(beta0=10.0, kappa=0.5)
        ... )
        >>> output = solve_sinkhorn(c, p, q, config)
    """
    m, n = cost_matrix.shape
    log_p = jnp.log(p)
    log_q = jnp.log(q)

    # 提取配置
    num_iters = config.num_iterations
    debiasing = config.debiasing
    beta_schedule = config.beta_schedule

    def step_fn(state: SinkhornState, _):
        """单步Sinkhorn迭代（Julia风格）"""
        u_prev, v_prev, beta_prev, t = state

        # ============ 更新 u ============
        # u[i] = -log(Σ_j exp(v[j] + log(q[j]) - β*c[i,j]))
        temp1 = v_prev[None, :] + log_q[None, :] - beta_prev * cost_matrix

        if debiasing:
            # Debiasing修正（基于Julia参考实现）
            t_float = jnp.array(t, dtype=jnp.float32)
            beta_t = beta_schedule(t)
            beta_tm1 = beta_schedule(t - 1)

            # correction = (β_t - β_{t-1}) / β_t
            correction_coef = (beta_t - beta_tm1) / beta_t
            correction_coef = jnp.where(t > 1, correction_coef, 0.0)
            temp1 = temp1 - correction_coef * u_prev[:, None]

        u_new = -logsumexp_stable(temp1, axis=1, keepdims=False)

        # ============ 更新 beta ============
        beta_new = beta_schedule(t + 1)

        # ============ 更新 v ============
        # v[j] = -log(Σ_i exp(u[i] + log(p[i]) - β*c[i,j]))
        temp2 = u_new[:, None] + log_p[:, None] - beta_new * cost_matrix
        v_new = -logsumexp_stable(temp2, axis=0, keepdims=False)

        # ============ 计算传输计划 ============
        # π[i,j] = exp(u[i] + log(p[i]) + v[j] + log(q[j]) - β*c[i,j])
        log_plan = (u_new[:, None] + log_p[:, None] + v_new[None, :] +
                    log_q[None, :] - beta_new * cost_matrix)
        plan = jnp.exp(log_plan)

        # 下一状态
        next_state = SinkhornState(u=u_new,
                                   v=v_new,
                                   beta=beta_new,
                                   iteration=t + 1)

        # 输出
        output = (u_new, v_new, beta_new, plan)

        return next_state, output

    # 初始状态
    initial_beta = beta_schedule(1)
    initial_state = SinkhornState(u=jnp.zeros(m),
                                  v=jnp.zeros(n),
                                  beta=initial_beta,
                                  iteration=1)

    # 运行迭代
    final_state, history = jax.lax.scan(step_fn,
                                        initial_state,
                                        xs=None,
                                        length=num_iters)

    # 解包历史
    hist_u, hist_v, hist_beta, hist_plans = history

    return SinkhornOutput(
        u=hist_u,
        v=hist_v,
        beta=hist_beta,
        transport_plans=hist_plans,
        converged=True,  # 简化版本，总是返回True
        num_iterations=num_iters)


@jax.jit
def get_transport_plan(u: jnp.ndarray, v: jnp.ndarray, beta: float,
                       cost_matrix: jnp.ndarray, p: jnp.ndarray,
                       q: jnp.ndarray) -> jnp.ndarray:
    """
    从对偶变量重建传输计划
    
    π[i,j] = exp(u[i] + log(p[i]) + v[j] + log(q[j]) - β*c[i,j])
    """
    log_p = jnp.log(p)
    log_q = jnp.log(q)
    log_plan = (u[:, None] + log_p[:, None] + v[None, :] + log_q[None, :] -
                beta * cost_matrix)
    return jnp.exp(log_plan)
