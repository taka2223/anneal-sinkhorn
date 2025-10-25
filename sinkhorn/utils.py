"""
实用工具函数
包括：投影、误差计算、可视化等
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple


@jax.jit
def project_to_transport_set(p: jnp.ndarray, q: jnp.ndarray,
                             pi: jnp.ndarray) -> jnp.ndarray:
    """
    投影到传输约束集 Γ(p,q)
    实现 Algorithm 2 [Altschuler et al., 2017]
    
    参数：
        p: (m,) 源分布
        q: (n,) 目标分布
        pi: (m, n) 待投影的矩阵
    
    返回：
        (m, n) 投影后的传输计划
    """
    # 归一化
    pi = pi / jnp.sum(pi)

    # 第一步：缩放行
    row_sums = pi.sum(axis=1)
    a = jnp.minimum(1.0, jnp.where(row_sums > 0, p / row_sums, 1.0))
    pi_prime = a[:, None] * pi

    # 第二步：缩放列
    col_sums = pi_prime.sum(axis=0)
    b = jnp.minimum(1.0, jnp.where(col_sums > 0, q / col_sums, 1.0))
    pi_double_prime = pi_prime * b[None, :]

    # 第三步：修正残差
    delta_p = jnp.maximum(p - pi_double_prime.sum(axis=1), 0)
    delta_q = jnp.maximum(q - pi_double_prime.sum(axis=0), 0)

    correction = jnp.outer(delta_p, delta_q) / (jnp.sum(delta_p) + 1e-10)

    return pi_double_prime + correction


def compute_ot_error(transport_plan: jnp.ndarray, cost_matrix: jnp.ndarray,
                     p: jnp.ndarray, q: jnp.ndarray,
                     ot_cost_exact: float) -> Tuple[float, float]:
    """
    计算OT误差
    
    返回：
        (absolute_error, relative_error)
    """
    # 投影到约束集
    plan_projected = project_to_transport_set(p, q, transport_plan)

    # 计算代价
    cost = float(jnp.sum(plan_projected * cost_matrix))

    # 计算误差
    abs_error = cost - ot_cost_exact
    rel_error = abs_error / abs(
        ot_cost_exact) if ot_cost_exact != 0 else abs_error

    return abs_error, rel_error


def check_marginal_constraints(
        transport_plan: jnp.ndarray,
        p: jnp.ndarray,
        q: jnp.ndarray,
        tolerance: float = 1e-6) -> Tuple[bool, float, float]:
    """
    检查边际约束是否满足
    
    返回：
        (satisfied, p_error, q_error)
    """
    p_marginal = transport_plan.sum(axis=1)
    q_marginal = transport_plan.sum(axis=0)

    p_error = float(jnp.max(jnp.abs(p_marginal - p)))
    q_error = float(jnp.max(jnp.abs(q_marginal - q)))

    satisfied = (p_error < tolerance) and (q_error < tolerance)

    return satisfied, p_error, q_error


def compute_wasserstein_distance(x: jnp.ndarray,
                                 y: jnp.ndarray,
                                 p: jnp.ndarray,
                                 q: jnp.ndarray,
                                 power: float = 2.0) -> float:
    """
    计算Wasserstein距离（对于点云）
    
    参数：
        x: (m, d) 源点云
        y: (n, d) 目标点云
        p: (m,) 源权重
        q: (n,) 目标权重
        power: Wasserstein距离的幂次（通常是1或2）
    """
    from sinkhorn.core import solve_sinkhorn, SinkhornConfig, AlgorithmType
    from sinkhorn.schedule import polynomial_schedule

    # 计算代价矩阵
    cost_matrix = jnp.sum((x[:, None, :] - y[None, :, :])**2, axis=2)
    if power != 2:
        cost_matrix = cost_matrix**(power / 2)

    # 使用高精度Sinkhorn
    config = SinkhornConfig(algorithm_type=AlgorithmType.ANNEALED,
                            num_iterations=1000,
                            beta_schedule=polynomial_schedule(beta0=100.0,
                                                              kappa=0.5))

    output = solve_sinkhorn(cost_matrix, p, q, config)
    final_plan = output.transport_plans[-1]

    distance = float(jnp.sum(final_plan * cost_matrix))
    if power != 1:
        distance = distance**(1 / power)

    return distance
