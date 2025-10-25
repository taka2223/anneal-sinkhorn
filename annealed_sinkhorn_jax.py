"""
基于Julia参考实现的Annealed Sinkhorn
严格对应论文和参考实现
"""
import jax.numpy as jnp
import numpy as np
from typing import Tuple


def logsumexp_stable(x, axis, keepdims=False):
    """数值稳定的log-sum-exp"""
    max_x = jnp.max(x, axis=axis, keepdims=True)
    result = max_x + jnp.log(
        jnp.sum(jnp.exp(x - max_x), axis=axis, keepdims=True))
    if not keepdims:
        result = jnp.squeeze(result, axis=axis)
    return result


def solve_annealed_sinkhorn(
    cost_matrix: jnp.ndarray,
    p: jnp.ndarray,
    q: jnp.ndarray,
    beta0: float,
    kappa: float,
    num_iters: int,
    debiased: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Annealed Sinkhorn算法（Julia参考实现的Python版本）
    
    对偶变量定义：
        a[i] = exp(u[i]) * p[i]
        b[j] = exp(v[j]) * q[j]
        π[i,j] = a[i] * exp(-β*c[i,j]) * b[j]
    
    参数：
        cost_matrix: (m, n) 代价矩阵
        p: (m,) 源分布
        q: (n,) 目标分布
        beta0: 初始温度
        kappa: 退火指数
        num_iters: 迭代次数
        debiased: 是否去偏
    
    返回：
        history_u: (num_iters, m) u的历史
        history_v: (num_iters, n) v的历史
        history_beta: (num_iters,) β的历史
        history_plans: (num_iters, m, n) π的历史
    """
    m, n = cost_matrix.shape

    # 初始化对偶变量
    u = jnp.zeros(m)  # (m,) 列向量
    v = jnp.zeros(n)  # (n,) 行向量

    # log概率
    log_p = jnp.log(p)
    log_q = jnp.log(q)

    # 当前温度
    beta = beta0

    # 存储历史
    history_u = []
    history_v = []
    history_beta = []
    history_plans = []

    for t in range(1, num_iters + 1):
        # ============ 更新 u ============
        # temp1[i,j] = v[j] + log(q[j]) - β*c[i,j]
        temp1 = v[None, :] + log_q[None, :] - beta * cost_matrix  # (m, n)

        if debiased and t > 1:
            # 计算修正系数：(t^κ - (t-1)^κ) / t^κ
            # 注意：β_t = β0 * (t+1)^κ，所以这里用t而不是t+1
            correction_coef = (t**kappa - (t - 1)**kappa) / (t**kappa)
            # temp1 -= correction_coef * u
            temp1 = temp1 - correction_coef * u[:, None]

        # u[i] = -log(Σ_j exp(temp1[i,j]))
        u = -logsumexp_stable(temp1, axis=1, keepdims=False)

        # ============ 更新 β ============
        # 注意：Julia中β在u和v之间更新！
        beta = beta0 * ((t + 1)**kappa)

        # ============ 更新 v ============
        # temp2[i,j] = u[i] + log(p[i]) - β*c[i,j]
        temp2 = u[:, None] + log_p[:, None] - beta * cost_matrix  # (m, n)

        # v[j] = -log(Σ_i exp(temp2[i,j]))
        v = -logsumexp_stable(temp2, axis=0, keepdims=False)

        # ============ 计算传输计划 ============
        # π[i,j] = exp(u[i] + log(p[i]) + v[j] + log(q[j]) - β*c[i,j])
        log_plan = u[:, None] + log_p[:, None] + v[None, :] + log_q[
            None, :] - beta * cost_matrix
        plan = jnp.exp(log_plan)

        # 保存历史
        history_u.append(u)
        history_v.append(v)
        history_beta.append(beta)
        history_plans.append(plan)

    return (jnp.stack(history_u), jnp.stack(history_v),
            jnp.array(history_beta), jnp.stack(history_plans))


def project_to_transport_set(p: jnp.ndarray, q: jnp.ndarray,
                             pi: jnp.ndarray) -> jnp.ndarray:
    """投影到传输约束集"""
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
