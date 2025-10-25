"""
Beta退火策略
提供各种预定义的退火schedule和自定义接口
"""
import jax.numpy as jnp
from typing import Callable

# Beta schedule类型别名
BetaSchedule = Callable[[int], float]


def constant_schedule(beta: float) -> BetaSchedule:
    """
    常数schedule（标准Sinkhorn）
    
    β_t = β (常数)
    
    参数：
        beta: 固定的逆温度
    """

    def schedule(t: int) -> float:
        return beta

    return schedule


def polynomial_schedule(beta0: float, kappa: float) -> BetaSchedule:
    """
    多项式退火schedule（论文推荐）
    
    β_t = β_0 * (1 + t)^κ
    
    参数：
        beta0: 初始温度
        kappa: 退火指数
            - κ = 0: 常数（标准Sinkhorn）
            - κ = 0.5: 论文推荐的标准Annealed
            - κ = 2/3: 论文推荐的Debiased Annealed
    
    示例：
        >>> schedule = polynomial_schedule(beta0=10.0, kappa=0.5)
        >>> beta_10 = schedule(10)  # β_10 = 10 * 11^0.5
    """

    def schedule(t: int) -> float:
        t_float = jnp.array(t, dtype=jnp.float32)
        return beta0 * ((1.0 + t_float)**kappa)

    return schedule


def exponential_schedule(beta0: float, rate: float) -> BetaSchedule:
    """
    指数退火schedule
    
    β_t = β_0 * exp(rate * t)
    
    参数：
        beta0: 初始温度
        rate: 增长率
    """

    def schedule(t: int) -> float:
        t_float = jnp.array(t, dtype=jnp.float32)
        return beta0 * jnp.exp(rate * t_float)

    return schedule


def piecewise_constant_schedule(beta_values: list[float],
                                breakpoints: list[int]) -> BetaSchedule:
    """
    分段常数schedule
    
    参数：
        beta_values: beta值列表
        breakpoints: 断点列表（迭代次数）
    
    示例：
        >>> # 前50步β=10, 51-100步β=50, 101+步β=100
        >>> schedule = piecewise_constant_schedule(
        ...     beta_values=[10.0, 50.0, 100.0],
        ...     breakpoints=[50, 100]
        ... )
    """

    def schedule(t: int) -> float:
        beta = beta_values[0]
        for i, bp in enumerate(breakpoints):
            if t > bp:
                beta = beta_values[i + 1]
            else:
                break
        return beta

    return schedule


def adaptive_schedule(beta0: float,
                      target_beta: float,
                      total_iters: int,
                      warmup_ratio: float = 0.1) -> BetaSchedule:
    """
    自适应schedule（带warmup）
    
    - Warmup阶段：线性增长
    - 主阶段：多项式增长到目标值
    
    参数：
        beta0: 初始温度
        target_beta: 目标温度
        total_iters: 总迭代次数
        warmup_ratio: warmup阶段比例
    """
    warmup_iters = int(total_iters * warmup_ratio)

    def schedule(t: int) -> float:
        if t <= warmup_iters:
            # 线性warmup
            alpha = t / warmup_iters
            return beta0 + alpha * (target_beta - beta0) * 0.1
        else:
            # 多项式增长
            progress = (t - warmup_iters) / (total_iters - warmup_iters)
            return beta0 + (target_beta - beta0) * (progress**2)

    return schedule


def cosine_schedule(beta0: float, beta_max: float,
                    total_iters: int) -> BetaSchedule:
    """
    余弦退火schedule
    
    β_t = β_0 + (β_max - β_0) * (1 - cos(πt/T)) / 2
    
    参数：
        beta0: 初始温度
        beta_max: 最大温度
        total_iters: 总迭代次数
    """

    def schedule(t: int) -> float:
        t_float = jnp.array(t, dtype=jnp.float32)
        progress = t_float / total_iters
        cosine_decay = (1.0 - jnp.cos(jnp.pi * progress)) / 2.0
        return beta0 + (beta_max - beta0) * cosine_decay

    return schedule


# 预定义的常用配置
PAPER_ANNEALED = lambda beta0: polynomial_schedule(beta0, kappa=0.5)
PAPER_DEBIASED = lambda beta0: polynomial_schedule(beta0, kappa=2 / 3)
