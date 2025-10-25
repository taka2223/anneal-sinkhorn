"""
评估指标和诊断工具
"""
import jax.numpy as jnp
import numpy as np
from typing import Dict, List
from sinkhorn.core import SinkhornOutput
from sinkhorn.utils import project_to_transport_set, compute_ot_error


def compute_convergence_metrics(output: SinkhornOutput,
                                cost_matrix: jnp.ndarray, p: jnp.ndarray,
                                q: jnp.ndarray,
                                ot_cost_exact: float) -> Dict[str, np.ndarray]:
    """
    计算收敛指标
    
    返回：
        包含各种指标的字典
    """
    num_iters = output.num_iterations

    errors = []
    relative_errors = []
    p_marginal_errors = []
    q_marginal_errors = []

    for t in range(num_iters):
        plan = output.transport_plans[t]

        # OT误差
        abs_err, rel_err = compute_ot_error(plan, cost_matrix, p, q,
                                            ot_cost_exact)
        errors.append(abs_err)
        relative_errors.append(rel_err)

        # 边际约束误差
        p_marg = plan.sum(axis=1)
        q_marg = plan.sum(axis=0)
        p_marginal_errors.append(float(jnp.max(jnp.abs(p_marg - p))))
        q_marginal_errors.append(float(jnp.max(jnp.abs(q_marg - q))))

    return {
        'ot_error': np.array(errors),
        'relative_error': np.array(relative_errors),
        'p_marginal_error': np.array(p_marginal_errors),
        'q_marginal_error': np.array(q_marginal_errors),
        'beta_schedule': np.array(output.beta),
        'iterations': np.arange(1, num_iters + 1)
    }


def estimate_convergence_rate(errors: np.ndarray, window: int = 20) -> float:
    """
    估计收敛率（在log-log图中的斜率）
    
    参数：
        errors: 误差序列
        window: 用于估计的窗口大小
    """
    if len(errors) < window:
        return np.nan

    # 取最后window个点
    recent_errors = errors[-window:]
    recent_iters = np.arange(len(errors) - window + 1, len(errors) + 1)

    # 过滤非正值
    valid = recent_errors > 0
    if valid.sum() < 2:
        return np.nan

    log_errors = np.log(recent_errors[valid])
    log_iters = np.log(recent_iters[valid])

    # 线性拟合
    slope = np.polyfit(log_iters, log_errors, 1)[0]

    return slope


def compare_algorithms(outputs: Dict[str,
                                     SinkhornOutput], cost_matrix: jnp.ndarray,
                       p: jnp.ndarray, q: jnp.ndarray,
                       ot_cost_exact: float) -> Dict[str, Dict]:
    """
    对比多个算法的性能
    
    参数：
        outputs: 算法名称 -> 输出 的字典
        cost_matrix, p, q: 问题参数
        ot_cost_exact: 精确OT代价
    
    返回：
        每个算法的指标字典
    """
    results = {}

    for name, output in outputs.items():
        metrics = compute_convergence_metrics(output, cost_matrix, p, q,
                                              ot_cost_exact)

        # 额外统计
        final_error = metrics['ot_error'][-1]
        convergence_rate = estimate_convergence_rate(metrics['ot_error'])

        results[name] = {
            'metrics': metrics,
            'final_error': final_error,
            'convergence_rate': convergence_rate,
            'iterations': output.num_iterations
        }

    return results
