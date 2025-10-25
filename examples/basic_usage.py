"""
基本使用示例
演示如何使用库进行OT求解
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import ot

from sinkhorn.core import solve_sinkhorn, SinkhornConfig, AlgorithmType
from sinkhorn.schedule import polynomial_schedule, constant_schedule
from sinkhorn.utils import project_to_transport_set
from sinkhorn.metrics import compute_convergence_metrics


def main():
    print("=" * 70)
    print("Annealed Sinkhorn - Example")
    print("=" * 70)

    # ========== 1. 准备数据 ==========
    key = jax.random.PRNGKey(42)
    m, n = 100, 100

    # 生成分布
    key, k1, k2 = jax.random.split(key, 3)
    p = jax.random.uniform(k1, (m, ))
    q = jax.random.uniform(k2, (n, ))
    p = p / p.sum()
    q = q / q.sum()

    # 生成几何代价
    key, k1, k2 = jax.random.split(key, 3)
    x = jax.random.uniform(k1, (m, 2))
    y = jax.random.uniform(k2, (n, 2))
    c = jnp.sum((x[:, None, :] - y[None, :, :])**2, axis=2)
    c = c / (c.max() - c.min())

    print(f"\n问题: m={m}, n={n}")
    print(f"代价矩阵: 范围=[{float(c.min()):.4f}, {float(c.max()):.4f}]")

    # 计算精确OT
    ot_exact = ot.emd2(np.array(p), np.array(q), np.array(c))
    print(f"精确OT代价: {ot_exact:.6f}")

    # ========== 2. 配置算法 ==========
    osc = float(c.max() - c.min())
    beta0 = 10.0 / osc
    num_iters = 1000

    # 配置1：标准Sinkhorn
    config_standard = SinkhornConfig(
        algorithm_type=AlgorithmType.STANDARD,
        num_iterations=num_iters,
        beta_schedule=constant_schedule(beta=1000.0)  # 固定高温度
    )

    # 配置2：Annealed Sinkhorn
    config_annealed = SinkhornConfig(algorithm_type=AlgorithmType.ANNEALED,
                                     num_iterations=num_iters,
                                     beta_schedule=polynomial_schedule(
                                         beta0, kappa=0.5))

    # 配置3：Debiased Annealed Sinkhorn
    config_debiased = SinkhornConfig(algorithm_type=AlgorithmType.DEBIASED,
                                     num_iterations=num_iters,
                                     beta_schedule=polynomial_schedule(
                                         beta0, kappa=2 / 3),
                                     debiasing=True)

    # ========== 3. 运行算法 ==========
    print(f"\n运行算法（{num_iters}次迭代）...")

    print("  - 标准Sinkhorn...")
    out_standard = solve_sinkhorn(c, p, q, config_standard)

    print("  - Annealed Sinkhorn...")
    out_annealed = solve_sinkhorn(c, p, q, config_annealed)

    print("  - Debiased Annealed Sinkhorn...")
    out_debiased = solve_sinkhorn(c, p, q, config_debiased)

    print("✓ 完成")

    # ========== 4. 评估结果 ==========
    print("\n计算收敛指标...")

    metrics_std = compute_convergence_metrics(out_standard, c, p, q, ot_exact)
    metrics_ann = compute_convergence_metrics(out_annealed, c, p, q, ot_exact)
    metrics_deb = compute_convergence_metrics(out_debiased, c, p, q, ot_exact)

    print(f'standard Sinkhorn u: {out_standard.u[:5,:10]}')
    print(f'standard Sinkhorn v: {out_standard.v[:5,:10]}')
    print(f'annealed Sinkhorn u: {out_annealed.u[:5,:10]}')
    print(f'annealed Sinkhorn v: {out_annealed.v[:5,:10]}')

    print("\n最终误差:")

    print(f"  标准Sinkhorn:  {metrics_std['ot_error'][-5:]}")
    print(f"  Annealed:      {metrics_ann['ot_error'][-5:]}")
    print(f"  Debiased:      {metrics_deb['ot_error'][-5:]}")
    print(f"  标准Sinkhorn:  {metrics_std['ot_error'][-1]:.2e}")
    print(f"  Annealed:      {metrics_ann['ot_error'][-1]:.2e}")
    print(f"  Debiased:      {metrics_deb['ot_error'][-1]:.2e}")

    print(f"  标准Sinkhorn:  {metrics_std['p_marginal_error'][-1]:.2e}")
    print(f"  Annealed:      {metrics_ann['p_marginal_error'][-1]:.2e}")
    print(f"  Debiased:      {metrics_deb['p_marginal_error'][-1]:.2e}")

    print(f"  标准Sinkhorn:  {metrics_std['q_marginal_error'][-1]:.2e}")
    print(f"  Annealed:      {metrics_ann['q_marginal_error'][-1]:.2e}")
    print(f"  Debiased:      {metrics_deb['q_marginal_error'][-1]:.2e}")

    # ========== 5. 可视化 ==========
    print("\n绘制结果...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：收敛曲线
    ax = axes[0]
    iters = metrics_std['iterations']
    ax.loglog(iters,
              metrics_std['ot_error'],
              'o-',
              label='标准Sinkhorn',
              alpha=0.7)
    ax.loglog(iters,
              metrics_ann['ot_error'],
              's-',
              label='Annealed Sinkhorn',
              alpha=0.7)
    ax.loglog(iters,
              metrics_deb['ot_error'],
              '^-',
              label='Debiased Annealed',
              alpha=0.7)

    # 理论斜率
    t_ref = np.linspace(50, num_iters, 50)
    ax.loglog(t_ref,
              0.01 * (t_ref**(-0.5)),
              'k--',
              alpha=0.3,
              label='O(t^(-1/2))')
    ax.loglog(t_ref,
              0.01 * (t_ref**(-2 / 3)),
              'k:',
              alpha=0.3,
              label='O(t^(-2/3))')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('OT Suboptimality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('收敛曲线对比')

    # 右图：Beta演化
    ax = axes[1]
    ax.plot(iters, metrics_std['beta_schedule'], label='标准（常数）')
    ax.plot(iters, metrics_ann['beta_schedule'], label='Annealed')
    ax.plot(iters, metrics_deb['beta_schedule'], label='Debiased')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Beta (逆温度)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('温度演化')

    plt.tight_layout()
    plt.savefig('basic_usage_result.png', dpi=150)
    print("✓ 结果已保存: basic_usage_result.png")

    plt.show()

    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
