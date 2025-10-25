"""
对比测试：循环版本 vs Scan版本
验证两者是否产生相同结果
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import ot
import time

from annealed_sinkhorn_jax import solve_annealed_sinkhorn as solve_loop
from annealed_sinkhorn_scan import (solve_annealed_sinkhorn_scan as solve_scan,
                                    project_to_transport_set)


def main():
    print("=" * 70)
    print("对比测试：循环版本 vs Scan版本")
    print("=" * 70)

    # ========== 设置 ==========
    key = jax.random.PRNGKey(42)
    m, n = 100, 100

    # 生成分布
    key, k1, k2 = jax.random.split(key, 3)
    p = jax.random.uniform(k1, (m, ))
    q = jax.random.uniform(k2, (n, ))
    p = p / p.sum()
    q = q / q.sum()

    # 生成代价矩阵
    key, k1, k2 = jax.random.split(key, 3)
    x = jax.random.uniform(k1, (m, 2))
    y = jax.random.uniform(k2, (n, 2))
    # c = jnp.sum((x[:, None, :] - y[None, :, :])**2, axis=2)
    c = jax.random.normal(key, (m, n))
    c = c / (c.max() - c.min())
    # c = (c - c.min()) / (c.max() - c.min())

    print(f"\n问题设置: m={m}, n={n}")
    print(f"代价矩阵范围: [{float(c.min()):.4f}, {float(c.max()):.4f}]")

    # 计算精确OT
    ot_cost_exact = ot.emd2(np.array(p), np.array(q), np.array(c))
    print(f"精确OT代价: {ot_cost_exact:.6f}")

    # ========== 参数 ==========
    osc = float(c.max() - c.min())
    beta0 = 10.0 / osc
    num_iters = 200
    kappa = 0.5

    print(f"\n算法参数:")
    print(f"  β₀ = {beta0:.4f}")
    print(f"  κ = {kappa}")
    print(f"  迭代次数 = {num_iters}")

    # ========== 运行循环版本 ==========
    print("\n" + "-" * 70)
    print("运行循环版本...")
    print("-" * 70)

    start = time.time()
    hist_u_loop, hist_v_loop, hist_beta_loop, hist_plans_loop = solve_loop(
        c, p, q, beta0, kappa, num_iters, debiased=False)
    time_loop = time.time() - start
    print(f"✓ 完成，用时: {time_loop:.3f}秒")

    # ========== 运行Scan版本 ==========
    print("\n" + "-" * 70)
    print("运行Scan版本（首次编译）...")
    print("-" * 70)

    # 第一次运行（包含编译时间）
    start = time.time()
    outputs_scan = solve_scan(c, p, q, beta0, kappa, num_iters, debiased=False)
    time_scan_first = time.time() - start
    print(f"✓ 完成，用时: {time_scan_first:.3f}秒（包含JIT编译）")

    # 第二次运行（纯执行时间）
    print("\n运行Scan版本（已编译）...")
    start = time.time()
    outputs_scan = solve_scan(c, p, q, beta0, kappa, num_iters, debiased=False)
    time_scan_second = time.time() - start
    print(f"✓ 完成，用时: {time_scan_second:.3f}秒")

    # 提取scan的结果
    hist_u_scan = outputs_scan.u
    hist_v_scan = outputs_scan.v
    hist_beta_scan = outputs_scan.beta
    hist_plans_scan = outputs_scan.plan

    # ========== 验证一致性 ==========
    print("\n" + "=" * 70)
    print("验证结果一致性")
    print("=" * 70)

    # 检查形状
    print(f"\n形状检查:")
    print(
        f"  循环: u={hist_u_loop.shape}, beta={hist_beta_loop.shape}, plans={hist_plans_loop.shape}"
    )
    print(
        f"  Scan: u={hist_u_scan.shape}, beta={hist_beta_scan.shape}, plans={hist_plans_scan.shape}"
    )

    # 检查beta历史
    beta_diff = jnp.abs(hist_beta_loop - hist_beta_scan)
    print(f"\nBeta差异:")
    print(f"  最大差异: {float(jnp.max(beta_diff)):.2e}")
    print(f"  平均差异: {float(jnp.mean(beta_diff)):.2e}")
    print(
        f"  循环: [{hist_beta_loop[0]:.2f}, {hist_beta_loop[1]:.2f}, ..., {hist_beta_loop[-1]:.2f}]"
    )
    print(
        f"  Scan: [{hist_beta_scan[0]:.2f}, {hist_beta_scan[1]:.2f}, ..., {hist_beta_scan[-1]:.2f}]"
    )

    # 检查u历史
    u_diff = jnp.abs(hist_u_loop - hist_u_scan)
    print(f"\nU差异:")
    print(f"  最大差异: {float(jnp.max(u_diff)):.2e}")
    print(f"  平均差异: {float(jnp.mean(u_diff)):.2e}")

    # 检查plans
    plans_diff = jnp.abs(hist_plans_loop - hist_plans_scan)
    print(f"\nPlans差异:")
    print(f"  最大差异: {float(jnp.max(plans_diff)):.2e}")
    print(f"  平均差异: {float(jnp.mean(plans_diff)):.2e}")

    # ========== 计算误差 ==========
    print("\n计算OT误差...")

    def compute_errors(hist_plans, name):
        errors = []
        for t in range(num_iters):
            pi_proj = project_to_transport_set(p, q, hist_plans[t])
            cost = float(jnp.sum(pi_proj * c))
            errors.append(cost - ot_cost_exact)
        errors = np.array(errors)
        print(
            f"  {name}: 最终误差={errors[-1]:.2e}, 范围=[{errors.min():.2e}, {errors.max():.2e}]"
        )
        return errors

    errors_loop = compute_errors(hist_plans_loop, "循环")
    errors_scan = compute_errors(hist_plans_scan, "Scan")

    error_diff = np.abs(errors_loop - errors_scan)
    print(f"\n误差曲线差异:")
    print(f"  最大差异: {error_diff.max():.2e}")
    print(f"  平均差异: {error_diff.mean():.2e}")

    # ========== 性能对比 ==========
    print("\n" + "=" * 70)
    print("性能对比")
    print("=" * 70)
    print(f"  循环版本:        {time_loop:.3f}秒")
    print(f"  Scan（首次）:    {time_scan_first:.3f}秒 (含JIT编译)")
    print(f"  Scan（第二次）:  {time_scan_second:.3f}秒")
    print(f"  加速比:          {time_loop/time_scan_second:.2f}x")

    # ========== 绘图对比 ==========
    print("\n绘制对比图...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    iterations = np.arange(1, num_iters + 1)

    # 左图：误差对比
    ax = axes[0]
    ax.loglog(iterations,
              errors_loop,
              'o-',
              linewidth=2,
              markersize=3,
              alpha=0.7,
              label='循环版本')
    ax.loglog(iterations,
              errors_scan,
              's--',
              linewidth=2,
              markersize=3,
              alpha=0.7,
              label='Scan版本')
    ax.set_xlabel('Iteration t', fontsize=11)
    ax.set_ylabel('OT suboptimality', fontsize=11)
    ax.set_title('误差收敛对比', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：差异
    ax = axes[1]
    ax.semilogy(iterations, error_diff, 'r-', linewidth=2, alpha=0.7)
    ax.set_xlabel('Iteration t', fontsize=11)
    ax.set_ylabel('|Error(Loop) - Error(Scan)|', fontsize=11)
    ax.set_title('两个版本的差异', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1e-10, color='k', linestyle='--', alpha=0.3, label='机器精度')
    ax.legend()

    plt.tight_layout()
    plt.savefig('loop_vs_scan_comparison.png', dpi=150)
    print("✓ 对比图已保存: loop_vs_scan_comparison.png")

    # ========== 结论 ==========
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    if jnp.max(plans_diff) < 1e-6:
        print("✓ 两个版本产生了几乎相同的结果（差异 < 1e-6）")
        print("✓ Scan实现是正确的！")
    else:
        print("⚠ 两个版本存在显著差异")
        print("  需要进一步调查...")

    if time_scan_second < time_loop:
        print(f"✓ Scan版本更快（加速{time_loop/time_scan_second:.2f}x）")
    else:
        print(f"  循环版本更快（对于小规模问题，JIT开销可能超过收益）")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
