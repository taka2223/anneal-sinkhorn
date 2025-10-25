"""
测试不同的对偶变量定义方式（修复版）
对比：Julia风格 vs 分离风格
"""
import jax
import jax.numpy as jnp
import numpy as np
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
    u_or_log_a: jnp.ndarray
    v_or_log_b: jnp.ndarray
    beta: float
    t: int


class ScanOutput(NamedTuple):
    var1: jnp.ndarray
    var2: jnp.ndarray
    beta: float
    plan: jnp.ndarray


# ============================================================
# 版本1：Julia风格
# ============================================================
@partial(jax.jit, static_argnames=['debiased', 'num_iters'])
def solve_julia_style(cost_matrix: jnp.ndarray,
                      p: jnp.ndarray,
                      q: jnp.ndarray,
                      beta0: float,
                      kappa: float,
                      num_iters: int,
                      debiased: bool = False):
    """Julia风格：a = exp(u)*p, b = exp(v)*q"""
    m, n = cost_matrix.shape
    log_p = jnp.log(p)
    log_q = jnp.log(q)

    def step_fn(state: ScanState, _):
        u_prev, v_prev, beta_prev, t = state

        # 更新 u
        temp1 = v_prev[None, :] + log_q[None, :] - beta_prev * cost_matrix

        if debiased:
            # ✅ 修复：使用 jnp.where 而不是 Python if
            t_float = jnp.array(t, dtype=jnp.float32)
            correction_coef = (t_float**kappa -
                               (t_float - 1)**kappa) / (t_float**kappa)
            # 当 t <= 1 时，correction_coef 无效（乘以0）
            correction_coef = jnp.where(t > 1, correction_coef, 0.0)
            temp1 = temp1 - correction_coef * u_prev[:, None]

        u_new = -logsumexp_stable(temp1, axis=1, keepdims=False)
        beta_new = beta0 * ((t + 1.0)**kappa)

        # 更新 v
        temp2 = u_new[:, None] + log_p[:, None] - beta_new * cost_matrix
        v_new = -logsumexp_stable(temp2, axis=0, keepdims=False)

        # 计算传输计划
        log_plan = (u_new[:, None] + log_p[:, None] + v_new[None, :] +
                    log_q[None, :] - beta_new * cost_matrix)
        plan = jnp.exp(log_plan)

        next_state = ScanState(u_new, v_new, beta=beta_new, t=t + 1)
        output = ScanOutput(var1=u_new, var2=v_new, beta=beta_new, plan=plan)

        return next_state, output

    initial_state = ScanState(u_or_log_a=jnp.zeros(m),
                              v_or_log_b=jnp.zeros(n),
                              beta=beta0,
                              t=1)

    _, outputs = jax.lax.scan(step_fn,
                              initial_state,
                              xs=None,
                              length=num_iters)
    return outputs


# ============================================================
# 版本2：分离风格
# ============================================================
@partial(jax.jit, static_argnames=['debiased', 'num_iters'])
def solve_separated_style(cost_matrix: jnp.ndarray,
                          p: jnp.ndarray,
                          q: jnp.ndarray,
                          beta0: float,
                          kappa: float,
                          num_iters: int,
                          debiased: bool = False):
    """分离风格：a 和 p 分离"""
    m, n = cost_matrix.shape
    log_p = jnp.log(p)
    log_q = jnp.log(q)

    def step_fn(state: ScanState, _):
        log_a_prev, log_b_prev, beta_prev, t = state

        # 更新 log_a
        temp1 = -beta_prev * cost_matrix + log_b_prev[None, :]
        log_a_standard = log_p - logsumexp_stable(
            temp1, axis=1, keepdims=False)

        if debiased:
            # ✅ 修复：使用 jnp.where
            t_float = jnp.array(t, dtype=jnp.float32)
            # 这里用不同的debiasing公式（适配分离风格）
            # log(a_t) = α * log(a_{t-1}) + log(p) - log(K_{t-1} b_{t-1})
            alpha = 1.0 - ((t_float - 1)**kappa) / (t_float**kappa)
            alpha = jnp.where(t > 1, alpha, 0.0)

            log_a_new = jnp.where(
                t > 1, alpha * log_a_prev + log_p -
                logsumexp_stable(temp1, axis=1, keepdims=False),
                log_a_standard)
        else:
            log_a_new = log_a_standard

        # 更新 beta
        beta_new = beta0 * ((t + 1.0)**kappa)

        # 更新 log_b
        temp2 = -beta_new * cost_matrix + log_a_new[:, None]
        log_b_new = log_q - logsumexp_stable(temp2, axis=0, keepdims=False)

        # 计算传输计划
        log_plan = log_a_new[:, None] - beta_new * cost_matrix + log_b_new[
            None, :]
        plan = jnp.exp(log_plan)

        next_state = ScanState(u_or_log_a=log_a_new,
                               v_or_log_b=log_b_new,
                               beta=beta_new,
                               t=t + 1)
        output = ScanOutput(var1=log_a_new,
                            var2=log_b_new,
                            beta=beta_new,
                            plan=plan)

        return next_state, output

    initial_state = ScanState(u_or_log_a=jnp.zeros(m),
                              v_or_log_b=jnp.zeros(n),
                              beta=beta0,
                              t=1)

    _, outputs = jax.lax.scan(step_fn,
                              initial_state,
                              xs=None,
                              length=num_iters)
    return outputs


# ============================================================
# 主测试函数
# ============================================================
def main():
    print("=" * 70)
    print("对偶变量定义方式测试（修复版）")
    print("=" * 70)

    # 设置
    key = jax.random.PRNGKey(42)
    m, n = 50, 50

    key, k1, k2 = jax.random.split(key, 3)
    p = jax.random.uniform(k1, (m, ))
    q = jax.random.uniform(k2, (n, ))
    p = p / p.sum()
    q = q / q.sum()

    key, k1, k2 = jax.random.split(key, 3)
    x = jax.random.uniform(k1, (m, 2))
    y = jax.random.uniform(k2, (n, 2))
    c = jnp.sum((x[:, None, :] - y[None, :, :])**2, axis=2)
    c = (c - c.min()) / (c.max() - c.min())

    print(f"\n问题设置: m={m}, n={n}")

    osc = float(c.max() - c.min())
    beta0 = 10.0 / osc
    num_iters = 100

    print(f"算法参数: β₀={beta0:.4f}, 迭代={num_iters}")

    # ========== 测试标准版本 ==========
    print("\n" + "=" * 70)
    print("测试 1: 标准版本（无debiasing）")
    print("=" * 70)

    print("\n运行 Julia 风格...")
    out_julia = solve_julia_style(c,
                                  p,
                                  q,
                                  beta0,
                                  0.5,
                                  num_iters,
                                  debiased=False)
    print("✓ 完成")

    print("\n运行 分离风格...")
    out_separated = solve_separated_style(c,
                                          p,
                                          q,
                                          beta0,
                                          0.5,
                                          num_iters,
                                          debiased=False)
    print("✓ 完成")

    # 对比
    print("\n对比结果:")
    beta_diff = jnp.abs(out_julia.beta - out_separated.beta)
    plan_diff = jnp.abs(out_julia.plan - out_separated.plan)

    print(f"  Beta差异: 最大={float(jnp.max(beta_diff)):.2e}, "
          f"平均={float(jnp.mean(beta_diff)):.2e}")
    print(f"  Plan差异: 最大={float(jnp.max(plan_diff)):.2e}, "
          f"平均={float(jnp.mean(plan_diff)):.2e}")

    max_diff_standard = float(jnp.max(plan_diff))

    if max_diff_standard < 1e-6:
        print(f"  ✅ 标准版本结果一致（差异 < 1e-6）")
        standard_ok = True
    elif max_diff_standard < 1e-3:
        print(f"  ⚠️  标准版本有微小差异（1e-6 < 差异 < 1e-3）")
        print(f"      这可能是由于数值精度或舍入误差")
        standard_ok = "minor"
    else:
        print(f"  ❌ 标准版本结果显著不同（差异 > 1e-3）")
        standard_ok = False

    # ========== 测试Debiased版本 ==========
    print("\n" + "=" * 70)
    print("测试 2: Debiased版本")
    print("=" * 70)

    print("\n运行 Julia 风格（debiased）...")
    out_julia_deb = solve_julia_style(c,
                                      p,
                                      q,
                                      beta0,
                                      2 / 3,
                                      num_iters,
                                      debiased=True)
    print("✓ 完成")

    print("\n运行 分离风格（debiased）...")
    out_sep_deb = solve_separated_style(c,
                                        p,
                                        q,
                                        beta0,
                                        2 / 3,
                                        num_iters,
                                        debiased=True)
    print("✓ 完成")

    # 对比
    print("\n对比结果:")
    beta_diff_deb = jnp.abs(out_julia_deb.beta - out_sep_deb.beta)
    plan_diff_deb = jnp.abs(out_julia_deb.plan - out_sep_deb.plan)

    print(f"  Beta差异: 最大={float(jnp.max(beta_diff_deb)):.2e}, "
          f"平均={float(jnp.mean(beta_diff_deb)):.2e}")
    print(f"  Plan差异: 最大={float(jnp.max(plan_diff_deb)):.2e}, "
          f"平均={float(jnp.mean(plan_diff_deb)):.2e}")

    max_diff_debiased = float(jnp.max(plan_diff_deb))

    if max_diff_debiased < 1e-6:
        print(f"  ✅ Debiased版本结果一致（差异 < 1e-6）")
        debiased_ok = True
    elif max_diff_debiased < 1e-3:
        print(f"  ⚠️  Debiased版本有微小差异（1e-6 < 差异 < 1e-3）")
        debiased_ok = "minor"
    else:
        print(f"  ❌ Debiased版本结果显著不同（差异 > 1e-3）")
        debiased_ok = False

    # ========== 详细检查 ==========
    print("\n" + "=" * 70)
    print("详细分析：每一步的差异演化")
    print("=" * 70)

    print("\n标准版本（每10步）:")
    print("  步数    Plan最大差异    相对误差")
    for t in [0, 9, 19, 29, 49, 99]:
        if t < num_iters:
            diff = float(
                jnp.max(jnp.abs(out_julia.plan[t] - out_separated.plan[t])))
            plan_max = float(jnp.max(out_julia.plan[t]))
            rel_err = diff / plan_max if plan_max > 0 else 0
            print(f"  {t+1:3d}     {diff:.2e}         {rel_err:.2e}")

    print("\nDebiased版本（每10步）:")
    print("  步数    Plan最大差异    相对误差")
    for t in [0, 9, 19, 29, 49, 99]:
        if t < num_iters:
            diff = float(
                jnp.max(jnp.abs(out_julia_deb.plan[t] - out_sep_deb.plan[t])))
            plan_max = float(jnp.max(out_julia_deb.plan[t]))
            rel_err = diff / plan_max if plan_max > 0 else 0
            print(f"  {t+1:3d}     {diff:.2e}         {rel_err:.2e}")

    # ========== 结论 ==========
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    if standard_ok == True and debiased_ok == True:
        print("✅ 两种对偶变量定义方式在机器精度内一致")
        print("✅ 对偶变量定义不是问题根源")
    elif standard_ok in [True, "minor"] and debiased_ok in [True, "minor"]:
        print("⚠️  两种定义产生了微小差异（< 1e-3）")
        print("   这可能是由于：")
        print("   - 浮点运算的顺序不同导致舍入误差累积")
        print("   - 数值稳定性的微小差异")
        print("   但差异很小，不足以解释最初实现的问题")
    else:
        print("❌ 两种定义产生了显著差异")
        print("   对偶变量的定义方式确实很重要！")
        print("   这可能是最初实现问题的重要线索")

    # 额外信息
    print(f"\n数值细节:")
    print(f"  标准版本最大差异: {max_diff_standard:.2e}")
    print(f"  Debiased版本最大差异: {max_diff_debiased:.2e}")
    print(
        f"  两者比率: {max_diff_debiased/max_diff_standard if max_diff_standard > 0 else float('inf'):.2f}x"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
