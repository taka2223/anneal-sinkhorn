"""
测试闭包捕获是否会导致问题
对比：显式传递 vs 闭包捕获
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
    u: jnp.ndarray
    v: jnp.ndarray
    beta: float
    t: int


class ScanOutput(NamedTuple):
    u: jnp.ndarray
    v: jnp.ndarray
    beta: float
    plan: jnp.ndarray


# ============================================================
# 版本1：显式传递（我们知道这个版本工作正常）
# ============================================================
@partial(jax.jit, static_argnames=['debiased', 'num_iters'])
def solve_explicit(cost_matrix: jnp.ndarray,
                   p: jnp.ndarray,
                   q: jnp.ndarray,
                   beta0: float,
                   kappa: float,
                   num_iters: int,
                   debiased: bool = False):
    """显式传递所有参数的版本"""
    m, n = cost_matrix.shape
    log_p = jnp.log(p)
    log_q = jnp.log(q)

    # ✅ cost_matrix 作为局部变量
    c = cost_matrix

    def step_fn(state: ScanState, _):
        u_prev, v_prev, beta_prev, t = state

        # ✅ 直接使用局部变量 c
        temp1 = v_prev[None, :] + log_q[None, :] - beta_prev * c

        if debiased and t > 1:
            t_float = jnp.array(t, dtype=jnp.float32)
            correction_coef = (t_float**kappa -
                               (t_float - 1)**kappa) / (t_float**kappa)
            temp1 = temp1 - correction_coef * u_prev[:, None]

        u_new = -logsumexp_stable(temp1, axis=1, keepdims=False)
        beta_new = beta0 * ((t + 1.0)**kappa)

        # ✅ 直接使用局部变量 c
        temp2 = u_new[:, None] + log_p[:, None] - beta_new * c
        v_new = -logsumexp_stable(temp2, axis=0, keepdims=False)

        # ✅ 直接使用局部变量 c
        log_plan = (u_new[:, None] + log_p[:, None] + v_new[None, :] +
                    log_q[None, :] - beta_new * c)
        plan = jnp.exp(log_plan)

        next_state = ScanState(u=u_new, v=v_new, beta=beta_new, t=t + 1)
        output = ScanOutput(u=u_new, v=v_new, beta=beta_new, plan=plan)

        return next_state, output

    initial_state = ScanState(u=jnp.zeros(m), v=jnp.zeros(n), beta=beta0, t=1)

    final_state, outputs = jax.lax.scan(step_fn,
                                        initial_state,
                                        xs=None,
                                        length=num_iters)

    return outputs


# ============================================================
# 版本2：闭包捕获（测试这是否会导致问题）
# ============================================================
@partial(jax.jit, static_argnames=['debiased', 'num_iters'])
def solve_closure(cost_matrix: jnp.ndarray,
                  p: jnp.ndarray,
                  q: jnp.ndarray,
                  beta0: float,
                  kappa: float,
                  num_iters: int,
                  debiased: bool = False):
    """使用闭包捕获cost_matrix的版本"""
    m, n = cost_matrix.shape
    log_p = jnp.log(p)
    log_q = jnp.log(q)

    # ⚠️ 不创建局部变量，直接在闭包中引用cost_matrix

    def step_fn(state: ScanState, _):
        u_prev, v_prev, beta_prev, t = state

        # ⚠️ 闭包捕获：直接使用外部的 cost_matrix
        temp1 = v_prev[None, :] + log_q[None, :] - beta_prev * cost_matrix

        if debiased and t > 1:
            t_float = jnp.array(t, dtype=jnp.float32)
            correction_coef = (t_float**kappa -
                               (t_float - 1)**kappa) / (t_float**kappa)
            temp1 = temp1 - correction_coef * u_prev[:, None]

        u_new = -logsumexp_stable(temp1, axis=1, keepdims=False)
        beta_new = beta0 * ((t + 1.0)**kappa)

        # ⚠️ 闭包捕获
        temp2 = u_new[:, None] + log_p[:, None] - beta_new * cost_matrix
        v_new = -logsumexp_stable(temp2, axis=0, keepdims=False)

        # ⚠️ 闭包捕获
        log_plan = (u_new[:, None] + log_p[:, None] + v_new[None, :] +
                    log_q[None, :] - beta_new * cost_matrix)
        plan = jnp.exp(log_plan)

        next_state = ScanState(u=u_new, v=v_new, beta=beta_new, t=t + 1)
        output = ScanOutput(u=u_new, v=v_new, beta=beta_new, plan=plan)

        return next_state, output

    initial_state = ScanState(u=jnp.zeros(m), v=jnp.zeros(n), beta=beta0, t=1)

    final_state, outputs = jax.lax.scan(step_fn,
                                        initial_state,
                                        xs=None,
                                        length=num_iters)

    return outputs


# ============================================================
# 版本3：嵌套闭包（更复杂的场景）
# ============================================================
@partial(jax.jit, static_argnames=['debiased', 'num_iters'])
def solve_nested_closure(cost_matrix: jnp.ndarray,
                         p: jnp.ndarray,
                         q: jnp.ndarray,
                         beta0: float,
                         kappa: float,
                         num_iters: int,
                         debiased: bool = False):
    """使用嵌套闭包的版本（类似OTT Geometry对象的情况）"""
    m, n = cost_matrix.shape
    log_p = jnp.log(p)
    log_q = jnp.log(q)

    # ⚠️ 模拟OTT Geometry对象
    class FakeGeometry:

        def __init__(self):
            self.cost_matrix = cost_matrix

    geom = FakeGeometry()

    def step_fn(state: ScanState, _):
        u_prev, v_prev, beta_prev, t = state

        # ⚠️ 通过对象属性访问（更深层的闭包）
        temp1 = v_prev[None, :] + log_q[None, :] - beta_prev * geom.cost_matrix

        if debiased and t > 1:
            t_float = jnp.array(t, dtype=jnp.float32)
            correction_coef = (t_float**kappa -
                               (t_float - 1)**kappa) / (t_float**kappa)
            temp1 = temp1 - correction_coef * u_prev[:, None]

        u_new = -logsumexp_stable(temp1, axis=1, keepdims=False)
        beta_new = beta0 * ((t + 1.0)**kappa)

        temp2 = u_new[:, None] + log_p[:, None] - beta_new * geom.cost_matrix
        v_new = -logsumexp_stable(temp2, axis=0, keepdims=False)

        log_plan = (u_new[:, None] + log_p[:, None] + v_new[None, :] +
                    log_q[None, :] - beta_new * geom.cost_matrix)
        plan = jnp.exp(log_plan)

        next_state = ScanState(u=u_new, v=v_new, beta=beta_new, t=t + 1)
        output = ScanOutput(u=u_new, v=v_new, beta=beta_new, plan=plan)

        return next_state, output

    initial_state = ScanState(u=jnp.zeros(m), v=jnp.zeros(n), beta=beta0, t=1)

    final_state, outputs = jax.lax.scan(step_fn,
                                        initial_state,
                                        xs=None,
                                        length=num_iters)

    return outputs


# ============================================================
# 测试主函数
# ============================================================
def main():
    print("=" * 70)
    print("闭包捕获测试")
    print("=" * 70)

    # 设置
    key = jax.random.PRNGKey(42)
    m, n = 50, 50  # 稍小一点，方便观察

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
    print(f"代价矩阵范围: [{float(c.min()):.4f}, {float(c.max()):.4f}]")

    # 参数
    osc = float(c.max() - c.min())
    beta0 = 10.0 / osc
    num_iters = 100
    kappa = 0.5

    print(f"\n算法参数: β₀={beta0:.4f}, κ={kappa}, 迭代={num_iters}")

    # ========== 运行三个版本 ==========
    print("\n" + "-" * 70)
    print("运行版本1: 显式传递")
    print("-" * 70)
    out_explicit = solve_explicit(c,
                                  p,
                                  q,
                                  beta0,
                                  kappa,
                                  num_iters,
                                  debiased=False)
    print("✓ 完成")

    print("\n" + "-" * 70)
    print("运行版本2: 闭包捕获")
    print("-" * 70)
    out_closure = solve_closure(c,
                                p,
                                q,
                                beta0,
                                kappa,
                                num_iters,
                                debiased=False)
    print("✓ 完成")

    print("\n" + "-" * 70)
    print("运行版本3: 嵌套闭包（模拟Geometry对象）")
    print("-" * 70)
    out_nested = solve_nested_closure(c,
                                      p,
                                      q,
                                      beta0,
                                      kappa,
                                      num_iters,
                                      debiased=False)
    print("✓ 完成")

    # ========== 对比结果 ==========
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)

    def compare(name1, out1, name2, out2):
        print(f"\n{name1} vs {name2}:")

        # Beta
        beta_diff = jnp.abs(out1.beta - out2.beta)
        print(f"  Beta差异: 最大={float(jnp.max(beta_diff)):.2e}, "
              f"平均={float(jnp.mean(beta_diff)):.2e}")

        # U
        u_diff = jnp.abs(out1.u - out2.u)
        print(f"  U差异:    最大={float(jnp.max(u_diff)):.2e}, "
              f"平均={float(jnp.mean(u_diff)):.2e}")

        # V
        v_diff = jnp.abs(out1.v - out2.v)
        print(f"  V差异:    最大={float(jnp.max(v_diff)):.2e}, "
              f"平均={float(jnp.mean(v_diff)):.2e}")

        # Plans
        plan_diff = jnp.abs(out1.plan - out2.plan)
        print(f"  Plan差异: 最大={float(jnp.max(plan_diff)):.2e}, "
              f"平均={float(jnp.mean(plan_diff)):.2e}")

        # 判断
        max_diff = float(jnp.max(plan_diff))
        if max_diff < 1e-6:
            print(f"  ✅ 结果一致！（差异 < 1e-6）")
            return True
        elif max_diff < 1e-3:
            print(f"  ⚠️  有微小差异（1e-6 < 差异 < 1e-3）")
            return False
        else:
            print(f"  ❌ 结果显著不同！（差异 > 1e-3）")
            return False

    # 对比
    ok1 = compare("显式传递", out_explicit, "闭包捕获", out_closure)
    ok2 = compare("显式传递", out_explicit, "嵌套闭包", out_nested)
    ok3 = compare("闭包捕获", out_closure, "嵌套闭包", out_nested)

    # ========== 详细检查（如果有差异）==========
    if not (ok1 and ok2 and ok3):
        print("\n" + "=" * 70)
        print("详细检查（前5步）")
        print("=" * 70)

        for t in range(min(5, num_iters)):
            print(f"\n迭代 t={t+1}:")
            print(f"  Beta - 显式: {out_explicit.beta[t]:.6f}, "
                  f"闭包: {out_closure.beta[t]:.6f}, "
                  f"嵌套: {out_nested.beta[t]:.6f}")
            print(f"  U[0] - 显式: {out_explicit.u[t,0]:.6f}, "
                  f"闭包: {out_closure.u[t,0]:.6f}, "
                  f"嵌套: {out_nested.u[t,0]:.6f}")
            print(
                f"  Plan差异 - 闭包: {float(jnp.max(jnp.abs(out_explicit.plan[t] - out_closure.plan[t]))):.2e}, "
                f"嵌套: {float(jnp.max(jnp.abs(out_explicit.plan[t] - out_nested.plan[t]))):.2e}"
            )

    # ========== 结论 ==========
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    if ok1 and ok2 and ok3:
        print("✅ 所有三个版本产生了相同的结果！")
        print("✅ 闭包捕获在JAX中是安全的（至少在这个场景下）")
        print("\n这意味着：最初实现的问题不是闭包捕获本身，")
        print("而是算法逻辑错误（beta传递、对偶变量定义等）")
    else:
        print("⚠️  不同版本产生了不同的结果！")
        print("这说明：闭包捕获可能确实是问题的一部分")
        print("\n需要进一步调查...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
