def calculate_partial_derivatives():
    """
    计算偏导数的示例
    """
    import sympy as sp

    print("=" * 60)
    print("偏导数计算示例")
    print("=" * 60)

    # 定义符号变量
    x, y, z = sp.symbols('x y z')

    # 示例1: f(x, y) = x²y + sin(x) + e^y
    print("\n【示例1: f(x, y) = x²y + sin(x) + eʸ】")
    f1 = x ** 2 * y + sp.sin(x) + sp.exp(y)

    df1_dx = sp.diff(f1, x)
    df1_dy = sp.diff(f1, y)

    print(f"f(x, y) = {f1}")
    print(f"∂f/∂x = {df1_dx}")
    print(f"∂f/∂y = {df1_dy}")

    # 在点 (π/2, 0) 处求值
    x0, y0 = sp.pi / 2, 0
    print(f"\n在点 (π/2, 0) 处:")
    print(f"  ∂f/∂x = {df1_dx.subs({x: x0, y: y0}).evalf():.4f}")
    print(f"  ∂f/∂y = {df1_dy.subs({x: x0, y: y0}).evalf():.4f}")

    # 示例2: f(x, y, z) = x²yz + xy²z + xyz²
    print("\n【示例2: f(x, y, z) = x²yz + xy²z + xyz²】")
    f2 = x ** 2 * y * z + x * y ** 2 * z + x * y * z ** 2

    df2_dx = sp.diff(f2, x)
    df2_dy = sp.diff(f2, y)
    df2_dz = sp.diff(f2, z)

    print(f"f(x, y, z) = {f2}")
    print(f"∂f/∂x = {df2_dx}")
    print(f"∂f/∂y = {df2_dy}")
    print(f"∂f/∂z = {df2_dz}")

    # 示例3: 复合函数
    print("\n【示例3: f(x, y) = sin(xy) + ln(x² + y²)】")
    f3 = sp.sin(x * y) + sp.log(x ** 2 + y ** 2)

    df3_dx = sp.diff(f3, x)
    df3_dy = sp.diff(f3, y)

    print(f"∂f/∂x = {df3_dx}")
    print(f"∂f/∂y = {df3_dy}")


if __name__ == "__main__":
    calculate_partial_derivatives()