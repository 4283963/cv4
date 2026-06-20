import numpy as np


def calculate_temperature_distribution(
    inner_temp: float,
    insulation_thickness: float,
    env_temp: float,
    pipe_inner_radius: float = 0.1,
    pipe_wall_thickness: float = 0.01,
    pipe_conductivity: float = 50.0,
    insulation_conductivity: float = 0.04,
    convection_coeff: float = 10.0,
    num_nodes: int = 200
) -> dict:
    """
    计算多层圆筒壁（管壁 + 保温层）+ 外壁对流的一维稳态径向温度分布。

    采用**精确解析解**直接求解，确保能量守恒严格满足，温度不会出现
    超温或物理上不合理的数值。

    物理模型:
        内壁 r = r1:  T = T1                    (定温, Dirichlet)
        管壁区域:     T(r) = T1 - Q'·ln(r/r1)/(2πk1)
        保温层区域:   T(r) = T(r2) - Q'·ln(r/r2)/(2πk2)
        外壁 r = r3:  -k·dT/dr = h(T3 - Tenv)   (对流, Robin)

    总热阻（单位管长）:
        R'_total = ln(r2/r1)/(2πk1) + ln(r3/r2)/(2πk2) + 1/(2πh r3)

    参数:
        inner_temp: 管道内壁温度 T1 (°C)
        insulation_thickness: 保温层厚度 δ = r3 - r2 (m)
        env_temp: 环境温度 Tenv (°C)
        pipe_inner_radius: 管道内半径 r1 (m)
        pipe_wall_thickness: 管壁厚度 = r2 - r1 (m)
        pipe_conductivity: 管壁导热系数 k1 (W/m·K)
        insulation_conductivity: 保温层导热系数 k2 (W/m·K)
        convection_coeff: 外壁对流换热系数 h (W/m²·K)
        num_nodes: 径向节点数（决定输出数组长度）

    返回:
        dict: 温度分布、半径坐标、边界温度、各层半径等信息
    """
    r1 = pipe_inner_radius
    r2 = r1 + pipe_wall_thickness
    r3 = r2 + insulation_thickness

    if insulation_thickness <= 0:
        raise ValueError("保温层厚度必须大于 0")

    R_pipe = np.log(r2 / r1) / (2.0 * np.pi * pipe_conductivity)
    R_insul = np.log(r3 / r2) / (2.0 * np.pi * insulation_conductivity)
    R_conv = 1.0 / (2.0 * np.pi * convection_coeff * r3)
    R_total = R_pipe + R_insul + R_conv

    Q_per_L = (inner_temp - env_temp) / R_total

    r_nodes = np.linspace(r1, r3, num_nodes)
    T_nodes = np.empty(num_nodes)
    in_pipe = r_nodes <= r2
    in_insul = ~in_pipe

    T_nodes[in_pipe] = (
        inner_temp
        - Q_per_L * np.log(r_nodes[in_pipe] / r1) / (2.0 * np.pi * pipe_conductivity)
    )

    T_pipe_outer = inner_temp - Q_per_L * R_pipe
    T_nodes[in_insul] = (
        T_pipe_outer
        - Q_per_L * np.log(r_nodes[in_insul] / r2) / (2.0 * np.pi * insulation_conductivity)
    )

    T_outer_surface = T_nodes[-1]
    T_outer_check = env_temp + Q_per_L * R_conv

    assert abs(T_outer_surface - T_outer_check) < 1e-8, (
        f"外壁温度不一致: T(end)={T_outer_surface:.6f} vs "
        f"Tenv+Q'R_conv={T_outer_check:.6f}"
    )

    T_max = float(np.max(T_nodes))
    T_min = float(np.min(T_nodes))
    physical_max = max(inner_temp, env_temp) + 1e-8
    physical_min = min(inner_temp, env_temp) - 1e-8

    assert T_max <= physical_max, (
        f"违反能量守恒: 最高温度 {T_max:.4f}°C 超过 "
        f"max(T1,Tenv)={physical_max:.4f}°C"
    )
    assert T_min >= physical_min, (
        f"违反能量守恒: 最低温度 {T_min:.4f}°C 低于 "
        f"min(T1,Tenv)={physical_min:.4f}°C"
    )
    assert np.all(np.diff(T_nodes) <= 1e-10), (
        "温度分布必须从内壁向外壁单调非递增"
    )

    return {
        "temperatures": T_nodes.tolist(),
        "radii": r_nodes.tolist(),
        "pipe_inner_radius": r1,
        "pipe_outer_radius": r2,
        "insulation_outer_radius": r3,
        "min_temp": T_min,
        "max_temp": T_max,
        "outer_surface_temp": float(T_outer_surface),
        "heat_flux_per_length": float(Q_per_L),
        "pipe_wall_temp_drop": float(inner_temp - T_pipe_outer),
        "insulation_temp_drop": float(T_pipe_outer - T_outer_surface),
        "convection_temp_drop": float(T_outer_surface - env_temp),
    }


def _fdm_debug_solver(*args, **kwargs):
    """
    （保留用于调试）有限体积法求解器。
    注意：当 k1/k2 > 100 时界面附近会有较大截断误差。
    生产代码请使用解析解 calculate_temperature_distribution。
    """
    (inner_temp, insulation_thickness, env_temp) = args[:3]
    r1 = kwargs.get("pipe_inner_radius", 0.1)
    r2 = r1 + kwargs.get("pipe_wall_thickness", 0.01)
    r3 = r2 + insulation_thickness
    k1 = kwargs.get("pipe_conductivity", 50.0)
    k2 = kwargs.get("insulation_conductivity", 0.04)
    h = kwargs.get("convection_coeff", 10.0)
    N = kwargs.get("num_nodes", 200)

    r = np.linspace(r1, r3, N)
    dr = r[1] - r[0]
    k = np.where(r <= r2, k1, k2)

    A = np.zeros((N, N))
    b = np.zeros(N)
    A[0, 0] = 1.0
    b[0] = inner_temp

    def hm(a, b_):
        return 2.0 * a * b_ / (a + b_) if (a > 0 and b_ > 0) else 0.5 * (a + b_)

    for i in range(1, N - 1):
        kw = hm(k[i], k[i - 1])
        ke = hm(k[i], k[i + 1])
        rw = r[i] - dr / 2.0
        re = r[i] + dr / 2.0
        A[i, i - 1] = -kw * rw
        A[i, i] = kw * rw + ke * re
        A[i, i + 1] = -ke * re

    rN, rwN = r[-1], r[-1] - dr / 2.0
    kwN = hm(k[-1], k[-2])
    A[N - 1, N - 2] = -2.0 * kwN * rwN
    A[N - 1, N - 1] = 2.0 * kwN * rwN + 2.0 * h * rN * dr
    b[N - 1] = 2.0 * h * rN * dr * env_temp

    T = np.linalg.solve(A, b)
    return T.tolist()
