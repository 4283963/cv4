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
    num_nodes: int = 200,
    damage_angle: float = None,
    damage_width: float = 45.0,
    damage_factor: float = 5.0,
) -> dict:
    """
    计算管道横截面的二维稳态温度分布。

    当 damage_angle == 0（无局部破损）时，使用一维解析解（精确、快速）。
    当 damage_angle > 0 时，使用二维极坐标有限体积法求解。

    局部破损模型：在以 damage_angle 为中心、宽度为 damage_width 的扇区内，
    保温层的等效导热系数乘以 damage_factor（通常 >1，表示隔热效果变差）。

    参数:
        inner_temp: 管道内壁温度 (°C)
        insulation_thickness: 保温层厚度 (m)
        env_temp: 环境温度 (°C)
        pipe_inner_radius: 管道内半径 (m)
        pipe_wall_thickness: 管壁厚度 (m)
        pipe_conductivity: 管壁导热系数 (W/m·K)
        insulation_conductivity: 保温层导热系数 (W/m·K)
        convection_coeff: 外壁对流换热系数 (W/m²·K)
        num_nodes: 径向节点数（一维模式）
        damage_angle: 破损区域中心角 (度)，0 表示无破损，一维对称
        damage_width: 破损区域的角宽度 (度)
        damage_factor: 破损区域保温层导热系数放大倍数（>1 表示更差的隔热）

    返回:
        dict: 温度分布数据
    """
    if damage_angle is None or damage_width is None or damage_width <= 0:
        return _one_dimensional_solution(
            inner_temp, insulation_thickness, env_temp,
            pipe_inner_radius, pipe_wall_thickness,
            pipe_conductivity, insulation_conductivity,
            convection_coeff, num_nodes
        )

    return _two_dimensional_solution(
        inner_temp, insulation_thickness, env_temp,
        pipe_inner_radius, pipe_wall_thickness,
        pipe_conductivity, insulation_conductivity,
        convection_coeff,
        damage_angle, damage_width, damage_factor,
        nr=50, nth=120,
    )


def _one_dimensional_solution(
    T1, thickness, Tenv, r1, pipe_thick,
    k_pipe, k_insul, h, nr
):
    """
    一维稳态多层圆筒壁导热 + 外壁对流 的精确解析解。
    """
    r2 = r1 + pipe_thick
    r3 = r2 + thickness

    if thickness <= 0:
        raise ValueError("保温层厚度必须大于 0")

    R1 = np.log(r2 / r1) / (2.0 * np.pi * k_pipe)
    R2 = np.log(r3 / r2) / (2.0 * np.pi * k_insul)
    R3 = 1.0 / (2.0 * np.pi * h * r3)
    R_total = R1 + R2 + R3

    Q_per_L = (T1 - Tenv) / R_total

    r_nodes = np.linspace(r1, r3, nr)
    T_nodes = np.empty(nr)
    in_pipe = r_nodes <= r2

    T_nodes[in_pipe] = T1 - Q_per_L * np.log(r_nodes[in_pipe] / r1) / (2.0 * np.pi * k_pipe)

    T_pipe_outer = T1 - Q_per_L * R1
    T_nodes[~in_pipe] = T_pipe_outer - Q_per_L * np.log(r_nodes[~in_pipe] / r2) / (2.0 * np.pi * k_insul)

    T_outer_surface = float(T_nodes[-1])

    return {
        "mode": "1D",
        "temperatures": T_nodes.tolist(),
        "radii": r_nodes.tolist(),
        "pipe_inner_radius": r1,
        "pipe_outer_radius": r2,
        "insulation_outer_radius": r3,
        "min_temp": float(np.min(T_nodes)),
        "max_temp": float(np.max(T_nodes)),
        "outer_surface_temp": T_outer_surface,
        "heat_flux_per_length": float(Q_per_L),
        "pipe_wall_temp_drop": float(T1 - T_pipe_outer),
        "insulation_temp_drop": float(T_pipe_outer - T_outer_surface),
        "convection_temp_drop": float(T_outer_surface - Tenv),
    }


def _two_dimensional_solution(
    T1, thickness, Tenv, r1, pipe_thick,
    k_pipe, k_insul, h,
    damage_angle_deg, damage_width_deg, damage_factor,
    nr, nth,
):
    """
    二维极坐标有限体积法求解器。
    采用控制容积法，界面导热系数用调和平均，周向周期性边界。

    未知量排布：idx = i * nth + j，i 为径向索引，j 为周向索引。
    """
    r2 = r1 + pipe_thick
    r3 = r2 + thickness

    r = np.linspace(r1, r3, nr)
    dr = r[1] - r[0]

    theta = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
    dtheta = theta[1] - theta[0]

    damage_center = np.deg2rad(damage_angle_deg)
    damage_half = np.deg2rad(damage_width_deg) / 2.0

    k = np.zeros((nr, nth))
    for j in range(nth):
        th = theta[j]
        dth = _angular_distance(th, damage_center)
        in_damage = dth <= damage_half
        k_damage = k_insul * damage_factor if in_damage else k_insul
        for i in range(nr):
            if r[i] <= r2:
                k[i, j] = k_pipe
            else:
                k[i, j] = k_damage

    N = nr * nth
    A = np.zeros((N, N))
    b = np.zeros(N)

    def idx(i, j):
        return i * nth + (j % nth)

    for i in range(nr):
        for j in range(nth):
            p = idx(i, j)

            if i == 0:
                A[p, p] = 1.0
                b[p] = T1
                continue

            r_i = r[i]
            r_w = r_i - dr / 2.0
            r_e = r_i + dr / 2.0

            k_w = _harmonic_mean(k[i, j], k[i - 1, j])
            a_W = k_w * r_w * dtheta / dr

            if i < nr - 1:
                k_e = _harmonic_mean(k[i, j], k[i + 1, j])
                a_E = k_e * r_e * dtheta / dr
            else:
                a_E = 0.0

            j_s = (j - 1) % nth
            j_n = (j + 1) % nth
            k_s = _harmonic_mean(k[i, j], k[i, j_s])
            k_n = _harmonic_mean(k[i, j], k[i, j_n])

            a_S = k_s * dr / (r_i * dtheta)
            a_N = k_n * dr / (r_i * dtheta)

            a_P = a_W + a_E + a_S + a_N
            b_val = 0.0

            if i == nr - 1:
                a_conv = h * r_e * dtheta
                a_P += a_conv
                b_val += a_conv * Tenv

            A[p, idx(i - 1, j)] -= a_W
            if i < nr - 1:
                A[p, idx(i + 1, j)] -= a_E
            A[p, idx(i, j_s)] -= a_S
            A[p, idx(i, j_n)] -= a_N
            A[p, p] = a_P
            b[p] = b_val

    T_flat = np.linalg.solve(A, b)
    T = T_flat.reshape((nr, nth))

    T_max = float(np.max(T))
    T_min = float(np.min(T))
    physical_max = max(T1, Tenv) + 1e-4
    physical_min = min(T1, Tenv) - 1e-4

    assert T_max <= physical_max, f"违反能量守恒: T_max={T_max:.4f} > {physical_max:.4f}"
    assert T_min >= physical_min, f"违反能量守恒: T_min={T_min:.4f} < {physical_min:.4f}"

    T_outer_min = float(np.min(T[-1, :]))
    T_outer_max = float(np.max(T[-1, :]))

    theta_deg = np.rad2deg(theta).tolist()

    return {
        "mode": "2D",
        "temperatures_2d": T.tolist(),
        "radii": r.tolist(),
        "thetas": theta_deg,
        "pipe_inner_radius": r1,
        "pipe_outer_radius": r2,
        "insulation_outer_radius": r3,
        "min_temp": T_min,
        "max_temp": T_max,
        "outer_surface_temp_avg": float(np.mean(T[-1, :])),
        "outer_surface_temp_min": T_outer_min,
        "outer_surface_temp_max": T_outer_max,
        "damage_angle": damage_angle_deg,
        "damage_width": damage_width_deg,
        "damage_factor": damage_factor,
    }


def _angular_distance(a, b):
    """
    计算两个角度（弧度）之间的最小绝对差值，考虑 2π 周期性。
    """
    d = abs(a - b)
    d = d % (2.0 * np.pi)
    if d > np.pi:
        d = 2.0 * np.pi - d
    return d


def _harmonic_mean(k1, k2):
    if k1 <= 0.0 or k2 <= 0.0:
        return 0.5 * (k1 + k2)
    return 2.0 * k1 * k2 / (k1 + k2)
