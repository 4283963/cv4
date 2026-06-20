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
    使用有限差分法计算一维径向稳态温度分布

    参数:
        inner_temp: 管道内壁温度 (°C)
        insulation_thickness: 保温层厚度 (m)
        env_temp: 环境温度 (°C)
        pipe_inner_radius: 管道内半径 (m), 默认 0.1m
        pipe_wall_thickness: 管壁厚度 (m), 默认 0.01m
        pipe_conductivity: 管壁导热系数 (W/m·K), 默认 50 W/m·K (钢材)
        insulation_conductivity: 保温层导热系数 (W/m·K), 默认 0.04 W/m·K
        convection_coeff: 对流换热系数 (W/m²·K), 默认 10 W/m²·K
        num_nodes: 径向节点总数

    返回:
        包含温度分布数组、半径坐标数组和各层位置信息的字典
    """
    pipe_outer_radius = pipe_inner_radius + pipe_wall_thickness
    insulation_outer_radius = pipe_outer_radius + insulation_thickness

    r = np.linspace(pipe_inner_radius, insulation_outer_radius, num_nodes)
    dr = r[1] - r[0]

    k = np.ones(num_nodes)
    pipe_wall_mask = r <= pipe_outer_radius
    insulation_mask = r > pipe_outer_radius
    k[pipe_wall_mask] = pipe_conductivity
    k[insulation_mask] = insulation_conductivity

    T = np.ones(num_nodes) * env_temp
    T[0] = inner_temp

    A = np.zeros((num_nodes, num_nodes))
    b = np.zeros(num_nodes)

    A[0, 0] = 1.0
    b[0] = inner_temp

    for i in range(1, num_nodes - 1):
        ri = r[i]
        k_plus = 0.5 * (k[i] + k[i + 1])
        k_minus = 0.5 * (k[i] + k[i - 1])

        a_w = k_minus * (ri - dr / 2) / dr
        a_e = k_plus * (ri + dr / 2) / dr
        a_p = a_w + a_e

        A[i, i - 1] = -a_w
        A[i, i] = a_p
        A[i, i + 1] = -a_e
        b[i] = 0.0

    r_outer = r[-1]
    k_outer = k[-1]
    Bi = convection_coeff * dr / k_outer
    A[-1, -2] = -2.0
    A[-1, -1] = 2.0 + 2 * Bi * r_outer / (r_outer - dr / 2)
    b[-1] = 2 * Bi * r_outer / (r_outer - dr / 2) * env_temp

    T = np.linalg.solve(A, b)

    return {
        "temperatures": T.tolist(),
        "radii": r.tolist(),
        "pipe_inner_radius": pipe_inner_radius,
        "pipe_outer_radius": pipe_outer_radius,
        "insulation_outer_radius": insulation_outer_radius,
        "min_temp": float(min(T)),
        "max_temp": float(max(T)),
        "outer_surface_temp": float(T[-1])
    }
