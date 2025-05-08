import torch
import numpy as np
import open3d as o3d
from typing import Union
from copy import deepcopy

from non_rigid_icp.Method.trans import toPointCloud, toMesh


def renderGeometries(geometry_list, window_name="Geometry List", point_show_normal: bool = False):
    if not isinstance(geometry_list, list):
        geometry_list = [geometry_list]

    o3d.visualization.draw_geometries(geometry_list, window_name, point_show_normal=point_show_normal)
    return True

def renderPoints(points: np.ndarray, window_name="Points", point_show_normal: bool = False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return renderGeometries(pcd, window_name, point_show_normal)

def toPaintedMesh(
    mesh: o3d.geometry.TriangleMesh,
    color: Union[np.ndarray, list] = [0.1, 0.1, 0.9],
) -> o3d.geometry.TriangleMesh:
    painted_mesh = deepcopy(mesh)
    painted_mesh.paint_uniform_color(color)
    return painted_mesh

def renderColoredMeshes(
    mesh_list: list,
    color_list: list,
) -> bool:
    painted_mesh_list = []

    for mesh, color in zip(mesh_list, color_list):
        painted_mesh = toPaintedMesh(mesh, color)

        painted_mesh_list.append(painted_mesh)

    o3d.visualization.draw_geometries(painted_mesh_list)
    return True

def sphericalToCartesian(radius, phi, theta):
    """
    将球面坐标 (phi, theta) 转换为笛卡尔坐标 (x, y, z)
    :param radius: 相机到物体的距离
    :param phi: 方位角（水平旋转），0° 在 x 轴正方向，增加时绕 z 轴逆时针旋转
    :param theta: 极角（垂直旋转），0° 在 z 轴正方向，90° 在 xy 平面上
    :return: (x, y, z) 坐标
    """
    phi = np.radians(phi)
    theta = np.radians(theta)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.array([x, y, z])

def renderGeometryOffScreen(geometry, phi=45, theta=30, radius=3.0, width=800, height=600):
    """
    使用 Open3D 离屏渲染器渲染点云或三角网格，并返回图像数据 (numpy 数组格式)

    :param geometry: Open3D 点云 (o3d.geometry.PointCloud) 或 三角网格 (o3d.geometry.TriangleMesh)
    :param phi: 方位角 (0-360°)，决定相机在 xy 平面上的旋转角度
    :param theta: 极角 (0-180°)，决定相机在 z 轴上的仰角
    :param radius: 相机到物体的距离
    :param width: 渲染图像宽度
    :param height: 渲染图像高度
    :return: 渲染后的 RGB 图像 (numpy 数组格式)
    """
    # 创建离屏渲染器
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # 创建材质
    material = o3d.visualization.rendering.MaterialRecord()
    if isinstance(geometry, o3d.geometry.PointCloud):
        material.shader = "defaultUnlit"
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        material.shader = "defaultLit"

    # 设置背景颜色
    render.scene.set_background([1, 1, 1, 1])  # 白色背景

    # 添加几何体
    render.scene.add_geometry("object", geometry, material)

    # 计算相机位置（球面坐标 -> 笛卡尔坐标）
    eye = sphericalToCartesian(radius, phi, theta)
    center = np.array([0, 0, 0])  # 物体始终位于原点
    up = np.array([0, 1, 0])  # 设定 Z 轴向上

    # 设置摄像机视角
    render.scene.camera.look_at(center, eye, up)

    # 渲染到 numpy 数组
    img = render.render_to_image()
    img_np = np.asarray(img)  # 转换为 numpy 数组
    return img_np

def renderPointsOffScreen(
    points: Union[torch.Tensor, np.ndarray],
    triangles: Union[np.ndarray, None]=None,
    phi=45,
    theta=30,
    radius=3.0,
    width=800,
    height=600,
):
    if triangles is None:
        geometry = toPointCloud(points)
    else:
        geometry = toMesh(points, triangles)
        geometry.compute_vertex_normals()
    return renderGeometryOffScreen(geometry, phi, theta, radius, width, height)
