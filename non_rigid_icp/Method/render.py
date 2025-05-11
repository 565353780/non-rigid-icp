import torch
import platform
import numpy as np
import open3d as o3d
from typing import Union
from copy import deepcopy

from non_rigid_icp.Method.trans import toPointCloud, toMesh


def renderPoints(points: np.ndarray, window_name="Points", point_show_normal: bool = False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return renderGeometries(pcd, window_name, point_show_normal)

def renderGeometries(geometry_list, window_name="Geometry List", point_show_normal: bool = False):
    if isinstance(geometry_list, np.ndarray):
        return renderPoints(geometry_list, window_name, point_show_normal)

    if not isinstance(geometry_list, list):
        geometry_list = [geometry_list]

    o3d.visualization.draw_geometries(geometry_list, window_name, point_show_normal=point_show_normal)
    return True

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

def renderGeometryOnScreen(geometry, phi=45, theta=30, radius=3.0, width=800, height=600, window_name="OnScreenRender"):
    """
    使用 Open3D 的窗口渲染器渲染点云或三角网格，并返回窗口截图的图像数据 (numpy 数组格式)

    :param geometry: Open3D 点云 (o3d.geometry.PointCloud) 或 三角网格 (o3d.geometry.TriangleMesh)
    :param phi: 方位角 (0-360°)，决定相机在 xy 平面上的旋转角度
    :param theta: 极角 (0-180°)，决定相机在 z 轴上的仰角
    :param radius: 相机到物体的距离
    :param width: 渲染图像宽度
    :param height: 渲染图像高度
    :param window_name: 窗口名称
    :return: 渲染后的 RGB 图像 (numpy 数组格式)
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height, visible=True)
    vis.add_geometry(geometry)
    ctr = vis.get_view_control()
    # 计算相机位置
    eye = sphericalToCartesian(radius, phi, theta)
    center = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    ctr.set_lookat(center)
    ctr.set_front((center - eye) / np.linalg.norm(center - eye))
    ctr.set_up(up)
    ctr.set_zoom(0.7)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(False)
    vis.destroy_window()
    img_np = (np.asarray(img) * 255).astype(np.uint8)
    return img_np

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

def renderGeometryImage(geometry, phi=45, theta=30, radius=3.0, width=800, height=600):
    if isinstance(geometry, o3d.geometry.PointCloud) or isinstance(geometry, o3d.geometry.TriangleMesh):
        render_geometry = geometry
    else:
        render_geometry = geometry.toO3DMesh()

    system = platform.system().lower()
    if system == "linux":
        return renderGeometryOffScreen(render_geometry, phi, theta, radius, width, height)
    else:
        return renderGeometryOnScreen(render_geometry, phi, theta, radius, width, height)

def estimateRadius(geometry):
    """
    根据几何体的大小自动估计合适的相机距离
    
    :param geometry: Open3D 点云或三角网格
    :return: 估计的相机距离
    """
    if isinstance(geometry, o3d.geometry.PointCloud):
        points = np.asarray(geometry.points)
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        points = np.asarray(geometry.vertices)
    else:
        # 尝试使用toO3DMesh方法
        try:
            mesh = geometry.toO3DMesh()
            points = np.asarray(mesh.vertices)
        except:
            # 默认距离
            return 3.0
    
    # 计算点云的边界框
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    # 计算对角线长度
    diagonal = np.linalg.norm(max_bound - min_bound)
    
    # 根据对角线长度估计合适的相机距离
    # 通常距离是对角线长度的1.5-2倍
    return diagonal * 1.8

def calculateViewSpecificRadius(geometry, phi, theta, fov_degree=60.0, aspect_ratio=1.0):
    """
    根据几何体的轴对齐包围盒(ABB)和视角方向计算最佳相机距离
    
    :param geometry: Open3D 几何体（点云或三角网格）
    :param phi: 方位角 (0-360°)
    :param theta: 极角 (0-180°)
    :param fov_degree: 相机视场角（度）
    :param aspect_ratio: 图像宽高比
    :return: 最佳相机距离
    """
    # 获取轴对齐包围盒
    if isinstance(geometry, o3d.geometry.PointCloud):
        abb = geometry.get_axis_aligned_bounding_box()
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        abb = geometry.get_axis_aligned_bounding_box()
    else:
        # 尝试使用toO3DMesh方法
        try:
            mesh = geometry.toO3DMesh()
            abb = mesh.get_axis_aligned_bounding_box()
        except:
            # 默认使用estimateRadius
            return estimateRadius(geometry)
    
    # 获取包围盒的尺寸
    min_bound = np.asarray(abb.min_bound)
    max_bound = np.asarray(abb.max_bound)
    extent = max_bound - min_bound
    
    # 计算视角方向向量
    phi_rad = np.radians(phi)
    theta_rad = np.radians(theta)
    view_dir = np.array([
        np.sin(theta_rad) * np.cos(phi_rad),
        np.sin(theta_rad) * np.sin(phi_rad),
        np.cos(theta_rad)
    ])
    
    # 计算视角方向上的投影尺寸
    # 对于正交视角，我们需要考虑垂直于视线方向的平面上的尺寸
    if np.isclose(theta, 0) or np.isclose(theta, 180):  # 上视图或下视图
        # 对于上/下视图，我们关注XY平面的尺寸
        visible_size = max(extent[0], extent[1])
        # 下视图需要更大的安全距离
        if np.isclose(theta, 180):
            safety_factor = 2.0
    elif np.isclose(phi % 180, 0):  # 前视图或后视图
        # 对于前/后视图，我们关注YZ平面的尺寸
        visible_size = max(extent[1], extent[2])
    elif np.isclose(phi % 180, 90):  # 左视图或右视图
        # 对于左/右视图，我们关注XZ平面的尺寸
        visible_size = max(extent[0], extent[2])
        # 左视图需要更大的安全距离
        if np.isclose(phi, 90):
            safety_factor = 2.0
    else:
        # 对于其他视角，我们取最大尺寸
        visible_size = np.max(extent)
    
    # 计算所需的相机距离，使物体完全在视野内
    # 使用视场角计算
    fov_rad = np.radians(fov_degree)
    # 添加一个安全系数，确保物体完全在视野内
    # 如果在前面的计算中没有设置特定视角的安全系数，则使用默认值
    if 'safety_factor' not in locals():
        safety_factor = 1.2
    distance = (visible_size * 0.5 * safety_factor) / np.tan(fov_rad * 0.5)
    
    # 重置安全系数，避免影响下一次计算
    safety_factor = 1.2
    
    return distance

def renderGeometryImages(geometry, width=800, height=600):
    """
    渲染几何体的6个正交视角（前、后、左、右、上、下）并将它们组合成一个2x3的网格图像
    通过在同一个渲染上下文中改变相机位置来提高渲染速度
    根据物体在不同方向的尺寸自适应调整相机距离
    
    :param geometry: Open3D 点云、三角网格或自定义几何体（需要有toO3DMesh方法）
    :param width: 单个视角图像的宽度
    :param height: 单个视角图像的高度
    :return: 包含6个视角的组合图像（numpy数组格式）
    """
    if isinstance(geometry, o3d.geometry.PointCloud) or isinstance(geometry, o3d.geometry.TriangleMesh):
        render_geometry = geometry
    else:
        render_geometry = geometry.toO3DMesh()

    # 定义6个正交视角的相机参数（方位角和极角）
    # 前、后、左、右、上、下
    views = [
        (0, 90),    # 前视图 (phi=0, theta=90)
        (180, 90),  # 后视图 (phi=180, theta=90)
        (90, 90),   # 左视图 (phi=90, theta=90)
        (270, 90),  # 右视图 (phi=270, theta=90)
        (0, 0),     # 上视图 (phi=0, theta=0)
        (0, 180)    # 下视图 (phi=0, theta=180)
    ]
    
    # 创建2x3的网格图像
    grid_height = height * 2
    grid_width = width * 3
    grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    # 将6个视角图像放置在网格中的位置
    positions = [
        (0, 0),             # 前视图位置
        (0, width),          # 后视图位置
        (0, width * 2),      # 左视图位置
        (height, 0),         # 右视图位置
        (height, width),     # 上视图位置
        (height, width * 2)  # 下视图位置
    ]
    
    # 计算每个视角的最佳相机距离
    view_distances = []
    for phi, theta in views:
        # 根据物体在不同方向的尺寸计算最佳相机距离
        distance = calculateViewSpecificRadius(render_geometry, phi, theta)
        view_distances.append(distance)
    
    system = platform.system().lower()
    if system == "linux":
        # 使用离屏渲染器
        render = o3d.visualization.rendering.OffscreenRenderer(width, height)
        
        # 创建材质
        material = o3d.visualization.rendering.MaterialRecord()
        if isinstance(render_geometry, o3d.geometry.PointCloud):
            material.shader = "defaultUnlit"
        elif isinstance(render_geometry, o3d.geometry.TriangleMesh):
            material.shader = "defaultLit"
        
        # 设置背景颜色
        render.scene.set_background([1, 1, 1, 1])  # 白色背景
        
        # 添加几何体（只添加一次）
        render.scene.add_geometry("object", render_geometry, material)
        
        # 渲染每个视角
        images = []
        for (phi, theta), radius in zip(views, view_distances):
            # 计算相机位置（球面坐标 -> 笛卡尔坐标）
            eye = sphericalToCartesian(radius, phi, theta)
            center = np.array([0, 0, 0])  # 物体始终位于原点
            up = np.array([0, 1, 0])  # 设定 Y 轴向上
            
            # 设置摄像机视角
            render.scene.camera.look_at(center, eye, up)
            
            # 渲染到 numpy 数组
            img = render.render_to_image()
            img_np = np.asarray(img)  # 转换为 numpy 数组
            images.append(img_np)
    else:
        # 使用窗口渲染器
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)  # 设置为不可见窗口
        vis.add_geometry(render_geometry)  # 只添加一次几何体
        
        # 获取视图控制器
        ctr = vis.get_view_control()
        
        # 渲染每个视角
        images = []
        for (phi, theta), radius in zip(views, view_distances):
            # 计算相机位置
            eye = sphericalToCartesian(radius, phi, theta)
            center = np.array([0, 0, 0])
            up = np.array([0, 1, 0])
            
            # 设置相机参数
            ctr.set_lookat(center)
            ctr.set_front((center - eye) / np.linalg.norm(center - eye))
            ctr.set_up(up)
            ctr.set_zoom(0.7)
            
            # 更新渲染器
            vis.poll_events()
            vis.update_renderer()
            
            # 捕获屏幕
            img = vis.capture_screen_float_buffer(False)
            img_np = (np.asarray(img) * 255).astype(np.uint8)
            images.append(img_np)
        
        # 清理资源
        vis.destroy_window()
    
    # 将渲染的图像放入网格中
    for img, (row, col) in zip(images, positions):
        grid_image[row:row+height, col:col+width] = img

    return grid_image
