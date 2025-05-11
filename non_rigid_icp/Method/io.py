import os
import numpy as np
import open3d as o3d
from typing import Union, Dict, Tuple, List, Any


def loadMeshFile(mesh_file_path: str) -> Union[o3d.geometry.TriangleMesh, None]:
    if not os.path.exists(mesh_file_path):
        print("[ERROR][io::loadMeshFile]")
        print("\t mesh file not exist!")
        print("\t mesh_file_path:", mesh_file_path)
        return None

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def loadPLYAttributes(mesh_file_path: str) -> Union[Dict[str, np.ndarray], None]:
    """
    加载PLY三角网格文件，并提取所有额外的顶点和面属性

    Args:
        mesh_file_path: PLY文件路径

    Returns:
        Tuple包含:
            - o3d.geometry.TriangleMesh: 加载的网格
            - Dict[str, np.ndarray]: 包含所有额外属性的字典，键为属性名，值为numpy数组
    """
    if not os.path.exists(mesh_file_path):
        print("[ERROR][io::loadPLYMeshFile]")
        print("\t mesh file not exist!")
        print("\t mesh_file_path:", mesh_file_path)
        return None

    # 提取额外的属性
    attributes = {}

    # 读取PLY文件头部以获取属性信息
    vertex_properties = []
    face_properties = []
    current_element = None
    vertex_count = 0
    face_count = 0

    try:
        with open(mesh_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == 'end_header':
                    break

                if line.startswith('element vertex'):
                    current_element = 'vertex'
                    vertex_count = int(line.split()[-1])
                elif line.startswith('element face'):
                    current_element = 'face'
                    face_count = int(line.split()[-1])
                elif line.startswith('property') and current_element is not None:
                    parts = line.split()
                    # 跳过标准属性 (x,y,z 和 vertex_indices)
                    if current_element == 'vertex' and parts[-1] not in ['x', 'y', 'z']:
                        vertex_properties.append(parts[-1])
                    elif current_element == 'face' and 'vertex_indices' not in line:
                        face_properties.append(parts[-1])

        # 如果有额外的顶点属性，读取它们
        if vertex_properties:
            # 使用numpy直接从PLY文件读取数据
            data_started = False
            vertex_data = [[] for _ in range(len(vertex_properties))]
            vertex_counter = 0

            with open(mesh_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == 'end_header':
                        data_started = True
                        continue

                    if data_started and vertex_counter < vertex_count:
                        # 解析顶点行
                        values = line.split()
                        if len(values) >= 3 + len(vertex_properties):  # x,y,z + 额外属性
                            for i, prop in enumerate(vertex_properties):
                                vertex_data[i].append(float(values[3 + i]))
                        vertex_counter += 1
                    elif vertex_counter >= vertex_count:
                        break

            # 将数据转换为numpy数组并存储在属性字典中
            for i, prop in enumerate(vertex_properties):
                attributes[f'vertex_{prop}'] = np.array(vertex_data[i])

        # 如果有额外的面属性，读取它们
        if face_properties:
            face_data = [[] for _ in range(len(face_properties))]
            face_counter = 0
            data_started = False
            vertex_lines_to_skip = vertex_count

            with open(mesh_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == 'end_header':
                        data_started = True
                        continue

                    if data_started:
                        if vertex_lines_to_skip > 0:
                            vertex_lines_to_skip -= 1
                            continue

                        if face_counter < face_count:
                            values = line.split()
                            # 第一个值通常是顶点索引的数量
                            idx_count = int(values[0])
                            if len(values) >= idx_count + 1 + len(face_properties):
                                for i, prop in enumerate(face_properties):
                                    face_data[i].append(float(values[idx_count + 1 + i]))
                            face_counter += 1
                        else:
                            break

            # 将数据转换为numpy数组并存储在属性字典中
            for i, prop in enumerate(face_properties):
                attributes[f'face_{prop}'] = np.array(face_data[i])

    except Exception as e:
        print(f"[WARNING][io::loadPLYMeshFile] Error parsing PLY attributes: {e}")
        # 即使解析额外属性失败，仍然返回基本网格

    return attributes
