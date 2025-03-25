import os
import cv2
import torch
import trimesh
import numpy as np
import open3d as o3d
from typing import Union
from torch.utils.tensorboard import SummaryWriter

from non_rigid_icp.Method.trans import toNumpy


class Logger(object):
    def __init__(self, log_folder_path: Union[str, None] = None, is_mute: bool = False) -> None:
        self.summary_writer = None

        self.log_name_dict = {}

        if log_folder_path is not None:
            self.setLogFolder(log_folder_path)

        self.is_mute = is_mute
        self.error_outputed = is_mute
        return

    def reset(self) -> bool:
        self.summary_writer = None

        self.log_name_dict = {}
        return True

    def isValid(self) -> bool:
        return self.summary_writer is not None

    def setLogFolder(self, log_folder_path: str) -> bool:
        self.summary_writer = SummaryWriter(log_folder_path)
        return True

    def setStep(self, name: str, step: int) -> bool:
        self.log_name_dict[name] = step
        return True

    def getNameStep(self, name: str) -> int:
        if name not in self.log_name_dict.keys():
            self.setStep(name, 0)
            return 0

        return self.log_name_dict[name]

    def addStep(self, name: str) -> bool:
        return self.setStep(name, self.getNameStep(name) + 1)

    def addScalar(self, name: str, value: float, step: Union[int, None] = None) -> bool:
        if not self.isValid():
            if self.is_mute or self.error_outputed:
                return False

            print("[ERROR][Logger::addScalar]")
            print("\t isValid failed!")
            self.error_outputed = True
            return False

        if step is not None:
            self.setStep(name, step)

        name_step = self.getNameStep(name)

        self.summary_writer.add_scalar(name, value, name_step)

        self.addStep(name)
        return True

    def addImage(self, name: str, image: np.ndarray, step: Union[int, None] = None) -> bool:
        if not self.isValid():
            if self.is_mute or self.error_outputed:
                return False

            print("[ERROR][Logger::addImage]")
            print("\t isValid failed!")
            self.error_outputed = True
            return False

        if step is not None:
            self.setStep(name, step)

        name_step = self.getNameStep(name)

        image = image.transpose(2, 0, 1)

        self.summary_writer.add_image(name, image, name_step)
        return True

    def addVideo(
        self,
        name: str,
        video: Union[torch.Tensor, np.ndarray],
        fps: int = 10,
        step: Union[int, None] = None,
    ) -> bool:
        if not self.isValid():
            if self.is_mute or self.error_outputed:
                return False

            print("[ERROR][Logger::addVideo]")
            print("\t isValid failed!")
            self.error_outputed = True
            return False

        if step is not None:
            self.setStep(name, step)

        name_step = self.getNameStep(name)

        if isinstance(video, torch.Tensor):
            video_tensor = video
        else:
            video_tensor = torch.tensor(toNumpy(video), dtype=torch.uint8)

        if len(video_tensor.shape) == 4:
            video_tensor = video_tensor.unsqueeze(0)

        self.summary_writer.add_video(name, video_tensor, name_step, fps)
        return True

    def addVideoFile(
        self,
        name: str,
        video_file_path: str,
        fps: int = 10,
        step: Union[int, None] = None,
    ) -> bool:
        if not os.path.exists(video_file_path):
            if self.is_mute:
                return False

            print("[ERROR][Logger::addVideoFile]")
            print("\t video file not exist!")
            print('\t video_file_path:', video_file_path)
            return False

        cap = cv2.VideoCapture(video_file_path)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.tensor(frame, dtype=torch.uint8).permute(2, 0, 1)
            frames.append(frame)

        cap.release()

        video_tensor = torch.stack(frames).unsqueeze(0)

        if not self.addVideo(name, video_tensor, fps, step):
            print("[ERROR][Logger::addVideoFile]")
            print("\t addVideo failed!")
            return False
        return True

    def addPointCloud(self, name: str, pcd: Union[o3d.geometry.PointCloud, np.ndarray, torch.Tensor], step: Union[int, None]=None) -> bool:
        if not self.isValid():
            if self.is_mute or self.error_outputed:
                return False

            print("[ERROR][Logger::addPointCloud]")
            print("\t isValid failed!")
            self.error_outputed = True
            return False

        if isinstance(pcd, o3d.geometry.PointCloud):
            vertices = torch.tensor(np.asarray(pcd.points)).unsqueeze(0)
        elif isinstance(pcd, np.ndarray):
            vertices = torch.tensor(pcd).reshape(1, -1, 3)
        elif isinstance(pcd, torch.Tensor):
            vertices = pcd.reshape(1, -1, 3)
        else:
            print('[ERROR][Logger::addPointCloud]')
            print('\t pcd type not valid!')
            return False

        if step is not None:
            self.setStep(name, step)

        name_step = self.getNameStep(name)

        self.summary_writer.add_mesh(name, vertices, global_step=name_step)
        return True

    def addMesh(self, name: str, mesh: Union[o3d.geometry.TriangleMesh, trimesh.Trimesh], step: Union[int, None]=None) -> bool:
        if not self.isValid():
            if self.is_mute or self.error_outputed:
                return False

            print("[ERROR][Logger::addMesh]")
            print("\t isValid failed!")
            self.error_outputed = True
            return False

        if isinstance(mesh, o3d.geometry.TriangleMesh):
            vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32).unsqueeze(0)
            faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.int32).unsqueeze(0)
        elif isinstance(mesh, trimesh.Trimesh):
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0)
            faces = torch.tensor(mesh.faces, dtype=torch.int32).unsqueeze(0)
        else:
            print('[ERROR][Logger::addMesh]')
            print('\t mesh type not valid!')
            return False

        if step is not None:
            self.setStep(name, step)

        name_step = self.getNameStep(name)

        self.summary_writer.add_mesh(name, vertices, faces=faces, global_step=name_step)
        return True

    def addPointCloudFile(self, name: str, pcd_file_path: str, step: Union[int, None]=None) -> bool:
        if not os.path.exists(pcd_file_path):
            if self.is_mute:
                return False

            print("[ERROR][Logger::addPointCloudFile]")
            print("\t pcd file not exist!")
            print('\t pcd_file_path:', pcd_file_path)
            return False

        pcd = o3d.io.read_point_cloud(pcd_file_path)

        if not self.addPointCloud(name, pcd, step):
            print("[ERROR][Logger::addPointCloudFile]")
            print("\t addPointCloud failed!")
            return False

        return True

    def addMeshFile(self,
                    name: str,
                    mesh_file_path: str,
                    step: Union[int, None]=None,
                    backend: str = 'open3d',
                    ) -> bool:
        if not os.path.exists(mesh_file_path):
            if self.is_mute:
                return False

            print("[ERROR][Logger::addMeshFile]")
            print("\t mesh file not exist!")
            print('\t mesh_file_path:', mesh_file_path)
            return False

        if backend == 'open3d':
            mesh = o3d.io.read_triangle_mesh(mesh_file_path)
        elif backend == 'trimesh':
            mesh = trimesh.load(mesh_file_path)
        else:
            print('[ERROR][Logger::addMeshFile]')
            print('\t backend not valid!')
            print('\t backend:', backend)
            print('\t valid backends: open3d trimesh')
            return False

        if not self.addMesh(name, mesh, step):
            print("[ERROR][Logger::addMeshFile]")
            print("\t addMesh failed!")
            return False

        return True
