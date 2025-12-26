import torch
import numpy as np
from typing import Union, Tuple

from non_rigid_icp.Method.trans import toTensor


class TargetVerticesConstraint(object):
    def __init__(self) -> None:
        self.vertex_idxs_list = []
        self.target_vertices_list = []

        self.vertex_idxs_tensor = None
        self.target_vertices_tensor = None
        return

    def isValid(self) -> bool:
        if len(self.vertex_idxs_list) == 0:
            return False

        for vertex_idxs in self.vertex_idxs_list:
            if vertex_idxs.shape[0] > 0:
                return True

        return False

    def addConstraint(
        self, vertex_idxs: np.ndarray, target_vertices: np.ndarray
    ) -> bool:
        if vertex_idxs.shape[0] == 0:
            print("[ERROR][TargetVerticesConstraint::addConstraint]")
            print("\t vertex_idxs is empty!")
            return False

        if target_vertices.shape[0] == 0:
            print("[ERROR][TargetVerticesConstraint::addConstraint]")
            print("\t target_vertices is empty!")
            return False

        if vertex_idxs.shape[0] != target_vertices.shape[0]:
            print("[ERROR][TargetVerticesConstraint::addConstraint]")
            print("\t vertex_idxs and target_vertices size mismatch!")
            print("\t vertex_idxs.shape:", vertex_idxs.shape)
            print("\t target_vertices.shape:", target_vertices.shape)
            return False

        if target_vertices.shape[1] != 3:
            print("[ERROR][TargetVerticesConstraint::addConstraint]")
            print("\t target_vertices must have shape (N, 3)!")
            print("\t target_vertices.shape:", target_vertices.shape)
            return False

        self.vertex_idxs_list.append(vertex_idxs)
        self.target_vertices_list.append(target_vertices)
        return True

    def getConstraint(self) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        if len(self.vertex_idxs_list) == 0:
            return None

        vertex_idxs = np.concatenate(self.vertex_idxs_list)
        target_vertices = np.vstack(self.target_vertices_list)

        return vertex_idxs, target_vertices

    def updateTensor(
        self, device: str = "cpu", dtype=torch.float32
    ) -> bool:
        constraint = self.getConstraint()
        if constraint is None:
            print("[ERROR][TargetVerticesConstraint::updateTensor]")
            print("\t getConstraint returns None!")
            return False

        vertex_idxs, target_vertices = constraint

        self.vertex_idxs_tensor = toTensor(vertex_idxs, device, torch.int64)
        self.target_vertices_tensor = toTensor(target_vertices, device, dtype)

        return True

    def computeL1Loss(
        self, deformed_vertices: torch.Tensor
    ) -> torch.Tensor:
        """
        计算逐点的 L1 loss

        Args:
            deformed_vertices: 变形后的顶点，shape 为 (N_vertices, 3)

        Returns:
            L1 loss，标量
        """
        if self.vertex_idxs_tensor is None or self.target_vertices_tensor is None:
            print("[ERROR][TargetVerticesConstraint::computeL1Loss]")
            print("\t tensors not initialized! Please call updateTensor first!")
            return torch.tensor(0.0).to(deformed_vertices.device)

        # 获取变形后的目标顶点位置
        selected_vertices = deformed_vertices[self.vertex_idxs_tensor]

        # 计算 L1 loss
        l1_loss = torch.mean(torch.abs(selected_vertices - self.target_vertices_tensor))

        return l1_loss
