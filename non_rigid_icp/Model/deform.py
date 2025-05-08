import torch
import numpy as np
from torch import nn
from typing import Union

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Data.deform_field import DeformField
from non_rigid_icp.Method.trans import toTensor


class DeformModel(object):
    def __init__(
        self,
        mesh: Mesh,
        device: str = 'cpu',
        fixed_vertex_idxs: Union[np.ndarray, list, None] = None,
        fixed_target_positions: Union[np.ndarray, list, None] = None,
        rigid_part_idxs: Union[np.ndarray, list, None] = None,
    ):
        # source mesh data
        self.vertices = toTensor(mesh.vertices, device, torch.float32).unsqueeze(-1)

        # non rigid deform
        self.deform_field = DeformField(mesh, device)

        # fixed points
        self.fixed_vertex_idxs = None
        self.fixed_target_positions = None
        if fixed_vertex_idxs is not None and fixed_target_positions is not None:
            self.fixed_vertex_idxs = toTensor(fixed_vertex_idxs, device, torch.int64)
            self.fixed_target_positions = toTensor(fixed_target_positions, device, torch.float32)

        # rigid part idxs
        self.rigid_part_idxs = None
        if rigid_part_idxs is not None:
            self.rigid_part_idxs = toTensor(rigid_part_idxs, device, torch.int64)
        return

    def setDeformGradState(self, need_grad: bool, vertex_mask: Union[torch.Tensor, None] = None) -> bool:
        return self.deform_field.setGradState(need_grad, vertex_mask)

    def stiffness(self) -> torch.Tensor:
        return self.deform_field.stiffness()

    def deform(self):
        out_x = torch.matmul(self.deform_field.rotate_matrixs, self.vertices)
        out_x = out_x + self.deform_field.translates
        return out_x.squeeze(-1)
