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
        vertex_group_idxs: Union[np.ndarray, list, None] = None,
    ):
        # source mesh data
        self.vertices = toTensor(mesh.vertices, device, torch.float32).unsqueeze(-1)

        # non rigid deform
        self.deform_field = DeformField(mesh, device, vertex_group_idxs)

        # fixed points
        self.fixed_vertex_idxs = None
        self.fixed_target_positions = None
        if fixed_vertex_idxs is not None and fixed_target_positions is not None:
            self.fixed_vertex_idxs = toTensor(fixed_vertex_idxs, device, torch.int64)
            self.fixed_target_positions = toTensor(fixed_target_positions, device, torch.float32)
        return

    def setDeformGradState(self, need_grad: bool, vertex_mask: Union[torch.Tensor, None] = None) -> bool:
        return self.deform_field.setGradState(need_grad, vertex_mask)

    def deform(self):
        out_x = torch.matmul(self.deform_field.toRotateMatrixs(), self.vertices)
        out_x = out_x + self.deform_field.toTranslates()
        return out_x.squeeze(-1)

    def stiffness(self) -> torch.Tensor:
        return self.deform_field.stiffness()

    @torch.no_grad()
    def toDeformedFixedVertices(self) -> Union[torch.Tensor, None]:
        if self.fixed_vertex_idxs is None:
            print('[ERROR][DeformModel::toDeformedFixedVertices]')
            print('\t fixed vertex not selected!')
            return None

        fixed_vertices = self.vertices[self.fixed_vertex_idxs]
        fixed_rotate_matrixs = self.deform_field.toRotateMatrixs()[self.fixed_vertex_idxs].detach()
        fixed_translates = self.deform_field.toTranslates()[self.fixed_vertex_idxs].detach()

        deformed_fixed_vertices = torch.matmul(fixed_rotate_matrixs, fixed_vertices)
        deformed_fixed_vertices = deformed_fixed_vertices + fixed_translates
        return deformed_fixed_vertices.squeeze(-1)

    #FIXME: need to support the case of multiple land marks on single fixed surface
    def moveToLandMark(self) -> bool:
        if (self.fixed_vertex_idxs is None) or (self.fixed_target_positions is None):
            return True

        deformed_fixed_vertices = self.toDeformedFixedVertices()
        if deformed_fixed_vertices is None:
            print('[ERROR][DeformModel::moveToLandMark]')
            print('\t toDeformedFixedVertices failed!')
            return False

        delta_move_vectors = self.fixed_target_positions - deformed_fixed_vertices

        valid_vertex_idxs = self.deform_field.compact_vertex_group_idxs[self.fixed_vertex_idxs]

        self.deform_field.translates[valid_vertex_idxs] = self.deform_field.translates[valid_vertex_idxs] + delta_move_vectors
        return True
