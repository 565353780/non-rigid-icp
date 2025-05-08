import torch
import numpy as np
from torch import nn
from typing import Union

from non_rigid_icp.Data.deform_field import DeformField
from non_rigid_icp.Method.trans import toNumpy, toTensor


class DeformModel(object):
    def __init__(
        self,
        vertex_num: int,
        triangles: torch.Tensor,
        device: str = 'cpu',
    ):
        self.deform_field = DeformField(vertex_num, triangles, device)
        return

    def setDeformGradState(self, need_grad: bool, vertex_mask: Union[torch.Tensor, None] = None) -> bool:
        return self.deform_field.setGradState(need_grad, vertex_mask)

    def deform(self, vertices: torch.Tensor):
        vertices = vertices.unsqueeze(-1)
        out_x = torch.matmul(self.deform_field.rotate_matrixs, vertices)
        out_x = out_x + self.deform_field.translates
        return out_x.squeeze(-1)
