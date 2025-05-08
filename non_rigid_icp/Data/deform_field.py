import torch
import numpy as np
from torch import nn

from non_rigid_icp.Method.trans import toNumpy, toTensor


class DeformField(object):
    def __init__(
        self,
        vertex_num: int,
        triangles: torch.Tensor,
    ) -> None:
        triangles = toNumpy(triangles)

        edges = np.vstack(
            [triangles[:, :2], triangles[:, 1:3], triangles[:, [0, 2]]])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        edges = toTensor(edges, dtype=torch.int64)
        self.register_buffer("edges", edges)

        self.rotate_matrixs = torch.eye(3).reshape(1, 3, 3).repeat(vertex_num, 1, 1)  # N * 3 * 3
        self.translates = torch.zeros(3).reshape(1, 3, 1).repeat(vertex_num, 1, 1)  # N * 3 * 1
        return

    def setGradState(self, need_grad: bool, vertex_mask: Union[torch.Tensor, None] = None) -> bool:
        if vertex_mask is None:
            self.rotate_matrixs.requires_grad_(need_grad)
            self.translates.requires_grad_(need_grad)
            return True

        self.rotate_matrixs[vertex_mask].requires_grad_(need_grad)
        self.translates[vertex_mask].requires_grad_(need_grad)
        return True

    def stiffness(self):
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]
        affine_weight = torch.cat((self.rotate_matrixs, self.translates), dim=2)  # N * 3 * 4
        w1 = torch.index_select(affine_weight, dim=0, index=idx1)
        w2 = torch.index_select(affine_weight, dim=0, index=idx2)
        w_diff = (w1 - w2)**2
        return w_diff
