import torch
import numpy as np
from torch import nn

from non_rigid_icp.Method.trans import toNumpy, toTensor


class AffineTransformLocal(nn.Module):
    """
    Implements a local affine transformation module as a neural network layer. This class is designed
    to apply affine transformations to features or coordinates in a localized manner, allowing for
    different transformations at different spatial locations or feature positions.
    The module includes stiffness term to ensure that the close points have similar transformation.
    """

    def __init__(self, vertex_num: int, triangles: torch.Tensor):
        """
        Initializes the LocalAffine module with the specified number of points and batch size.
        """
        super(AffineTransformLocal, self).__init__()

        triangles = toNumpy(triangles)

        edges = np.vstack(
            [triangles[:, :2], triangles[:, 1:3], triangles[:, [0, 2]]])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        edges = toTensor(edges, dtype=torch.int64)
        self.register_buffer("edges", edges)

        self.A = nn.Parameter(
            torch.eye(3).reshape(1, 3, 3).repeat(vertex_num, 1, 1))  # N * 3 * 3
        self.b = nn.Parameter(
            torch.zeros(3).reshape(1, 3, 1).repeat(vertex_num, 1, 1))  # N * 3 * 1
        return

    def stiffness(self):
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]
        affine_weight = torch.cat((self.A, self.b), dim=2)  # N * 3 * 4
        w1 = torch.index_select(affine_weight, dim=0, index=idx1)
        w2 = torch.index_select(affine_weight, dim=0, index=idx2)
        w_diff = (w1 - w2)**2
        return w_diff

    def forward(self, vertices: torch.Tensor):
        vertices = vertices.unsqueeze(-1)
        out_x = torch.matmul(self.A, vertices)
        out_x = out_x + self.b
        return out_x.squeeze(-1)
