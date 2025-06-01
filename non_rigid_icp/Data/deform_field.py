import torch
import numpy as np
from typing import Union, Tuple

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Method.render import renderEdges

from non_rigid_icp.Constraint.stiffness import StiffnessConstraint


class DeformField(object):
    def __init__(
        self,
        mesh: Mesh,
        device: str = "cpu",
        vertex_group_idxs: Union[np.ndarray, list, None] = None,
    ) -> None:
        self.stiffness_constraint = StiffnessConstraint()

        self.device = device

        self.stiffness_constraint.addEdgeConstraint(mesh)
        self.stiffness_constraint.addVerticesGroupConstraint(vertex_group_idxs, device)

        # Diff Params
        unique_group_num = self.stiffness_constraint.unique_vertex_group_idxs.shape[0]
        self.rotate_matrixs = (
            torch.eye(3).reshape(1, 3, 3).repeat(unique_group_num, 1, 1).to(self.device)
        )
        self.translates = (
            torch.zeros(3)
            .reshape(1, 3, 1)
            .repeat(unique_group_num, 1, 1)
            .to(self.device)
        )

        # renderEdges(mesh, self.diff_group_edges.detach().clone().cpu().numpy())
        return

    def setGradState(
        self, need_grad: bool, vertex_mask: Union[torch.Tensor, None] = None
    ) -> bool:
        if vertex_mask is None:
            self.rotate_matrixs.requires_grad_(need_grad)
            self.translates.requires_grad_(need_grad)
            return True

        self.rotate_matrixs[vertex_mask].requires_grad_(need_grad)
        self.translates[vertex_mask].requires_grad_(need_grad)
        return True

    def stiffness(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idx1 = self.stiffness_constraint.diff_group_edges[:, 0]
        idx2 = self.stiffness_constraint.diff_group_edges[:, 1]

        compact_idx1 = self.stiffness_constraint.compact_vertex_group_idxs[idx1]
        compact_idx2 = self.stiffness_constraint.compact_vertex_group_idxs[idx2]

        r1 = torch.index_select(self.rotate_matrixs, dim=0, index=compact_idx1)
        r2 = torch.index_select(self.rotate_matrixs, dim=0, index=compact_idx2)
        r_diff = (r1 - r2) ** 2

        t1 = torch.index_select(self.translates, dim=0, index=compact_idx1)
        t2 = torch.index_select(self.translates, dim=0, index=compact_idx2)
        t_diff = (t1 - t2) ** 2

        return r_diff, t_diff

    def toRotateMatrixs(self) -> torch.Tensor:
        return self.rotate_matrixs[self.stiffness_constraint.compact_vertex_group_idxs]

    def toTranslates(self) -> torch.Tensor:
        return self.translates[self.stiffness_constraint.compact_vertex_group_idxs]
