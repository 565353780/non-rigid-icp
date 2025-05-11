import torch
import numpy as np
from torch import nn
from typing import Union

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Method.trans import toNumpy, toTensor


class DeformField(object):
    def __init__(
        self,
        mesh: Mesh,
        device: str = 'cpu',
        vertex_group_idxs: Union[np.ndarray, list, None] = None,
    ) -> None:
        self.device = device

        triangles = mesh.triangles

        edges = np.vstack(
            [triangles[:, :2], triangles[:, 1:3], triangles[:, [0, 2]]])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        # Source Data
        self.vertex_num = mesh.vertices.shape[0]
        self.edges = edges

        # Processed Data
        self.vertex_group_idxs = torch.empty(0)
        self.compact_vertex_group_idxs = torch.empty(0)
        self.diff_group_edges = torch.empty(0)

        # Diff Params
        self.rotate_matrixs = torch.empty(0)
        self.translates = torch.empty(0)

        self.updateGroupIdxs(vertex_group_idxs)
        return

    def updateGroupIdxs(
        self,
        vertex_group_idxs: Union[np.ndarray, list, None] = None,
    ) -> bool:
        if vertex_group_idxs is None:
            vertex_group_idxs = np.arange(0, self.vertex_num, dtype=np.int64)
        if isinstance(vertex_group_idxs, list):
            vertex_group_idxs = np.asarray(vertex_group_idxs, dtype=np.int64)
        self.vertex_group_idxs = toTensor(vertex_group_idxs, self.device, torch.int64)

        unique_groups = np.unique(vertex_group_idxs)
        group_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_groups)}
        compact_vertex_group_idxs = np.array([group_mapping[idx] for idx in vertex_group_idxs])
        self.compact_vertex_group_idxs = toTensor(compact_vertex_group_idxs, self.device, torch.int64)

        diff_group_edges_list = []
        for i, edge in enumerate(self.edges):
            v1, v2 = edge
            if compact_vertex_group_idxs[v1] != compact_vertex_group_idxs[v2]:
                diff_group_edges_list.append(i)
        diff_group_edges_idxs = np.asarray(diff_group_edges_list, dtype=np.int64)
        diff_group_edges = self.edges[diff_group_edges_idxs]
        self.diff_group_edges = toTensor(diff_group_edges, self.device, torch.int64)

        unique_group_num = unique_groups.shape[0]
        self.rotate_matrixs = torch.eye(3).reshape(1, 3, 3).repeat(unique_group_num, 1, 1).to(self.device)
        self.translates = torch.zeros(3).reshape(1, 3, 1).repeat(unique_group_num, 1, 1).to(self.device)
        return True

    def setGradState(self, need_grad: bool, vertex_mask: Union[torch.Tensor, None] = None) -> bool:
        if vertex_mask is None:
            self.rotate_matrixs.requires_grad_(need_grad)
            self.translates.requires_grad_(need_grad)
            return True

        self.rotate_matrixs[vertex_mask].requires_grad_(need_grad)
        self.translates[vertex_mask].requires_grad_(need_grad)
        return True

    def stiffness(self):
        idx1 = self.diff_group_edges[:, 0]
        idx2 = self.diff_group_edges[:, 1]

        compact_idx1 = self.compact_vertex_group_idxs[idx1]
        compact_idx2 = self.compact_vertex_group_idxs[idx2]

        affine_weight = torch.cat((self.rotate_matrixs, self.translates), dim=2)  # N * 3 * 4
        w1 = torch.index_select(affine_weight, dim=0, index=compact_idx1)
        w2 = torch.index_select(affine_weight, dim=0, index=compact_idx2)
        w_diff = (w1 - w2)**2
        return w_diff

    def toRotateMatrixs(self) -> torch.Tensor:
        return self.rotate_matrixs[self.compact_vertex_group_idxs]

    def toTranslates(self) -> torch.Tensor:
        return self.translates[self.compact_vertex_group_idxs]
