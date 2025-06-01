import torch
import numpy as np
from typing import Union, Tuple

from non_rigid_icp.Data.mesh import Mesh
from non_rigid_icp.Method.trans import toTensor


class StiffnessConstraint(object):
    def __init__(self) -> None:
        # Source Data
        self.vertex_num = 0
        self.edges = np.empty((0, 2), dtype=np.int64)
        self.unique_vertex_group_idxs = np.empty(0, dtype=np.int64)

        # Processed Data
        self.vertex_group_idxs = torch.empty(0)
        self.compact_vertex_group_idxs = torch.empty(0)
        self.diff_group_edges = torch.empty(0)

        self.stiffness_weights_dict = {}
        return

    def isValid(self) -> bool:
        return len(self.stiffness_weights_dict) > 0

    def addEdgeConstraint(self, mesh: Mesh) -> bool:
        self.vertex_num = mesh.vertices.shape[0]

        if mesh.triangles.shape[0] == 0:
            print("[ERROR][StiffnessConstraint::addEdgeConstraint]")
            print("\t triangles is empty!")
            return False

        edges = np.vstack(
            [mesh.triangles[:, :2], mesh.triangles[:, 1:3], mesh.triangles[:, [0, 2]]]
        )
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        self.edges = edges
        return True

    def addVerticesGroupConstraint(
        self,
        vertex_group_idxs: Union[np.ndarray, list, None] = None,
        device: str = "cpu",
    ) -> bool:
        if vertex_group_idxs is None:
            vertex_group_idxs = np.arange(0, self.vertex_num, dtype=np.int64)
        if isinstance(vertex_group_idxs, list):
            vertex_group_idxs = np.asarray(vertex_group_idxs, dtype=np.int64)
        self.vertex_group_idxs = toTensor(vertex_group_idxs, device, torch.int64)

        self.unique_vertex_group_idxs = np.unique(vertex_group_idxs)
        group_mapping = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(self.unique_vertex_group_idxs)
        }
        compact_vertex_group_idxs = np.array(
            [group_mapping[idx] for idx in vertex_group_idxs]
        )
        self.compact_vertex_group_idxs = toTensor(
            compact_vertex_group_idxs, device, torch.int64
        )

        diff_group_edges_list = []
        for i, edge in enumerate(self.edges):
            v1, v2 = edge
            if compact_vertex_group_idxs[v1] != compact_vertex_group_idxs[v2]:
                diff_group_edges_list.append(i)
        diff_group_edges_idxs = np.asarray(diff_group_edges_list, dtype=np.int64)
        diff_group_edges = self.edges[diff_group_edges_idxs]
        self.diff_group_edges = toTensor(diff_group_edges, device, torch.int64)
        return True
