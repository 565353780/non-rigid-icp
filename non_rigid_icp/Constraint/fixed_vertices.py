import numpy as np
from typing import Tuple


class FixedVerticesConstraint(object):
    def __init__(self):
        self.fixed_vertices_dict = {}
        return

    def isValid(self) -> bool:
        return len(self.fixed_vertices_dict) > 0

    def addConstraint(
        self, vertex_idxs: np.ndarray, target_positions: np.ndarray
    ) -> bool:
        if vertex_idxs.shape[0] != target_positions.shape[0]:
            print("[ERROR][FixedVerticesConstraint::addConstraint]")
            print("\t vertex_idxs and target_positions size mismatch!")
            print("\t vertex_idxs.shape:", vertex_idxs.shape)
            print("\t target_positions.shape:", target_positions.shape)
            return False

        for i in range(vertex_idxs.shape[0]):
            self.fixed_vertices_dict[vertex_idxs[i]] = target_positions[i]

        return True

    def getConstraint(self) -> Tuple[np.ndarray, np.ndarray]:
        vertex_idxs = []
        target_positions = []

        for vertex_idx, target_position in self.fixed_vertices_dict.items():
            vertex_idxs.append(vertex_idx)
            target_positions.append(target_position)

        veridxs_idxs = np.asarray(vertex_idxs, dtype=np.int64)
        target_positions = np.asarray(target_positions, dtype=np.float32)
        return veridxs_idxs, target_positions
