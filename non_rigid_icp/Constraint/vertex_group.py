import numpy as np


class VertexGroupConstraint(object):
    def __init__(self):
        self.vertex_group_dict = {}
        return

    def addConstraint(self, group_id: int, vertex_idxs: np.ndarray) -> bool:
        if group_id not in self.vertex_group_dict.keys():
            self.vertex_group_dict[group_id] = vertex_idxs
        else:
            self.vertex_group_dict[group_id] = self.vertex_group_dict[group_id].union(
                vertex_idxs
            )
        return True

    def getConstraint(self, vertex_num: int) -> np.ndarray:
        min_individual_group_idx = max(list(self.vertex_group_dict.keys())) + 1

        vertex_group_idxs = np.arange(
            min_individual_group_idx,
            min_individual_group_idx + vertex_num,
            dtype=np.int64,
        )
        for group_id, vertex_idxs in self.vertex_group_dict.items():
            vertex_group_idxs[vertex_idxs] = group_id

        return vertex_group_idxs
