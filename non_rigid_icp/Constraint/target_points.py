import torch
import numpy as np
from typing import Union

from non_rigid_icp.Method.trans import toTensor


class TargetPointsConstraint(object):
    def __init__(self) -> None:
        self.points_list = []

        self.points_tensor = None
        return

    def isValid(self) -> bool:
        if len(self.points_list) == 0:
            return False

        for points in self.points_list:
            if points.shape[0] > 0:
                return True

        return False

    def addConstraint(self, points: np.ndarray) -> bool:
        if points.shape[0] == 0:
            print("[ERROR][TargetPointsConstraint::addConstraint]")
            print("\t points is empty!")
            return False

        self.points_list.append(points)
        return True

    def getConstraint(self) -> Union[np.ndarray, None]:
        if len(self.points_list) == 0:
            return None

        target_points = np.vstack(self.points_list)

        return target_points

    def updateTensor(self, device: str = "cpu", dtype=torch.float32) -> bool:
        target_points = self.getConstraint()
        if target_points is None:
            print("[ERROR][TargetPointsConstraint::updateTensor]")
            print("\t getConstraint returns None!")
            return False

        self.points_tensor = toTensor(target_points, device, dtype).unsqueeze(0)
        return True
