class StiffnessConstraint(object):
    def __init__(self) -> None:
        self.stiffness_weights_dict = {}
        return

    def isValid(self) -> bool:
        return len(self.stiffness_weights_dict) > 0
