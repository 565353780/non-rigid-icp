import torch
import torch.nn as nn


class AffineTransformLocal(nn.Module):
    """
    Implements a local affine transformation module as a neural network layer. This class is designed
    to apply affine transformations to features or coordinates in a localized manner, allowing for
    different transformations at different spatial locations or feature positions.
    The module includes stiffness term to ensure that the close points have similar transformation.
    """

    def __init__(self, num_points: int, edges: torch.Tensor):
        """
        Initializes the LocalAffine module with the specified number of points and batch size.
        """
        super(AffineTransformLocal, self).__init__()
        self.A = nn.Parameter(
            torch.eye(3).reshape(1, 3, 3).repeat(num_points, 1, 1))  # N * 3 * 3
        self.b = nn.Parameter(
            torch.zeros(3).reshape(1, 3, 1).repeat(num_points, 1, 1))  # N * 3 * 1
        self.edges = edges
        self.num_points = num_points
        return

    def stiffness(self):
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]
        affine_weight = torch.cat((self.A, self.b), dim=2)  # N * 3 * 4
        w1 = torch.index_select(affine_weight, dim=0, index=idx1)
        w2 = torch.index_select(affine_weight, dim=0, index=idx2)
        w_diff = (w1 - w2)**2
        return w_diff

    def forward(self, x, return_stiff=False):
        '''
            x should have shape of N * 3
        '''
        x = x.unsqueeze(-1)
        out_x = torch.matmul(self.A, x)
        out_x = out_x + self.b
        out_x.squeeze_(-1)
        if return_stiff:
            stiffness = self.stiffness()
            return out_x, stiffness
        else:
            return out_x


if __name__ == "__main__":
    # Test the LocalAffine module
    num_points = 10
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
                          [6, 7], [7, 8], [8, 9], [9, 0]])
    x = torch.randn(num_points, 3)
    local_affine = AffineTransformLocal(num_points, edges)
    out_x, stiffness = local_affine(x, return_stiff=True)
    print(out_x.shape, stiffness.shape, local_affine.A.shape)
    print(out_x)
    print(stiffness)
    print(local_affine.A)
