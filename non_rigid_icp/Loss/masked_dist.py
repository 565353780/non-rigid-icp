import torch

def toMaskedDistLoss(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    max_dist: float = 0.04,
) -> torch.Tensor:
    dists2 = (pts1 - pts2)**2

    dists_mask = torch.sum(dists2, dim=1) < max_dist**2

    masked_dists2 = dists2[dists_mask]

    masked_dist_loss = torch.sum(masked_dists2)

    return masked_dist_loss
