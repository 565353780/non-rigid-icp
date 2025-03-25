import torch


@torch.no_grad()
def toL1ChamferDistance(dist1: torch.Tensor, dist2: torch.Tensor) -> float:
    dist1 = torch.sqrt(dist1.detach().clone())
    dist2 = torch.sqrt(dist2.detach().clone())

    mean_dist1 = torch.mean(dist1)
    mean_dist2 = torch.mean(dist2)

    l1_chamfer = mean_dist1 + mean_dist2

    return l1_chamfer.item()

@torch.no_grad()
def toL2ChamferDistance(dist1: torch.Tensor, dist2: torch.Tensor) -> float:
    dist1 = dist1.detach().clone()
    dist2 = dist2.detach().clone()

    mean_dist1 = torch.mean(dist1)
    mean_dist2 = torch.mean(dist2)

    l2_chamfer = mean_dist1 + mean_dist2

    return l2_chamfer.item()
