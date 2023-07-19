import torch
import roma

def create_grid(shape, out_size, device):
    dx = torch.linspace(-(out_size - 0.5)/shape[4], (out_size - 0.5)/shape[4], out_size, device=device)
    dy = torch.linspace(-(out_size - 0.5)/shape[3], (out_size - 0.5)/shape[3], out_size, device=device)
    dz = torch.linspace(-(out_size - 0.5)/shape[2], (out_size - 0.5)/shape[2], out_size, device=device)

    grid = torch.cartesian_prod(dx, dy, dz).reshape(1, out_size, out_size, out_size, 3)
    grid = grid.permute(0, 3, 2, 1, 4).repeat(shape[0], 1, 1, 1, 1)
    return grid

def random_flip(grid):
    flipper = ((torch.rand(grid.size(0), 1, 1, 1, 3, device=grid.device) > 0.5).to(torch.float32) * 2 - 1)
    return grid * flipper


def random_rotation(grid):
    return torch.einsum("bijhc,bcd->bijhd", grid, roma.random_rotmat(grid.size(0), device=grid.device))