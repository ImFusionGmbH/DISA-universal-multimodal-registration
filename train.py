import torch
from torch.utils.data import DataLoader
from neural import DISANet
from dataset import PatchDataset3D
import numpy as np
import torch.nn.functional as F
from lc2 import LC2
from grid import *
from tqdm import tqdm
import os
import torch.optim


class LRScheduler:
    def __init__(self, optimizer):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.count = 0
        self.lr = 0

    def compute_lr(self, step: int, loss_value: float):
        raise NotImplementedError

    def step(self, loss_value=0.0, size=1):
        self.count += size

        self.lr = self.compute_lr(self.count, loss_value)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        if self.count % 50 == 0:
            print("Learning rate:", self.lr)

        return self.lr


class WarmupStepScheduler(LRScheduler):
    def __init__(self, optimizer, lr, warmup_batches=500, drop_every=100000, gamma=0.33):
        self.initial_lr = lr
        self.warmup_batches = warmup_batches
        self.drop_every = drop_every
        self.gamma = gamma

        super().__init__(optimizer)

    def compute_lr(self, step: int, loss_value: float):
        # first batches do LR warmup
        if step <= self.warmup_batches:
            return self.initial_lr * (np.exp((step / self.warmup_batches)) - 1) / (np.exp(1) - 1)
        else:
            return self.initial_lr * (self.gamma ** (step // self.drop_every))



class Trainer:
    def __init__(self):
        self.device = torch.device("cuda")
        self.epochs = 35

        model = DISANet().to(self.device)
        self.model = torch.compile(model)

        self.dataloader = self.get_dataloader(256)
        self.validation_dataloader = self.get_validation_dataloader(256)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3, amsgrad=False)
        self.scheduler = WarmupStepScheduler(self.optimizer, lr=1e-3, warmup_batches=200, drop_every=8000, gamma=0.5)

        self.lc2 = LC2(radiuses=[3,5,7])

    def get_dataloader(self, batch_size):
        paths = [(f"/data/{i}.npz", "mov", "fix") for i in range(0, 48)]

        dataset = PatchDataset3D(paths, 51, repeats=1)
        
        print("Train dataset contains", len(dataset), "patches")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=False, drop_last=True)

    def get_validation_dataloader(self, batch_size):
        paths = [(f"/data/{i}.npz", "mov", "fix") for i in range(48, 54)]
        dataset = PatchDataset3D(paths, 51)
        
        print("Validation dataset contains", len(dataset), "patches")
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=False, drop_last=False)

    def step(self, vol_a, vol_b, augment=True):
        bs = vol_a.size(0)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                grid = create_grid(vol_a.size(), 33, vol_a.device)
                if augment:
                    # Random invert
                    p = 0.3
                    vol_a = ((torch.rand(vol_a.shape[0], 1, 1, 1, 1, device=vol_a.device) > p).float()*2 - 1) * vol_a
                    vol_b = ((torch.rand(vol_b.shape[0], 1, 1, 1, 1, device=vol_a.device) > p).float()*2 - 1) * vol_b

                    # Random shift and scale
                    shift = 0.2
                    scale = 0.3
                    vol_a = shift*torch.randn(vol_a.shape[0], 1, 1, 1, 1, device=vol_a.device) + vol_a
                    vol_a = vol_a * (1 + scale*(torch.rand(vol_a.shape[0], 1, 1, 1, 1, device=vol_a.device)*2 - 1))
                    vol_b = shift*torch.randn(vol_b.shape[0], 1, 1, 1, 1, device=vol_b.device) + vol_b
                    vol_b = vol_b * (1 + scale*(torch.rand(vol_b.shape[0], 1, 1, 1, 1, device=vol_b.device)*2 - 1))

                    # Random noise
                    noise = 0.01
                    vol_a = vol_a + noise * torch.randn(vol_a.shape[0], 1, 1, 1, 1, device=vol_a.device)
                    vol_b = vol_b + noise * torch.randn(vol_b.shape[0], 1, 1, 1, 1, device=vol_b.device)

                    grid = random_flip(grid)
                    grid = random_rotation(grid)

                vol_b = F.grid_sample(vol_b, grid, align_corners=True, padding_mode="border")
                vol_a = F.grid_sample(vol_a, grid, align_corners=True, padding_mode="border")
                
                target = self.lc2(vol_a, vol_b)

        x = torch.cat((vol_a, vol_b), dim=0)

        y = self.model(x)
        assert y.size(2) % 2 == 1
        c = y.size(2) // 2
        y = y[:, :, c, c, c]

        y = y / torch.norm(y, dim=1, keepdim=True).clamp_min(1.0)

        y_us = y[:bs]
        y_mr = y[bs:]

        pred = torch.einsum("bi,bi->b", y_us, y_mr)
        return torch.mean((pred - target)**2)

    def train(self):
        for epoch in range(self.epochs):
            # Train loop
            self.model.train()
            epoch_loss = 0
            num_batches = len(self.dataloader)
            for batch_idx, (vol_a, vol_b) in enumerate(self.dataloader):
                vol_a = vol_a.to(self.device)
                vol_b = vol_b.to(self.device)
                
                loss = self.step(vol_a, vol_b, augment=True)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss.item())

                epoch_loss += loss.item()

                if batch_idx % 20 == 0:
                    print(f"Batch {batch_idx} / {num_batches}, loss: {loss.item()}")

            print(f"Epoch {epoch+1}, Training loss: {epoch_loss / num_batches}")

            # Validation loop
            self.model.eval()
            val_loss = 0
            num_batches = len(self.validation_dataloader)
            for batch_idx, (vol_a, vol_b) in enumerate(tqdm(self.validation_dataloader)):
                with torch.no_grad():
                    vol_a = vol_a.to(self.device)
                    vol_b = vol_b.to(self.device)

                    loss = self.step(vol_a, vol_b, augment=False)
                    val_loss += loss.item()
            print(f"Epoch {epoch+1}, Validation loss: {val_loss / num_batches}")

            path = os.path.join("/output", f"{epoch}")
            print(f"Saving model checkpoint at '{path}'")

            torch.save(self.model.state_dict(), f"{path}.pth")


def main():
    seed = 42
    torch.manual_seed(seed)

    global _seed
    _seed = seed
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    Trainer().train()


if __name__ == "__main__":
    main()
