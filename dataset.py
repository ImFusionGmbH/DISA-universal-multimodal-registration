import numpy as np
import torch
from torch.utils.data import Dataset

class PatchDataset3D(Dataset):
    def __init__(self, paths, patch_size=33, repeats=1):
        self.patch_size = patch_size
        self.patch_radius = (patch_size - 1) // 2
        self.repeats = repeats

        self.mov = []
        self.fix = []
        self.indices = []

        for i, (path, mov, fix) in enumerate(paths):
            obj = np.load(path)
            r = self.patch_radius + 1
            self.mov.append(np.pad(obj[mov][0, :, :, :, 0], ((r, r), (r, r), (r, r)), mode="symmetric"))
            self.fix.append(np.pad(obj[fix][0, :, :, :, 0], ((r, r), (r, r), (r, r)), mode="symmetric"))
            lc2 = obj["lc2"]

            # Limit the pairs with LC2 < 0.1 to a 1/10th of the dataset
            small_count = 0
            discarded_count = 0
            for line, sym in zip(obj["indices"], lc2):
                if sym > 0.1 or small_count / len(lc2) < 0.1: 
                    small_count += sym < 0.1
                    us_pos = line[:3]
                    mr_pos = line[3:]
                    self.indices.append((i, us_pos, i, mr_pos))
                else:
                    discarded_count += 1

            print(f"Discarded {discarded_count} patches from {path}")

    def __len__(self):
        return len(self.indices) * self.repeats

    def sample_patch(self, vol, pos):
        x, y, z = np.maximum(pos, 0)
        # z = max(self.patch_radius, min(z, vol.shape[0] - self.patch_radius - 1)) - self.patch_radius
        # y = max(self.patch_radius, min(y, vol.shape[1] - self.patch_radius - 1)) - self.patch_radius
        # x = max(self.patch_radius, min(x, vol.shape[2] - self.patch_radius - 1)) - self.patch_radius
        res = vol[z:z+self.patch_size, y:y+self.patch_size, x:x+self.patch_size]
        if res.shape[2] != self.patch_size:
            print(pos)
            print(vol.shape)
            print(res.shape)
            print()
        return res

    def __getitem__(self, i):
        i = i % len(self.indices)
        us_index, us_pos, mr_index, mr_pos = self.indices[i]

        mov = np.ascontiguousarray(self.sample_patch(self.mov[us_index], us_pos))
        fix = np.ascontiguousarray(self.sample_patch(self.fix[mr_index], mr_pos))

        return torch.FloatTensor(mov).unsqueeze(0), torch.FloatTensor(fix).unsqueeze(0)
