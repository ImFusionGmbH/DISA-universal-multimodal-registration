import torch
import torch.nn.functional as F

class LC2:
    def __init__(self, radiuses=(3,5,7)):
        self.radiuses = radiuses
        self.f = torch.zeros(3, 1, 3, 3, 3)
        self.f[0, 0, 1, 1, 0] = 1
        self.f[0, 0, 1, 1, 2] = -1
        self.f[1, 0, 1, 0, 1] = 1
        self.f[1, 0, 1, 2, 1] = -1
        self.f[2, 0, 0, 1, 1] = 1
        self.f[2, 0, 2, 1, 1] = -1

    def __call__(self, us, mr):
        s = self.run(us, mr, self.radiuses[0])
        for r in self.radiuses[1:]:
            s += self.run(us, mr, r)
        return s / len(self.radiuses)

    def run(self, us, mr, radius=9, alpha=1e-3, beta=1e-2):
        us = us.squeeze(1)
        mr = mr.squeeze(1)

        bs = mr.size(0)
        pad = (mr.size(1) - (2*radius+1)) // 2
        count = (2*radius+1)**3

        self.f = self.f.to(mr.device)

        grad = torch.norm(F.conv3d(mr.unsqueeze(1), self.f, padding=1), dim=1)

        A = torch.ones(bs, 3, count, device=mr.device)
        A[:, 0] = mr[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        A[:, 1] = grad[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        b = us[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)

        C = torch.einsum("bip,bjp->bij", A, A) / count + torch.eye(3, device=mr.device).unsqueeze(0) * alpha
        Atb = torch.einsum("bip,bp->bi", A, b) / count
        coeff = torch.linalg.solve(C, Atb)
        var = torch.mean(b**2, dim=1) - torch.mean(b, dim=1)**2
        dist = torch.mean(b**2, dim=1) + torch.einsum("bi,bj,bij->b", coeff, coeff, C) - 2*torch.einsum("bi,bi->b", coeff, Atb)
        sym = (var - dist)/var.clamp_min(beta)
        
        return sym.clamp(0, 1)