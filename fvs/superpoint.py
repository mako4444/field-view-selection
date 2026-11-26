"""SuperPoint (DeTone et al., CVPRW 2018) with the layout of the released weights `superpoint_v1.pth`."""
import torch
import torch.nn as nn


class SuperPoint(nn.Module):
    def __init__(self, n_cell_size: int = 8, dim_desc: int = 256):
        super().__init__()
        self.n_cell_size = n_cell_size
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, dim_desc
        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = nn.Conv2d(c5, d1, 1, 1, 0)
        self.softmax = nn.Softmax(dim=1)


def load_superpoint(path, dev):
    sp = SuperPoint().to(dev).eval()
    sp.load_state_dict(torch.load(path, map_location='cpu'))
    return sp


@torch.no_grad()
def superpoint_det_desc(sp, imgs: torch.Tensor, chunk: int = 4):
    """imgs (B,H,W,3) in [0,1]. Returns the full-resolution detection map (B,H,W) and the
    descriptor map at 1/8 resolution (B,256,H/8,W/8)."""
    from torchvision.transforms.functional import rgb_to_grayscale
    dets, descs = [], []
    cs = sp.n_cell_size
    for s in range(0, imgs.shape[0], chunk):
        x = imgs[s:s + chunk]
        b, H, W, _ = x.shape
        z = rgb_to_grayscale(x.permute(0, 3, 1, 2), num_output_channels=1)
        z = sp.relu(sp.conv1a(z)); z = sp.relu(sp.conv1b(z)); z = sp.pool(z)
        z = sp.relu(sp.conv2a(z)); z = sp.relu(sp.conv2b(z)); z = sp.pool(z)
        z = sp.relu(sp.conv3a(z)); z = sp.relu(sp.conv3b(z)); z = sp.pool(z)
        z = sp.relu(sp.conv4a(z)); z = sp.relu(sp.conv4b(z))
        semi = sp.convPb(sp.relu(sp.convPa(z)))
        prob = sp.softmax(semi)
        nodust = prob[:, :-1].permute(0, 2, 3, 1)
        det = nodust.reshape(b, H // cs, W // cs, cs, cs)
        det = det.permute(0, 1, 3, 2, 4).reshape(b, (H // cs) * cs, (W // cs) * cs)
        d = sp.convDb(sp.relu(sp.convDa(z)))
        d = d / d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        dets.append(det); descs.append(d)
    return torch.cat(dets), torch.cat(descs)
