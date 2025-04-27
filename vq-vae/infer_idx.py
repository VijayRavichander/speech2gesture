from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import os
from bvh import Bvh
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from bvh import Bvh  # Ensure you have a BVH parser installed
from collections import defaultdict
import re
from pathlib import Path

class BVHFolderMotionDataset(Dataset):
    """
    Each 4-second BVH clip in *folder_path* is one sample of shape
    (480, num_channels).  Global µ/σ normalisation is optional.
    """
    def __init__(self, folder_path, window_size=480, normalize=True):
        self.window_size = window_size          # kept for assertions
        self.normalize = normalize
        self.data = []
        self.files: list[Path]       = []

        # load every .bvh (recursively)
        for file in Path(folder_path).rglob("*.bvh"):
            with file.open() as f:
                mocap = Bvh(f.read())          # lightweight parser :contentReference[oaicite:3]{index=3}
            frames = np.asarray(mocap.frames, dtype=np.float32)

            # safeguard: drop files that are not exactly 480 frames
            if len(frames) != self.window_size:
                print(f"skip {file.name}: {len(frames)} frames")
                continue

            self.data.append(frames)
            self.files.append(file)  

        if not self.data:
            raise RuntimeError("No 480-frame BVH clips found.")

        self.data = np.stack(self.data)         # (N, 480, C)

        if normalize:
            self.mean = self.data.mean((0, 1), keepdims=True)
            self.std  = self.data.std((0, 1), keepdims=True) + 1e-8
            self.data = (self.data - self.mean) / self.std
        else:
            self.mean = self.std = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        motion = torch.from_numpy(self.data[idx])  
        name   = self.files[idx].name 
        return motion, name  # float32 by default

# Residual Block with 1D Conv
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)

# Encoder for BVH motion sequences
class VQEncoder(nn.Module):
    def __init__(self, in_dim, channels, n_down):
        super().__init__()
        layers = [nn.Conv1d(in_dim, channels[0], kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True),
                  ResBlock(channels[0])]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i - 1], channels[i], kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i])
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.permute(0, 2, 1)

# Decoder for BVH motion sequences
class VQDecoder(nn.Module):
    def __init__(self, input_size, channels, n_resblk, n_up, output_dim = 228):
        super().__init__()
        layers = []
        if channels[0] != input_size:
            layers.append(nn.Conv1d(input_size, channels[0], kernel_size=3, padding=1))

        for _ in range(n_resblk):
            layers.append(ResBlock(channels[0]))

        for i in range(n_up - 1):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        layers.append(nn.Conv1d(channels[-1], channels[-1], kernel_size=3, padding=1))

        layers += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(channels[-1], output_dim, kernel_size=3, padding=1)
        ]

        self.net = nn.Sequential(*layers)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.permute(0, 2, 1)

# Vector Quantizer
class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))

        return loss, z_q, min_encoding_indices, perplexity

    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def get_codebook_entry(self, indices):
        """
        :param indices(B, seq_len):
        :return z_q(B, seq_len, e_dim):
        """
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q 

# Wrapper: VQ-VAE Model
class Motion_VQ_VAE(nn.Module):
    def __init__(self, input_dim, code_dim, channels, n_down=3, n_resblk=3, num_codes=512, beta=0.25):
        super().__init__()
        self.encoder = VQEncoder(input_dim, channels, n_down)
        self.quantizer = Quantizer(num_codes, code_dim, beta)
        self.decoder = VQDecoder(code_dim, list(reversed(channels)), n_resblk, n_down)

    def forward(self, x):
        z_e = self.encoder(x)

        loss, z_q, indices, perplexity = self.quantizer(z_e)

        x_recon = self.decoder(z_q)

        return x_recon, loss, perplexity, indices

    def encode(self, x):
        z_e = self.encoder(x)
        return self.quantizer.encode(z_e)

    def decode(self, indices):
        z_q = self.quantizer.decode(indices)
        return self.decoder(z_q)
    
print("Loading the Test Dataset")
dataset = BVHFolderMotionDataset("../data/bvh2clips", window_size=480)

pred = [116, 119, 218, 129, 48, 3, 164, 13, 97, 245, 216, 76, 75, 36, 178, 58, 28, 181, 118, 89, 144, 244, 149, 95, 9, 149, 209, 119, 149, 39, 152, 213, 49, 199, 126, 229, 103, 51, 226, 138, 163, 253, 96, 60, 202, 251, 232, 119, 208, 121, 20, 45, 105, 181, 216, 218, 169, 200, 211, 144, 44, 57, 55, 126, 67, 20, 158, 186, 62, 207, 107, 59, 133, 247, 145, 20, 48, 122, 126, 127, 156, 125, 35, 79, 127, 117, 132, 20, 7, 132, 4, 104, 207, 57, 189, 171, 185, 74, 20, 225, 34, 121, 246, 5, 73, 160, 230, 42, 12, 226, 138, 20, 130, 148, 119, 80, 107, 1, 236, 46, 151, 19, 182, 122, 250, 172, 89, 24, 231, 187, 63, 236, 82, 40, 141, 110, 66, 209, 29, 142, 176, 0, 254, 20, 171, 34, 53, 39, 127, 177, 154, 199, 127, 182, 234, 78]

motion_vqvae = Motion_VQ_VAE(
    input_dim=228,
    code_dim=228,
    channels=[228, 228, 228],
    n_down=3,
    n_resblk=2,
    num_codes=256
)
# motion_vqvae = motion_vqvae.to("cuda")
motion_vqvae.load_state_dict(torch.load("./saved_models/motion_vqvae_weights_1.pth"))
motion_vqvae.eval()

pred_tensor = torch.tensor([pred])
# pred_tensor = pred_tensor.to("cuda")
z_q = motion_vqvae.quantizer.get_codebook_entry(pred_tensor)
x_recon = motion_vqvae.decoder(z_q)

print(x_recon.shape)

def write_bvh(file_path, motion_tensor, hierarchy_str):
    """
    Writes a BVH file with the given hierarchy and motion tensor.
    
    Args:
        file_path (str): Output path for the .bvh file.
        motion_tensor (torch.Tensor): Shape (N, T, D) to be reshaped to (N*T, D).
        hierarchy_str (str): Hierarchy text block from original BVH.
    """
    assert motion_tensor.ndim == 3, "Expected 3D tensor (B, T, C)"
    
    # Flatten the tensor (e.g., 10, 480, 228 → 4800, 228)
    flat_motion = motion_tensor.reshape(-1, motion_tensor.size(-1))  # shape: (4800, 228)
    
    with open(file_path, "w") as f:
        # Write hierarchy first
        f.write(hierarchy_str.strip() + "\n")

        # Add motion header
        f.write(f"MOTION\n")
        f.write(f"Frames: {flat_motion.shape[0]}\n")
        f.write(f"Frame Time: 0.008333\n")

        # Write each frame as a space-separated float line
        for frame in flat_motion:
            line = " ".join([f"{v:.6f}" for v in frame.tolist()])
            f.write(line + "\n")

    print(f"✅ Wrote BVH to: {file_path}")
def extract_bvh_hierarchy(bvh_file):
    with open(bvh_file, 'r') as f:
        lines = f.readlines()

    motion_index = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    hierarchy = "".join(lines[:motion_index])

    return hierarchy

test_bvh_file = "../data/bvh2clips_test/Copy of 1_wayne_0_1_1_part_000.bvh"
template_hierarchy = extract_bvh_hierarchy(test_bvh_file) 
new_file_name = f"./preds/test.bvh"
mean = torch.tensor(dataset.mean, dtype=x_recon.dtype, device=x_recon.device)
std = torch.tensor(dataset.std, dtype=x_recon.dtype, device=x_recon.device)
x_recon_denorm = x_recon * std + mean
write_bvh(new_file_name, x_recon_denorm, template_hierarchy)