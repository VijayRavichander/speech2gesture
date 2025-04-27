import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import numpy as np
from bvh import Bvh
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
from pathlib import Path

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
    

class BVHFolderMotionDataset(Dataset):
    """
    Each 4-second BVH clip in *folder_path* is one sample of shape
    (480, num_channels).  Global µ/σ normalisation is optional.
    """
    def __init__(self, folder_path, window_size=480, normalize=True):
        self.window_size = window_size          # kept for assertions
        self.normalize = normalize
        self.data = []

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
        return torch.from_numpy(self.data[idx])  # float32 by default
  # shape: (480, num_channels)

def train_motion_vqvae(model,
          kl_weight,
          train_dataloader,
          eval_dataloader,
          training_iterations, 
          evaluation_iterations,
          model_type="VQVAE"):
    
    print("Training...")
        
    device = "cuda"

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0005)

    train_losses = []
    evaluation_losses = []

    pbar = tqdm(range(training_iterations))
    
    train = True
    
    epoch = 0
    while train:

        train_loss = []
        eval_loss = []

        for batch_idx, motion_feature in enumerate(train_dataloader):
            motion_feature = motion_feature.to(device)

            x_recon, q_loss, perplexity, indices = model(motion_feature)
            
            reconstruction_loss = torch.mean((motion_feature-x_recon)**2)

            loss = q_loss + reconstruction_loss
    
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss.append(loss.item())
        
78
        if epoch % 50 == 0:
            print(f"Epoch {epoch}")
            print(f"Train Loss: {np.mean(train_loss)}")
        
        epoch += 1
        pbar.update(1)

        if epoch >= training_iterations:
            print("Completed Training!")
            train = False
            break

    print("Final Training Loss", train_loss[-1])
    
    return model, train_loss, train_loss
print("Loading the Train Dataset")

# Folder containing only .bvh files
dataset = BVHFolderMotionDataset("../data/bvh2clips", window_size=480)

print(len(dataset))

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

print("Model Init")

motion_vqvae = Motion_VQ_VAE(
    input_dim=228,
    code_dim=228,
    channels=[228, 228, 228],
    n_down=3,
    n_resblk=2,
    num_codes=256
)

print("Model Training")

trained_motion_vqvae, tl, el = train_motion_vqvae(motion_vqvae, None, 
                                                  dataloader, dataloader,
                                                  training_iterations=2000, evaluation_iterations=1000)

torch.save(trained_motion_vqvae.state_dict(), './saved_models/motion_vqvae_weights_1.pth')