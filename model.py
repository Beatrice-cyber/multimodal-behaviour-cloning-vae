COLLAB = True
if COLLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    !pip install --force-reinstall git+https://github.com/joshuaspear/pymlrf.git
    !pip install wandb
    !pip install torchinfo
    !pip install jaxtyping
    !pip install git+https://github.com/Beatrice-cyber/comp0188_cw2_public.git
    !pip install typeguard==2.13.3
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import datetime
from pymlrf.Structs.torch import DatasetOutput
import copy
from comp0188_cw2.utils import load_all_files
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
from typing import List
from pymlrf.Structs.torch import DatasetOutput

from comp0188_cw2.config import (
    train_dh, val_dh, test_dh, WANDB_PROJECT
    )
from comp0188_cw2.models.CNNConfig import ConvolutionLayersConfig
from comp0188_cw2.models.base import BaseModel
from comp0188_cw2.models.JointCNNEncoder import JointCNNEncoder
from comp0188_cw2.models.CNN import CNN
from comp0188_cw2.models.MLP import MLP
from comp0188_cw2.Metric.WandBMetricOrchestrator import WandBMetricOrchestrator
from comp0188_cw2.Dataset.NpDictDataset import NpDictDataset
from comp0188_cw2.Loss.BalancedLoss import TrackerBalancedLoss
from comp0188_cw2 import logger
from comp0188_cw2.training.TrainingLoop import TorchTrainingLoop

pos_criterion = nn.MSELoss(reduction="mean")
grp_criterion = nn.CrossEntropyLoss(reduction="mean")
def collate_func(input_list:List[DatasetOutput])->DatasetOutput:
    pos = []
    _grp = []
    images = []
    obs = []
    for val in input_list:
        images.append(
            torch.concat(
                [val.input["front_cam_ob"], val.input["mount_cam_ob"]],
                dim=0
            )[None,:]
            )
        obs.append(
            torch.concat(
                [
                    val.input["ee_cartesian_pos_ob"],
                    val.input["ee_cartesian_vel_ob"],
                    val.input["joint_pos_ob"]
                    ],
                dim=0
            )[None,:]
        )
        pos.append(val.output["actions"][0:3][None,:])
        _grp.append(val.output["actions"][-1:][None])
    _grp = torch.concat(_grp, dim=0)
    grp = torch.zeros(_grp.shape[0],3)
    grp[torch.arange(len(grp)), _grp.squeeze().int()] = 1
    return DatasetOutput(
        input = {
            "images":torch.concat(images,dim=0),
            "obs":torch.concat(obs,dim=0),
            },
        output = {
            "pos":torch.concat(pos, dim=0),
            "grp":grp
            }
    )

class VAE_Encoder(nn.Module):
    def __init__(self, vector_latent_dim, latent_dim):
        super(VAE_Encoder, self).__init__()
        # CNNs for images
        self.image_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,kernel_size=(3, 3),stride=1,padding=1,dilation=1 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d( in_channels=8,out_channels=16,kernel_size=(3, 3), stride=1,padding=1 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=32,kernel_size=(3, 3),stride=1, padding=1 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=(3, 3),stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # MLPs for vectors
        self.vector_mlp1 = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, vector_latent_dim)
        )
        self.vector_mlp2 = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, vector_latent_dim)
        )
        self.vector_mlp3 = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, vector_latent_dim)
        )

        # Combine all embeddings
        cnn_output_size = self.image_cnn(torch.randn(1, 1, 224, 224)).view(1, -1).shape[1]
        combined_dim = 2 * cnn_output_size + 3 * vector_latent_dim
        self.image_latent_dim = cnn_output_size
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, latent_dim)

    def forward(self, img_front, img_mount, pos_ob, vel_ob, joint_ob):
        front_latent = self.image_cnn(img_front).view(img_front.size(0), -1)
        mount_latent = self.image_cnn(img_mount).view(img_mount.size(0), -1)
        pos_latent = self.vector_mlp1(pos_ob)
        vel_latent = self.vector_mlp2(vel_ob)
        joint_latent = self.vector_mlp3(joint_ob)

        combined = torch.cat([front_latent, mount_latent, pos_latent, vel_latent, joint_latent], dim=-1)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar


#  sample from normal dis to get z
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)  # Sample from N(0, 1)
    return mu + eps * std

# decoder train a mapping from z to x and get x_pred
class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, image_latent_dim, channels, height, width ,vector_latent_dim):
        super(VAE_Decoder, self).__init__()
        self.image_latent_dim = image_latent_dim
        self.vector_latent_dim = vector_latent_dim
        self.channels = channels
        self.height = height
        self.width = width
        # Fully connected layers to map z back to combined latent space
        self.fc = nn.Linear(latent_dim, 2 * (channels * height * width) + 3 * vector_latent_dim )

        # Decoders for images
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32, out_channels= 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=8 , out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Sigmoid()  # Use sigmoid to scale pixel values to [0, 1]
        )

        # Decoders for vectors
        self.vector_decoder1 = nn.Sequential(
            nn.Linear(vector_latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # Output size matches original pos_ob
        )
        self.vector_decoder2 = nn.Sequential(
            nn.Linear(vector_latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # Output size matches original vel_ob
        )
        self.vector_decoder3 = nn.Sequential(
            nn.Linear(vector_latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output size matches original joint_ob
        )

    def forward(self, z):
        vector_latent_dim = self.vector_latent_dim
        # Map latent vector z back to combined latent space
        combined_latent = self.fc(z)
        assert combined_latent.shape[-1] == 2 * (self.channels * self.height * self.width) + 3 * vector_latent_dim, \
    f"Expected {2 * (self.channels * self.height * self.width) + 3 * vector_latent_dim}, got {combined_latent.shape[-1]}"

        # Split combined latent space into components
        front_latent, mount_latent, pos_latent, vel_latent, joint_latent = torch.split(
            combined_latent, [ self.image_latent_dim,  self.image_latent_dim, vector_latent_dim, vector_latent_dim, vector_latent_dim], dim=-1
        )

        # Decode each component
        recon_img_front = self.image_decoder(front_latent.view(-1, self.channels , self.height , self.width))  # Reshape for deconvolution
        recon_img_mount = self.image_decoder(mount_latent.view(-1, self.channels , self.height , self.width))
        recon_pos_ob = self.vector_decoder1(pos_latent)
        recon_vel_ob = self.vector_decoder2(vel_latent)
        recon_joint_ob = self.vector_decoder3(joint_latent)

        return recon_img_front, recon_img_mount, recon_pos_ob, recon_vel_ob, recon_joint_ob



# use use mse and kl to have loss
class VAE(nn.Module):
    def __init__(self,  vector_latent_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(vector_latent_dim, latent_dim)
        image_latent_dim = self.encoder.image_latent_dim

        cnn_output_size = self.encoder.image_cnn(torch.randn(1, 1, 224, 224)).shape  # (1, channels, height, width)
        print(f"Encoder CNN output size: {cnn_output_size}")

        initial_channels = cnn_output_size[1]
        initial_height = cnn_output_size[2]
        initial_width = cnn_output_size[3]
        print(f"Channels: {initial_channels}, Height: {initial_height}, Width: {initial_width}")
        print(f"Computed image_latent_dim: {image_latent_dim}")

        self.decoder = VAE_Decoder(latent_dim, image_latent_dim, initial_channels, initial_height, initial_width, vector_latent_dim)

    def forward(self, img_front, img_mount, pos, vel, joint):
        mu, logvar = self.encoder(img_front, img_mount, pos, vel, joint)
        print(f"Mean of mu: {mu.mean().item()}, Std of logvar: {logvar.exp().mean().item()}")
        z = reparameterize(mu, logvar)
        recon_front, recon_mount, recon_pos, recon_vel, recon_joint = self.decoder(z)
        return recon_front, recon_mount, recon_pos, recon_vel, recon_joint, mu, logvar


def vae_loss(recon_img_front, img_front, recon_img_mount, img_mount, recon_pos, pos, recon_vel, vel, recon_joint, joint, mu, logvar):
    # Reconstruction Loss
    recon_loss = (
        F.mse_loss(recon_img_front, img_front, reduction='mean') +
        F.mse_loss(recon_img_mount, img_mount, reduction='mean') +
        F.mse_loss(recon_pos, pos, reduction='sum') +
        F.mse_loss(recon_vel, vel, reduction='sum') +
        F.mse_loss(recon_joint, joint, reduction='sum')
    )

    # KL Divergence Loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss_rescaled = kl_loss / latent_dim

    return recon_loss, kl_loss_rescaled




