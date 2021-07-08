##############################################################################
#model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

kernel_size = 2 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 2 # Our images have two colours (amplitude and frequency)
latent_dim = 4 # latent dimension for sampling

# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels*image_channels, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels*image_channels, out_channels=init_channels*2*image_channels, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2*image_channels, out_channels=init_channels*4*image_channels, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4*image_channels, out_channels=64*image_channels, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(2*128*image_channels, 2*64*image_channels)
        self.fc_mu = nn.Linear(64*image_channels*2, latent_dim)
        self.fc_log_var = nn.Linear(64*image_channels*2, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64*image_channels*2*2)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64*image_channels, out_channels=init_channels*4*image_channels, kernel_size=kernel_size, 
            stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*4*image_channels, out_channels=init_channels*2*image_channels, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*2*image_channels, out_channels=init_channels*image_channels, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*image_channels, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=0
        )
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        #print(x.shape)
        x = F.relu(self.enc2(x))
        #print(x.shape)
        x = F.relu(self.enc3(x))
        #print(x.shape)
        x = F.relu(self.enc4(x))
        #print(x.shape)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 2).reshape(batch, -1)
        #print(x.shape)
        hidden = self.fc1(x)
        #print(hidden.shape)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        #print(z.shape)
        z = z.view(50, 128, 2, 2)
        
        #print(len(z))
        #print(z)
 
        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction, mu, log_var