#testing script for the basic elements to training a convolutional variational 
#autoencoder for the time series masks.
#the architecture is based on that for the MNIST by debuggercafe:
#https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

    
##############################################################################
#train.py

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

clear_all()

import torch
import torch.optim as optim
import torch.nn as nn
import model
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import matplotlib
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from engine import train, validate
import matplotlib.pyplot as plt

from utility import save_reconstructed_images, image_to_vid, save_loss_plot
matplotlib.style.use('ggplot')
plt.rcParams.update({'font.size': 25})

def normalize(x):
    norm = np.linalg.norm(x)
    y = x/norm
    return y

def npy_loader(path):
    return torch.from_numpy(normalize(np.load(path))).float()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the model
model = model.ConvVAE().to(device)
# set the learning parameters
lr = 0.001
epochs = 100
batch_size = 50
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='mean')
# a list to save all the reconstructed images in PyTorch grid format
grid_images = []
presetN = 1

transform = transforms.Compose([
    transforms.Resize((24, 24)),
    #transforms.ToTensor(),
])

trainset = datasets.DatasetFolder(
    root=f'C:/Users/ljq-2/Documents/Spyder/SoundDiagnosisProject/SignalMasks/id_00forNN/Preset{presetN}/Training',
    loader=npy_loader,
    extensions='.npy',
    transform = transform
)
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)

testset = datasets.DatasetFolder(
    root=f'C:/Users/ljq-2/Documents/Spyder/SoundDiagnosisProject/SignalMasks/id_00forNN/Preset{presetN}/Validation',
    loader=npy_loader,
    extensions='.npy',
    transform = transform
)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False
)

# # training set and train data loader
# trainset = torchvision.datasets.MNIST(
#     root='../input', train=True, download=True, transform=transform
# )
# trainloader = DataLoader(
#     trainset, batch_size=batch_size, shuffle=True
# )
# # validation set and validation data loader
# testset = torchvision.datasets.MNIST(
#     root='../input', train=False, download=True, transform=transform
# )
# testloader = DataLoader(
#     testset, batch_size=batch_size, shuffle=False
# )

train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, trainloader, trainset, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images = validate(
        model, testloader, testset, device, criterion
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    ## save the reconstructed images from the validation loop
    #save_reconstructed_images(recon_images, epoch+1,presetN)
    ## convert the reconstructed images to PyTorch image grid format
    #image_grid = make_grid(recon_images.detach().cpu())
    #grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")
    
# save the reconstructions as a .gif file
#image_to_vid(grid_images)
# save the loss plots to disk
#save_loss_plot(train_loss, valid_loss,presetN)
plt.figure(figsize=(20, 14))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.legend(fontsize = 20)
plt.show()

print('TRAINING COMPLETE')