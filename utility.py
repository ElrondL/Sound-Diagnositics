##############################################################################
#utility.py

import imageio
import numpy as np
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
to_pil_image = transforms.ToPILImage()
def image_to_vid(images):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('../outputs/generated_images.gif', imgs)
def save_reconstructed_images(recon_images, epoch,preset):
    torch.save(recon_images.cpu(), f"C:/Users/ljq-2/Documents/Spyder/SoundDiagnosisProject/theNN/outputs/Preset{preset}/output{epoch}.jpg")
def save_loss_plot(train_loss, valid_loss,preset):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"C:/Users/ljq-2/Documents/Spyder/SoundDiagnosisProject/theNN/outputs/Preset{preset}/loss.jpg")
    plt.show()