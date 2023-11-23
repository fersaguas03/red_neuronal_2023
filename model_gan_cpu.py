# -*- coding: utf-8 -*-
"""model_gan

# Laboratorio N° 3 - GAN

"""

#Importar librerias
import torch
import torch.nn as nn
from torch.nn import init

#Función de Desprocesamiento de imagenes
def deprocess_img(x):
    return (x + 1.0) / 2.0

#Dimension del Ruido
NOISE_DIM = 96

#Función de Inicialización de Pesos
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

#Definir clase Generador
class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM, image_size=28, num_channels=1):
        super(Generator, self).__init__()
        self.fc = nn.Linear(noise_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=7, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, num_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.apply(initialize_weights)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 1024, 1, 1)
        x = nn.functional.relu(self.bn1(self.deconv1(x)))
        x = nn.functional.relu(self.bn2(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        return x

#Definir clase Discriminador
class Discriminator(nn.Module):
    def __init__(self, image_size=28, num_channels=1):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * (image_size // 4) * (image_size // 4), 1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = nn.functional.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#Funcón de Generación de Ruido
def sample_noise(batch_size, dim):
    """
    Genera un tensor de ruido aleatorio.

    batch_size: tamaño del lote
    dim: dimensión del tensor de ruido
    """
    return torch.randn(batch_size, dim)

