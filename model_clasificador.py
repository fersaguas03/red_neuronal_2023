# -*- coding: utf-8 -*-
"""model_clasificador.ipynb

# Laboratorio N° 1 - Clasificador

"""
#Importar librerias de PyTorch
import torch
import torch.nn as nn

#Definir clase Clasificador
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Inicializar el clasificador
classifier = Classifier()