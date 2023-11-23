#API
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from model_gan_cpu import Generator, deprocess_img, sample_noise
from model_clasificador import Classifier
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

app = FastAPI()

# Crear el generador
G = Generator()

# Ruta al archivo de pesos del generador
weights_path = 'C:\\Users\\SEBA\\Desktop\\Redes Neuronales\\proyecto gan\\generador_pesos.pth'

# Verificar si los pesos del generador ya se cargaron
if not hasattr(G, 'initialized') or not G.initialized:
    try:
        # Cargar los pesos previamente guardados del generador
        G.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        G.eval()  # Asegurarse de que el modelo esté en modo de evaluación
        G.initialized = True
        print(f'Pesos del generador cargados exitosamente desde: {weights_path}')
    except FileNotFoundError:
        print(f'ERROR: No se encontró el archivo de pesos del generador en la ruta: {weights_path}')
    except Exception as e:
        print(f'ERROR: Ocurrió un error al cargar los pesos del generador: {str(e)}')

# Inicializar el clasificador
classifier = Classifier()

# Cargar los pesos entrenados del clasificador
classifier.load_state_dict(torch.load("C:\\Users\\SEBA\\Desktop\\Redes Neuronales\\proyecto gan\\mnist_classifier.pth", map_location=torch.device('cpu')))
classifier.eval()  # Modo de evaluación

# Ruta para generar imagen y predecir con el clasificador
@app.get("/predict/{label}")
async def predict(label: int):
    # Generar una imagen con el generador
    with torch.no_grad():
        random_noise = sample_noise(1, 96)
        generated_image = G(random_noise)

    # Utilizar el clasificador en la imagen generada
    with torch.no_grad():
        output = classifier(generated_image)

    # Obtener la etiqueta predicha
    predicted_label = torch.argmax(output).item()

    generated_image_numpy = deprocess_img(generated_image.squeeze().cpu().numpy()) 

    return JSONResponse(content={"image": generated_image_numpy.tolist(), "label": label, "predicted_label": predicted_label})

