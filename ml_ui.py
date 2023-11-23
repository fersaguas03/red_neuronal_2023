import streamlit as st
import requests
from PIL import Image
import numpy as np
import time
from streamlit_lottie import st_lottie

# Funcion para nuestra animacion
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_0yfsb3a1.json")



# Funcion para nuestra animacion
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_0yfsb3a1.json")
with st.container():
    st.subheader("Redes Neuronales 🧠")
    st.title("Trabajo Final Integrador")
    st.write("Hola! Mi nombre es Fernando Saguas :wave:👋")

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Objetivo")
        st.write(
            """
En este proyecto, nos hemos inspirado en los laboratorios N°1 y 3, que abordan la creación de un clasificador de imágenes y un generador de imágenes GAN, respectivamente.
Este código da vida a un modelo generativo diseñado para aprender y generar imágenes que guardan similitud con el conjunto de entrenamiento MNIST. A través de la entrada de 
un número por parte del usuario, el programa busca predecir y encontrar coincidencias con el número obtenido de la imagen generada, repitiendo el proceso hasta lograr el emparejamiento deseado.
Para aportar un toque distintivo a la experiencia, hemos incorporado las librerías LottieFiles y Emojipedia. Además, hemos implementado mensajes informativos, de advertencia y de éxito, 
que no solo brindan funcionalidad sino que también añaden un estilo único a la interfaz, ¡invitándote a disfrutar de un juego interactivo y creativo!.
            """
        )
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")
    with st.container():
        st.write("---")
    
    # Bandera para la validación
    numero_coincidio = False
  
    # Números ingresados por el usuario
    label_input = st.text_input("Ingrese un numero del 1 al 9:")
    
# Botón para generar imagen
    generate_button = st.button('Generar Imagen')
      
# Botón para vaciar campos y resetear la página
    reset_button = st.button('Vaciar campos')

# Espacio en blanco para mostrar el mensaje de espera
loading_message = st.empty()

# Espacio en blanco para mostrar la imagen
image_placeholder = st.empty()

# Inicializar lista para almacenar los dígitos coincidentes
matching_digits = []

# Espacio en blanco para mostrar el mensaje y la etiqueta predicha
message_display = st.empty()

# Si se presiona el botón de "Vaciar campos", resetear la página
if reset_button:
    st.experimental_rerun()

# Iterar sobre cada dígito ingresado
for digit in label_input:
    while not numero_coincidio and generate_button:
        # Mostrar mensaje de espera mientras se genera la imagen
        loading_message.info("¡Espere un momento mientras se genera la imagen....")

        # Hacer una solicitud a la API para obtener la imagen y la etiqueta predicha
        response = requests.get(f"http://127.0.0.1:8000/predict/{digit}")

        if response.status_code == 200:
            data = response.json()
            image_array = np.array(data["image"])
            predicted_label = data["predicted_label"]

            # Ocultar mensaje de espera
            loading_message.empty()

            # Mostrar la imagen generada y la etiqueta predicha
            image_display = Image.fromarray((image_array * 255).astype(np.uint8))
            image_placeholder.image(image_display, caption=f"Generando imagen para el numero ingresado: {digit}", use_column_width=True)

            # Mostrar mensaje de etiqueta predicha
            message_display.warning(f"¡El Número {digit} no ha coincidido con el {predicted_label} obtenido de la imagen!. Seguimos intentando...")

            # Verificar coincidencia y agregar a la lista si coincide
            if predicted_label == int(digit):
                matching_digits.append(digit)

                # Actualizar bandera y salir del bucle interno
                numero_coincidio = True
            else:
                # Pequeño retraso antes de mostrar la siguiente imagen
                time.sleep(1)
        else:
            st.error(f"Error al obtener la imagen: {response.text}")

# Mostrar mensaje de coincidencias encontradas
if matching_digits:
    message_display.warning(f"¡El Número {digit} no ha coincidido con el {predicted_label} obtenido de la imagen!. Seguimos intentando...").empty()
    st.success(f"Se encontró coincidencia para los dígitos ingresados: {' - '.join(matching_digits)}")
else:
    st.warning("No se encontraron coincidencias para los dígitos ingresados.").empty()