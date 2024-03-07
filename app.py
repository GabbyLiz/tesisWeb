# Libreria para traduccion de texto
from googletrans import Translator
# Libreria para transformación de audio
from gtts import gTTS
# Librería para  mantener los datos como bytes en un búfer en memoria
from io import BytesIO
# Librería para convertir un objeto arbitrario Python en una serie de bytes
# Usa la función "load" para cargar los datos desde un archivo
from pickle import load
# Libreria para abrir imágenes en varios formatos
from PIL import Image
# Librería para desarrollar y evaluar modelos de aprendizaje profundo
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Librería para realizar cálculos matemáticos
import numpy as np
# Libreria para colocar imagen de fondo
import base64
# Libreria para desarrollar aplicaciones con Streamlit
import streamlit as st
# Librerías necesarias para descargar y cargar pesos desde Google Drive 
import gdown
import os 

translator = Translator()
# Enlace compartido de Google Drive al archivo HDF5 (reemplaza 'your_file_id')
enlace_google_drive = 'https://drive.google.com/uc?id=1xcvKb0Uqu5iVuNcKt_ry57J7GCm3TN3w'
# Descargar el archivo desde Google Drive
output_file_path = 'pesos.hdf5'
gdown.download(enlace_google_drive, output_file_path, quiet=False, confirm=True)
# Cargar el modelo con los pesos


# Definir el modelo CNN (inceptionv3, vgg16, resnet50)
def CNNModel(model_type):
    if model_type == 'inceptionv3':
        model = InceptionV3()
    elif model_type == 'vgg16':
        model = VGG16()
    elif model_type == 'resnet50':
        model = ResNet50()
    # Eliminar la última capa
    model.layers.pop()
    # Obtener la última capa después de eliminar
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def load_image(image_file):
    img = Image.open(image_file)
    return img


# Extraer características de la imagen
def extract_features(filename, model, model_type):
    if model_type == 'inceptionv3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        target_size = (299, 299)
    elif model_type == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        target_size = (224, 224)
    elif model_type == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        target_size = (224, 224)
    # Cargar y redimensionar la imagen
    image = load_image(filename)
    image = image.resize(target_size)

    # Convertir la imagen de pixeles a numpy array
    image = img_to_array(image)
    # Redimensionar datos para el modelo
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Preparar la imagen para el modelo CNN
    image = preprocess_input(image)
    # Pasar la imagen al modelo para obtener características codificadas
    features = model.predict(image, verbose=0)
    return features


def generate_caption_beam_search(model, tokenizer, image, max_len, beam_index=3):
    start_word = [[tokenizer.texts_to_sequences(['startseq'])[0], 0.0]]
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            # Secuencia las palabras más probables
            par_caps = pad_sequences([s[0]], maxlen=max_len)
            preds = model.predict([image, par_caps], verbose=0)
            # Tomar las mejores predicciones `beam_index` (es decir, las que tienen mayores probabilidades)
            # La función argsort ordena fácilmente los índices de una matriz dada
            word_preds = np.argsort(preds[0])[-beam_index:]

            # Crear una nueva lista para volver a pasarlos por el modelo
            for word in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(word)
                # Actualizar probabilidad
                prob += preds[0][word]
                #  Añadir como entrada para generar la siguiente palabra
                temp.append([next_cap, prob])

        start_word = temp
        # Ordenar según las probabilidades
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Tomar las palabras principales
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [word_for_id(i, tokenizer) for i in start_word]

    final_caption = []

    for word in intermediate_caption:
        if word == 'endseq':
            break
        else:
            final_caption.append(word)

    final_caption.append('endseq')
    return ' '.join(final_caption)


# Asignar un número entero a una palabra

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def image_caption(imagen, model_type, model_load_path, tokenizer_path1, max_length_model):
    # # Cargar el tokenizador
    tokenizer_path = tokenizer_path1
    tokenizer = load(open(tokenizer_path, 'rb'))

    # Longitud máxima de la secuencia (de entrenamiento)
    max_length = max_length_model

    # Cargar el modelo
    caption_model = load_model(model_load_path)

    image_model = CNNModel(model_type)
    # Codificar la imagen mediante el modelo CNN
    image = extract_features(imagen, image_model, model_type)
    # Generar los subtítulos mediante modelo RNN decodificador + búsqueda BEAM
    generated_caption = generate_caption_beam_search(caption_model, tokenizer, image, max_length, beam_index=3)

    # Generar los subtítulos mediante modelo RNN decodificador + Argmax
    # generated_caption = generate_caption(caption_model, tokenizer, image, max_length)
    # Quitar startseq y endseq
    caption = 'Image description: ' + generated_caption.split()[1].capitalize()
    for x in generated_caption.split()[2:len(generated_caption.split()) - 1]:
        caption = caption + ' ' + x
    caption += '.'

    return caption


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .css-9ycgxx {{
        display: none;
    }}
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # ---Estructura de la página---

    # Mostrar texto en formato de título (tema de tesis)
    st.title("Detección de objetos y semántica de una imagen")
    # Mostrar texto en formato de subtítulo
    st.header("_Al ingresar una imagen obtendrá una descripción general_")

    # Configurar la barra lateral para despliegue de modelos (4 en total)
    st.sidebar.title('Modelos')
    st.sidebar.subheader(
        'Los modelos pre-entrenados varían la calidad de detección, dependiendo de la configuración de cada uno en el entrenamiento de la red')

    app_mode = st.sidebar.selectbox('Seleccione uno de los siguientes modelos para la detección',
                                    ['Modelo 1'])
    if app_mode == 'Modelo 1':
        model_type = 'vgg16'
        max_length_model = 34
        model_load_path = 'pesos.hdf5'
        tokenizer_path1 = 'tokenizer.pkl'

    # Solicitar al usuario una imagen del tipo "jpg", "jpeg", "webp"
    uploaded_image = st.file_uploader("Seleccione una imagen de su dispositivo a continuación: ", type=["jpg", "jpeg", "webp"])

    if uploaded_image is not None:
        img = load_image(uploaded_image)
        # Mostrar una imagen
        st.image(img, caption='Imagen seleccionada', use_column_width=True)
        # Mostrar mensaje de éxito si la imagen se cargó correctamente
        st.success("¡Imagen Cargada con éxito!")
        st.info("Generando descripción y audio, por favor espere...")
        descripcion = image_caption(uploaded_image, model_type, model_load_path, tokenizer_path1, max_length_model)
        # Traducir texto de inglés a español
        translation = translator.translate(descripcion, dest='es')
        caption = translation.text
        # Escribir la descripción generadoa
        st.write(caption)


        language = 'es'
        text = str(caption)

        sound_file = BytesIO()
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.write_to_fp(sound_file)

        st.audio(sound_file)

    st.write("Ingresa en el [link](https://github.com/GabbyLiz/ic_web)" + " para ver el código completo")


add_bg_from_local('logo2.png')
