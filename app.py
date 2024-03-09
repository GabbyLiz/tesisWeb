import gdown
import requests
import os
# libreria para traduccion de texto
from googletrans import Translator
# libreria para transformación de audio
from gtts import gTTS
from io import BytesIO
from pickle import load
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
# Libreria para colocar imagen de fondo
import base64
# Libreria para desarrollar aplicaciones con Streamlit
import streamlit as st
translator = Translator()
# Definir 'descripcion' fuera del bloque with
modelo_cargado = None
imagen_cargada = None
descripcion = None
translation = None
caption = None
text = None
sound_file = None
model_type = None
max_length_model = None
model_load_path = None
tokenizer_path = None

# Enlace compartido de Google Drive al archivo HDF5 (reemplaza 'your_file_id')
enlace_google_drive1 = 'https://drive.google.com/uc?id=1FMoVJX2X-pgYV7noOXny6Xnq5lPjJetb'
enlace_google_drive2 = 'https://drive.google.com/uc?id=1ZVPsRsKW0Lsd5W0LPG9q-K1yGcPQ069Q'

# Variable de sesión para realizar un seguimiento del estado de la descarga
descarga_realizada = st.session_state.get('descarga_realizada', False)

# Crear una sección para cargar una imagen de fondo
background_image = 'logo2.png'  # Ruta de la imagen de fondo

# Función para cargar la imagen de fondo
with open('logo2.png', "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
st.markdown(
    f"""
    <style>
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

def load_vgg16_weights(local_path):
    model = VGG16(weights=local_path, include_top=True)
    return model

# Define the CNN model
def CNNModel(model_type):
    if model_type == 'inceptionv3':
        model = InceptionV3()
    elif model_type == 'vgg16':
        model = load_vgg16_weights('vgg16.h5')
    elif model_type == 'resnet50':
        model = ResNet50()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    return model
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Extraer características de la imagen
def extract_features(filename, model, model_type):
    if model_type == 'inceptionv3':
        from keras.applications.inception_v3 import preprocess_input
        target_size = (299, 299)
    elif model_type == 'resnet50':
        from keras.applications.resnet50 import preprocess_input
        target_size = (224, 224)
    elif model_type == 'vgg16':
        from keras.applications.vgg16 import preprocess_input
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
    # start_word --> [[idx,prob]] ;prob=0 initially
    start_word = [[tokenizer.texts_to_sequences(['startseq'])[0], 0.0]]
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_len)
            preds = model.predict([image,par_caps], verbose=0)
            # Tomar las mejores predicciones `beam_index` (es decir, las que tienen mayores probabilidades)
            word_preds = np.argsort(preds[0])[-beam_index:]

            # Crear una nueva lista para volver a pasarlos por el modelo
            for word in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(word)
                # Actualizar probabilidad
                prob += preds[0][word]
                #  Añadir como entrada para generar la siguiente palabra
                temp .append([next_cap, prob])

        start_word = temp
        # Ordenar según las probabilidades
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Tomar las palabras principales
        start_word = start_word[-beam_index:]


    start_word = start_word[-1][0]
    intermediate_caption = [int_to_word(i,tokenizer) for i in start_word]

    final_caption = []

    for word in intermediate_caption:
        if word=='endseq':
            break
        else:
            final_caption.append(word)

    final_caption.append('endseq')
    return ' '.join(final_caption)

# Asigna un número entero a una palabra
def int_to_word(integer, tokenizer):
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

st.title("Detección de objetos y semántica de una imagen")

# Obtener el tamaño original del archivo en bytes
with st.spinner('Obteniendo información del archivo...'):
    response = requests.head(enlace_google_drive1)
    file_size_original = int(response.headers['Content-Length'])


# Crear una sección para cargar una imagen
st.header('_Ingrese una imagen para realizar la descripción:_')


# Crear una barra lateral para las opciones de descarga
st.sidebar.subheader('Modelos')
opcion_descarga = st.sidebar.selectbox('Seleccione una opción para escoger el modelo que desea probar:',
                                       ['Seleccione', 'Modelo 1', 'Modelo 2', 'Modelo 3','Modelo 4'])

# Añadir un botón para cargar la imagen
imagen_cargada = st.file_uploader('Selecciona una imagen', type=['jpg', 'jpeg', 'png'])

# Tratamiento adicional para cada opción de descarga
if opcion_descarga == 'Modelo 1':
    # Tu código específico para la opción 1 aquí
    model_type = 'vgg16'
    max_length_model = 34
    model_load_path = 'pesos_modelo 1.hdf5'
    tokenizer_path = 'tokenizer.pkl'
    st.write('Realizar acciones adicionales para la Opción 1')

elif opcion_descarga == 'Modelo 2':
    # Tu código específico para la opción 2 aquí
    model_type = None
    max_length_model = ''
    model_load_path = ''
    tokenizer_path = ''
    st.write('Realizar acciones adicionales para la Opción 2')

elif opcion_descarga == 'Modelo 3':
    # Tu código específico para la opción 3 aquí
    model_type = None
    max_length_model = ''
    model_load_path = ''
    tokenizer_path = ''
    st.write('Realizar acciones adicionales para la Opción 3')

elif opcion_descarga == 'Modelo 4':
    # Tu código específico para la opción 4 aquí
    model_type = None
    max_length_model = ''
    model_load_path = ''
    tokenizer_path = ''
    st.write('Realizar acciones adicionales para la Opción 4')

# Mostrar la imagen cargada si existe
if imagen_cargada is not None:
    st.image(imagen_cargada, caption='Imagen cargada', use_column_width=True)
    st.success("¡Imagen subida con éxito!")
    # Verifica que todas las variables tengan valor asignado para ejecutar el código
    if model_type is not None and model_load_path is not None and tokenizer_path is not None and max_length_model is not None:
        # Tu código a ejecutar si todas las variables no son None
        with st.spinner('Obteniendo información del archivo...'):
            descripcion = image_caption(imagen_cargada, model_type, model_load_path, tokenizer_path, max_length_model)
            translation = translator.translate(descripcion, dest='es')
            caption = translation.text
            st.write(caption)
            language = 'es'
            text = str(caption)
            sound_file = BytesIO()
            myobj = gTTS(text=text, lang=language, slow=False)
            myobj.write_to_fp(sound_file)
            st.audio(sound_file)

    else:
        st.error("Faltan datos para generar la descripción")

    # if model_type is None:
    #     st.info("No hay modelo cargado")
    # else:
    #     st.info("Si hay modelo cargado")

# Botón para iniciar la descarga solo si se selecciona una opción válida y la descarga aún no se ha realizado
if opcion_descarga != 'Seleccione' and not descarga_realizada:

    if st.sidebar.button(f'Descargar Pesos para {opcion_descarga} desde Google Drive'):
        with st.spinner('Descargando los pesos...'):
            # Descargar el archivo desde Google Drive
            output_file_path1 = f'pesos_{opcion_descarga.lower()}.hdf5'
            output_file_path2 = 'vgg16.h5'
            gdown.download(enlace_google_drive1, output_file_path1, quiet=False)
            gdown.download(enlace_google_drive2, output_file_path2, quiet=False)

            # Obtener el tamaño del archivo después de la descarga
            file_size_downloaded = os.path.getsize(output_file_path1)
            file_size_downloaded2 = os.path.getsize(output_file_path2)

            st.success(f'Descarga completa para {opcion_descarga}. Puedes cargar los pesos ahora.')
            #st.info(f'Tamaño original de los archivos: {file_size_original / (1024 ** 2):.2f} MB')
            st.info(f'Tamaño del archivo descargado: {file_size_downloaded / (1024 ** 2):.2f} MB')
            st.info(f'Tamaño del archivo descargado: {file_size_downloaded2 / (1024 ** 2):.2f} MB')


            # Actualizar la variable de sesión para indicar que la descarga se ha realizado
            st.session_state.descarga_realizada = True

# Cargar el modelo con los pesos descargados solo si la descarga se ha realizado
if descarga_realizada and st.sidebar.button('Cargar Modelo con Pesos'):
    try:
        # Cargar el modelo con los pesos
        modelo_cargado = load_model(f'pesos_{opcion_descarga.lower()}.hdf5')
        st.success('El modelo ya fue descargado exitosamente.')

        # Realizar acciones adicionales con el modelo cargado si es necesario
        # ...
    except Exception as e:
        st.error(f'Error al cargar el modelo: {e}')

# Liberar memoria manualmente
del modelo_cargado
del imagen_cargada
del descripcion
del translation
del caption
del text
del sound_file
