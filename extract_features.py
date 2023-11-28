import os
import numpy as np
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# Caminho para o seu conjunto de dados
caminho_dataset = 'simpsons_dataset'

# Configuração do modelo VGG16
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input,
              outputs=base_model.get_layer('fc2').output)

# Função para extrair características de uma imagem


def extrair_caracteristicas(img_path):
    # Certifique-se de ajustar o tamanho conforme necessário
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)
    return features.flatten()


# Lista para armazenar características e rótulos
features_list = []
labels_list = []

# Iterar sobre as pastas de personagens
for personagem in os.listdir(caminho_dataset):
    personagem_path = os.path.join(caminho_dataset, personagem)

    # Certificar-se de que é um diretório
    if os.path.isdir(personagem_path):
        # Iterar sobre as imagens do personagem
        for imagem_nome in os.listdir(personagem_path):
            imagem_path = os.path.join(personagem_path, imagem_nome)
            # Extrair características e adicionar à lista
            features = extrair_caracteristicas(imagem_path)
            features_list.append(features)
            labels_list.append(personagem)

# Converter listas em arrays numpy
features_array = np.array(features_list)
labels_array = np.array(labels_list)

# Salvar características e rótulos em um arquivo de texto
np.savetxt('Features/v2/caracteristicas.txt', features_array)
np.savetxt('Features/v2/rotulos.txt', labels_array, fmt='%s')
