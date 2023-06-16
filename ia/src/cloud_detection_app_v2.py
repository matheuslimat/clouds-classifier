import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Função para remover ruído
def remove_noise(img_path):
    img = cv2.imread(img_path)
    img_without_noise = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img_without_noise

# Carrega o modelo treinado
model = load_model('cloud_detection_model.h5')

# Diretório contendo as imagens de teste
test_dir = 'C://DEV//clouds-classifier//ia//src//data_test'

# Cria listas vazias para armazenar os nomes dos arquivos de cada classe
cloud_files = []
noncloud_files = []

# Percorre cada imagem do diretório de teste e faz a previsão
for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)

    # Carrega a imagem e remove o ruído
    img = remove_noise(img_path)

    img = cv2.resize(img, (128, 128))
    img = img.reshape((1,) + img.shape)
    prediction = model.predict(img)
    
    # Adiciona o nome do arquivo à lista correspondente à classe prevista
    if prediction[0][0] > 0.5:
        cloud_files.append(img_name)
    else:
        noncloud_files.append(img_name)

print("Arquivos com nuvens:")
fig = plt.figure(figsize=(10,10))
fig.suptitle('Imagens com nuvens', fontsize=16)
for i, f in enumerate(cloud_files, 1):
    print(f)
    img_path = os.path.join(test_dir, f)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = fig.add_subplot(2, 2, i)
    ax.imshow(img)
    ax.set_title(f)
plt.tight_layout()
plt.show()

print("Arquivos sem nuvens:")
fig = plt.figure(figsize=(10,10))
fig.suptitle('Imagens sem nuvens', fontsize=16)
for i, f in enumerate(noncloud_files, 1):
    print(f)
    img_path = os.path.join(test_dir, f)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = fig.add_subplot(2, 2, i)
    ax.imshow(img)
    ax.set_title(f)
plt.tight_layout()
plt.show()
