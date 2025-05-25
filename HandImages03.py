# Execucção do Modelo Random Forest

import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from skimage.feature import hog
import time as time

def format_time(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:02}"

# Função para carregar imagens de um diretório e redimensionar
def load_folder(folder, img_size_width,img_size_hight, labels_dict=None, max_images=None, sort=False):
    print("Lendo arquivos na pasta: ",folder)
    images = []
    labels = []
    file_list = os.listdir(folder)
    for directory in file_list:
        count = 0
        print(directory)
        well_formed_directory = folder+'/'+directory
        file_list_inside_folder = os.listdir(well_formed_directory)
        for file_name in file_list_inside_folder:
            img_path = os.path.join(well_formed_directory, file_name)
            img = Image.open(img_path).resize((img_size_width,img_size_hight)).convert('RGB')
            #img = ImageOps.grayscale(img) # Usei imagens em Tons de Cinza mas a performance caiu
            img_array = np.array(img)
            images.append(img_array)
            count = count + 1
            #print(images)
            labels.append(directory)
        print('Qtd registros: ',count)
        print('-----------------------------------------------')
    return np.array(images), np.array(labels)

# Data set folder path: D:\rodri\Documents\OneDrive\Documentos\Cursos\Visual Computer Master\Trabalho Metodos Tradicionais\DataSet_HumanFaces
#                       D:\rodri\Documents\OneDrive\Documentos\Cursos\Visual Computer Master\Trabalho Metodos Tradicionais\DataSet_HandImages
img_size_width = 348  # Redimensionar imagens para width x hight
img_size_hight = 464
dataset_folder = 'D:/rodri/Documents/OneDrive/Documentos/Cursos/Visual Computer Master/Trabalho Metodos Tradicionais/DataSet_HandImages'

# Carregar imagens
#X_train, y_train = load_folder(dataset_folder, img_size_width,img_size_hight)
start = time.time()
print("Iniciando a carga das Imagens")
X, y = load_folder(dataset_folder, img_size_width,img_size_hight)
t = time.time() - start
print("Tempo total para carregar as Imagens do Dataset: ",format_time(t))

#Separando o dataset entre treino e teste
#https://scikit-learn.org/0.19/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print('Shape do Treino: ',X_train.shape)

import random
plt.figure(figsize=(10, 3))
for i in range(10):
  rnd = random.randint(0, len(X_train))
  image = X_train[rnd]
  true_label = y_train[rnd]

  plt.subplot(2, 5, i + 1)
  plt.imshow(image, cmap='gray')
  plt.title(f'{true_label}', fontsize=10)
  plt.axis('off')
plt.tight_layout()
#plt.show()

# Obter valores mínimos e máximos dos pixels
pixel_min = np.min(X_train)
pixel_max = np.max(X_train)

# Obter dimensões das imagens
image_heights = X_train.shape[1] * np.ones(X_train.shape[0])
image_widths = X_train.shape[2] * np.ones(X_train.shape[0])

# Plotar valores mínimos e máximos dos pixels
plt.figure(figsize=(12, 3))

plt.subplot(1, 2, 1)
plt.hist(X_train.flatten(), bins=50, color='blue', alpha=0.7)
plt.title(f'Valores de Pixels\nMínimo: {pixel_min}, Máximo: {pixel_max}')
plt.xlabel('Valor do Pixel')
plt.ylabel('Frequência')

# Plotar gráfico de dispersão de largura vs altura
plt.subplot(1, 2, 2)
plt.scatter(image_widths, image_heights, alpha=0.5)
plt.title(f'Largura vs Altura das Imagens\nAltura: Mínima: {image_heights.min()}, Máxima: {image_heights.max()}\nLargura: Mínima: {image_widths.min()}, Máxima: {image_widths.max()}')
plt.xlabel('Largura (pixels)')
plt.ylabel('Altura (pixels)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Normalizar os dados
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Verificar formas dos dados carregados
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

train_unique_values, train_counts = np.unique(y_train, return_counts=True)
print("Valores únicos:", train_unique_values)
print("Contagens:", train_counts)

X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

print('Apos o Flatten, shape do X_train: ',X_train_flat.shape)
print('Apos o Flatten, shape do X_test: ',X_test_flat.shape)

from sklearn.ensemble import RandomForestClassifier

# Treinamento de um modelo Random Forest
def train(X_train, y_train, n_estimators=100):
  model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split= 3, verbose = 1, random_state = True)
  model.fit(X_train, y_train.flatten())
  return model

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
import seaborn as sns

# Inferência e avaliação
def predict_and_evaluate(model, X_test, y_test):
    # Inferência
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Métricas
    print('Acurácia:', accuracy_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred, average='weighted'))

    # Matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()
    return y_pred,probabilities

# Extração de características utilizando HOG
def extract_hog_features(images):
    hog_features = []
    hog_images = []
    for image in images:
        features, img = hog(image, orientations=9,      # o número de direções angulares distintas para as quais os gradientes são calculados
                              pixels_per_cell=(8, 8),   # tamanho (em pixels) de cada célula na qual a imagem é dividida para calcular o histograma de gradientes orientados
                              cells_per_block=(2, 2),   # número de células em cada bloco. Os blocos são usados para normalizar os histogramas de gradientes dentro de células, melhorando a robustez a variações de iluminação e contraste.
                              visualize=True,           # se a imagem HOG (uma representação visual das características HOG) deve ser retornada junto com o vetor de características.
                              channel_axis=-1)          # índice do eixo do canal na imagem de entrada.
        hog_features.append(features)
        hog_images.append(img)
    return np.array(hog_features), hog_images

# Extraindo características HOG dos dados de treinamento e teste
start = time.time()
print("Iniciando HOG")
X_train_hog, train_hog_images = extract_hog_features(X_train)
X_test_hog, _ = extract_hog_features(X_test)
t = time.time() - start
print("Tempo total para extrair HOG do Dataset: ",format_time(t))

# Visualizar imagens HOG
def visualize_multiple_hog(images, hog_images, images_to_show=5):
    plt.figure(figsize=(6, images_to_show))

    for i in range(images_to_show):
        # Mostrar imagem original
        plt.subplot(images_to_show, 2, 2 * i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Imagem Original {i+1}', fontsize=8)
        plt.axis('off')

        # Mostrar imagem HOG
        plt.subplot(images_to_show, 2, 2 * i + 2)
        plt.imshow(hog_images[i], cmap='gray')
        plt.title(f'Imagem HOG {i+1}', fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

images_to_show = 5
visualize_multiple_hog(X_train, train_hog_images, images_to_show=images_to_show)
print(X_train_hog.shape)

model = train(X_train_hog, y_train)

print('Resultados de Teste com HOG')
predict_and_evaluate(model, X_test_hog, y_test)
print('Resultados de Treino com HOG')
predict_and_evaluate(model, X_train_hog, y_train)

# Parâmetros a serem testados para reduzir overfitting!
tuned_parameters = [{'n_estimators': [50, 100, 200],
                     'min_samples_leaf': [3, 5, 7]}]

# Executar o grid search
from sklearn.model_selection import GridSearchCV
model = GridSearchCV(RandomForestClassifier(n_jobs=-1), tuned_parameters, scoring='f1_macro')
model.fit(X_train_hog, y_train)
print(model.best_params_)

print('Resultados de Teste')
predict_and_evaluate(model, X_test_hog, y_test)
print('Resultados de Treino')
predict_and_evaluate(model, X_train_hog, y_train)