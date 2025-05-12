import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Função para carregar imagens de um diretório e redimensionar
def load_folder(folder, img_size, labels_dict=None, max_images=None, sort=False):
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
            img = Image.open(img_path).resize((img_size, img_size)).convert('RGB')
            img_array = np.array(img)
            images.append(img_array)
            count = count + 1
            #print(images)
            labels.append(directory)
        print('Qtd registros: ',count)
        print('-----------------------------------------------')
    return np.array(images), np.array(labels)

# Data set folder path: D:\rodri\Documents\OneDrive\Documentos\Cursos\Visual Computer Master\Trabalho Metodos Tradicionais\DataSet_HumanFaces
img_size = 48  # Redimensionar imagens para 48x48 pixels
train_folder = 'D:/rodri/Documents/OneDrive/Documentos/Cursos/Visual Computer Master/Trabalho Metodos Tradicionais/DataSet_HumanFaces/train'
test_folder  = 'D:/rodri/Documents/OneDrive/Documentos/Cursos/Visual Computer Master/Trabalho Metodos Tradicionais/DataSet_HumanFaces/test'

# Carregar imagens de treino
X_train, y_train = load_folder(train_folder, img_size)
print('Shape do Treino: ',X_train.shape)

# Carregar imagens de teste
X_test, y_test = load_folder(test_folder, img_size)
print('Shape do Teste: ',X_test.shape)

"""
# Função para detectar imagens corrompidas
def detect_corrupted_images(folder):
    corrupted_files = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            try:
                img = Image.open(os.path.join(folder, filename))
                img.verify()  # Verificar a integridade do arquivo
            except (IOError, SyntaxError) as e:
                corrupted_files.append(filename)
    return corrupted_files

# Detectar imagens corrompidas
corrupted_train_images = detect_corrupted_images(train_folder)
print(f"Imagens corrompidas no conjunto de treino: {corrupted_train_images}")

corrupted_submission_images = detect_corrupted_images(submission_folder)
print(f"Imagens corrompidas no conjunto de submissão: {corrupted_submission_images}")
"""

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
plt.show()

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

print('Apos o Flatten, shape do X_trian: ',X_train_flat.shape)
print('Apos o Flatten, shape do X_test: ',X_test_flat.shape)

from sklearn.decomposition import PCA

n_components = 5000
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

# Plotar variância acumulada
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))
plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Acumulada')
plt.title('Variância Acumulada Explicada pelos Componentes Principais')
plt.annotate("{:.2f}".format(cumulative_variance[-1]), xy=(n_components+1, cumulative_variance[-1]), color='red')
plt.grid(True)

from sklearn.linear_model import LogisticRegression
# Treinamento de um modelo Regressão Logistica
def train(X_train, y_train):
  model = LogisticRegression(max_iter=10000)
  model.fit(X_train, y_train)
  return model

model = train(X_train_pca, y_train)
X_train_pca.shape

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
import seaborn as sns

# Inferência e avaliação
def predict_and_evaluate(model, X_test, y_test):

    # Inferência
    y_pred = model.predict(X_test)

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

print('Resultados de Treino')
predict_and_evaluate(model, X_train_pca, y_train)
print('Resultados de Teste')
predict_and_evaluate(model, X_test_pca, y_test)



