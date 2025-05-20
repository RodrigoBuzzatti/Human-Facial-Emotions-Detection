import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

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
X, y = load_folder(dataset_folder, img_size_width,img_size_hight)

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

"""
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)
"""

# Treinamento de um modelo XGBoost de Regressao Logistica
#https://scikit-learn.org/stable/modules/ensemble.html#ensemble
#GradientBoostingClassifier and GradientBoostingRegressor, might be preferred for small sample sizes since binning may lead to split points that are too approximate in this setting.
def train(X_train, y_train):
  model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
  model.fit(X_train, y_train)
  #model.score(X_train, y_train)
  return model

print("Iniciando o modelo XGBoost")
model = train(X_train_flat, y_train)
print(X_train_flat.shape)


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

print('Resultados de Treino')
y_pred_treino,probabilities_treino = predict_and_evaluate(model, X_train_flat, y_train)

print('Resultados de Teste')
y_pred_teste,probabilities_teste = predict_and_evaluate(model, X_test_flat, y_test)

# Analise dos Erros
# Filtrar previsões incorretas
incorrect_indices = np.where(y_pred_teste != y_test)[0]

# Se houver previsões incorretas, selecione até 10 para exibir
num_images_to_show = min(10, len(incorrect_indices))
if num_images_to_show > 0:
    plt.figure(figsize=(10, 4))
    for i in range(num_images_to_show):
        incorrect_index = incorrect_indices[i]
        incorrect_image = X_test[incorrect_index]
        true_label = y_test[incorrect_index]
        predicted_label = y_pred_teste[incorrect_index]

        plt.subplot(2, 5, i + 1)
        plt.imshow(incorrect_image, cmap='gray')
        plt.title(f'P:={predicted_label}, V:={true_label}', fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Nenhuma previsão incorreta encontrada.")

# Criar DataFrame com informações das previsões incorretas
incorrect_predictions = []

for i in incorrect_indices:
    true_label = y_test[i]
    predicted_label = y_pred_teste[i]
    row = {
        'indice': i,
        'true': true_label,
        'pred': predicted_label
    }
    # Adicionar as probabilidades para cada classe
    for class_index in range(7):
        row[f'proba_{class_index}'] = probabilities_teste[i, class_index]
    incorrect_predictions.append(row)

df_incorrect_predictions = pd.DataFrame(incorrect_predictions)

# Exibir o DataFrame
print(df_incorrect_predictions.head(10))


