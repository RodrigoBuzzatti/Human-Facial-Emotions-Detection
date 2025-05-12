# Projeto de Detecção de Emoção Facial de Humanos

Este projeto utiliza técnicas de aprendizado de máquina para detecção de emoção facial de Humanos.
Ele faz parte do curso Computer Vision Master e foi solicitado pela professora orientadora Manoela Kohler o uso de técnicas tradicionais (sem o uso de técnias de redes neurais) 

## Descrição do Pipeline
Carregamento das Imagens: As imagens são carregadas a partir de diretórios especificados, contendo subpastas para imagens com defeito (def_front) e imagens sem defeito (ok_front).

Extração de Atributos: Dependendo da configuração, a extração de atributos pode ser feita simplesmente fazendo um flatten dos pixeis ou utilizando técnicas clássicas como Local Binary Patterns (LBP) e Histogram of Oriented Gradients (HOG), ou utilizando uma CNN pré-treinada como o VGG16.

Redução de Dimensionalidade: Após a extração dos atributos, é aplicado o PCA para reduzir a dimensionalidade dos dados, mantendo 99% da variância explicada.

Treinamento dos Modelos: Três modelos são treinados para efeitos de comparação

**Isolation Forest**: Para detecção de anomalias.

**One-Class SVM**: Para detecção de anomalias.

**Random Forest**: Modelo supervisionado para classificação das imagens.

Inferência e Avaliação: As inferências são realizadas nos dados de teste, e os resultados são avaliados e visualizados através de matrizes de confusão e relatórios de classificação.

## Resultados

Os resultados das inferências são exibidos diretamente no console e através de gráficos gerados pelo script `anomaly_detection.py`. As matrizes de confusão permitem avaliar o desempenho de cada modelo, indicando a proporção de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos.

## Estrutura do Projeto

- `anomaly_detection.py`: Script principal para carregar as imagens, extrair atributos, treinar os modelos e realizar as inferências.
- `utils.py`: Funções utilitárias para carga de imagens, extração de atributos e visualização de resultados.
- `requirements.txt`: Lista de dependências necessárias para executar o projeto.
- `.gitignore`: Arquivo que define quais arquivos ou pastas devem ser ignorados pelo Git.

## Dependências

Para instalar as dependências do projeto, utilize o seguinte comando:

```bash
pip install -r requirements.txt
````

## Dataset

O dataset foi baixado do kaggle [aqui](https://www.kaggle.com/datasets/msambare/fer2013/data).
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.
The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
The training set consists of 28,709 examples and the public test set consists of 3,589 examples.
