# Projeto de Detecção de Emoção Facial de Humanos

Este projeto utiliza técnicas de aprendizado de máquina.
Foram utilizados dois datasets: Uma para detecção de emoção facial de Humanos e outro para deteçção de imagens de mãos
Ele faz parte do curso Computer Vision Master e foi solicitado pela professora orientadora Manoela Kohler o uso de técnicas tradicionais (sem o uso de técnias de redes neurais) 
Como eu achei o dataset de deteção emoção facial confuso pois tem várias imagens que eu tive dificuldade de intepretar, resolvi também trabalhar com outro dataset"


## Descrição do Pipeline
Carregamento das Imagens: As imagens são carregadas a partir de diretórios especificados, contendo subpastas que já identifica suas categorias.

Extração de Atributos: Utilizei os proprios pixels como atributos, fazendo um flaten, mas como a performance foi ruim eu também apliquei Histogram of Oriented Gradients (HOG).

Redução de Dimensionalidade: Após a extração dos atributos, é aplicado o PCA para reduzir a dimensionalidade dos dados, mantendo entre 98% e 99% da variância explicada.

Treinamento dos Modelos: Três modelos são treinados para efeitos de comparação

**Logistic Regression**: Modelo supervisionado para classificação das imagens. 

**Ramdom Forest**: Modelo supervisionado para classificação das imagens.

**XGBoost**: Modelo supervisionado para classificação das imagens.

Inferência e Avaliação: As inferências são realizadas nos dados de teste, e os resultados são avaliados e visualizados através de matrizes de confusão e relatórios de classificação.

## Resultados

Os resultados das inferências são exibidos diretamente no console e através de gráficos gerados pelo script.

## Estrutura do Projeto

- `main.py`: Script para analisar o dataset de emoção facial.
- 'HandImages.py': Script para analisar o dataset de posição das mãos usando Logistic Regression
- - 'HandImages01.py': Idem acima,mas usando o modelo RandomForest
- - 'HandImages02.py': Idem acima, mas usando HOG em vez de pixels
- - 'XGBoost.py': Idem acima, mas usando o modelo XGBoost
- `.gitignore`: Arquivo que define quais arquivos ou pastas devem ser ignorados pelo Git.

## Dependências
Utilizer o PyCharm com um ambiente virtual usando o Pythom versão 3.13

## Dataset

Foram usados dois datasets:
- Facial Expression Recognition: https://www.kaggle.com/datasets/msambare/fer2013
- Hands Images: https://huggingface.co/datasets/trashsock/hands-images

## Resultados

Os resultados estão apresentados em uma planilha Excel. Em todas as iterações encontrei caracteristica de overfit e mesmo fazendo tunning dos hiperparametros nao consegui evitar o Overfit
