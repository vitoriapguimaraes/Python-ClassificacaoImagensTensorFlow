# Classificação de Imagens com Redes Neurais Convolucionais e TensorFlow

> Este projeto desenvolve um modelo de Inteligência Artificial baseado em Redes Neurais Convolucionais (CNNs) para classificar imagens em 10 categorias distintas, utilizando o dataset CIFAR-10.

O objetivo é permitir que o modelo reconheça corretamente novas imagens pertencentes a categorias como avião, carro, gato, navio, entre outras, promovendo uma aplicação prática de aprendizado profundo.

![Tela do sistema](https://github.com/vitoriapguimaraes/Python-ClassificacaoImagensTensorFlow/blob/main/results/software.gif)

## Funcionalidades Principais:

1.  Carregamento e Pré-processamento de Dados: O dataset CIFAR-10 é carregado e normalizado para otimizar o treinamento do modelo.
2.  Visualização de Dados: Exibe imagens do conjunto de dados para análise inicial. [cite: 39, 40]
3.  Construção do Modelo: Implementação de uma CNN composta por camadas de convolução, pooling e densas.
4.  Treinamento e Avaliação: Treinamento com ajuste de hiperparâmetros e avaliação no conjunto de teste.
5.  Classificação de Novas Imagens: Permite a entrada de novas imagens para serem classificadas pelo modelo.

## Tecnologias Utilizadas:

-   Python
-   TensorFlow/Keras
-   Matplotlib
-   Pillow (PIL)
-   NumPy

## Como Executar:

1.  Clone o repositório.
2.  Abra o arquivo `main.ipynb` em um ambiente como Jupyter Notebook, Google Colab, ou outro de sua preferência.
3.  Certifique-se de rodar todas as células do notebook na ordem, para carregar os dados, treinar o modelo e visualizar os resultados.
4.  Você pode substituir ou adicionar novas imagens no código para realizar classificações e testar o modelo.

## Como Usar:

1.  Siga as etapas em "Como Executar" para rodar o modelo.
2.  O modelo será capaz de analisar uma imagem previamente desconhecida e classificá-la corretamente em uma das 10 categorias do CIFAR-10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Estrutura de Diretórios:
```
├── data/                   # Diretório com de entrada de imagens
├── scripts/
│   ├── app.ipynb           # Notebook principal com todas as etapas do projeto 
│   └── app.py              # Versão em Python
├── results/
└── README.md                 
```

## Status:

✅ Concluído

> Melhorias que podem ser incluídas:
> - Aprimoramento do Modelo: Experimentar diferentes arquiteturas, como ResNet ou VGG, para melhorar a acurácia.
> - Implementação de Inferência em Tempo Real: Criar uma aplicação web ou desktop para carregar imagens e exibir a classificação do modelo.

## Mais Sobre Mim:

Acesse os arquivos disponíveis na [Pasta Documentos](https://github.com/vitoriapguimaraes/vitoriapguimaraes/tree/main/DOCUMENTOS) para mais informações sobre minhas qualificações e certificações. [cite: 45]