### Como usar a ferramenta?

Inicialmente, escolha o dataset o qual você quer trabalhar:

- **MNIST Digits:** Imagens de digitos de 0 a 9, de tamanho 28x28.

Quando o scatter plot aparecer, é possível ver os dados com os labels corretos. A projeção em 2D é resultado do algoritmo t-SNE. Para começar a utilizar a ferramenta, mude para a opção Manual Label e tente encontrar os clusters por você mesmo.


### Como o t-SNE funciona?

As imagens podem ser vistas como vetores longos de centenas ou milhares de dimensões, cada dimensão representando um tom de cor ou cinza (se a imagem for em preto e branco). Por exemplo, uma imagem 28x28 (como MNIST) pode ser desenrolada em um vetor 784-dimensional. Mantida dessa maneira, seria extremamente difícil visualizar nosso conjunto de dados, especialmente se ele contiver dezenas de milhares de amostras; de fato, a única maneira seria passar por todas as imagens e acompanhar como elas foram escritas.

O algoritmo t-SNE resolve esse problema reduzindo o número de dimensões de seus conjuntos de dados, para que você possa visualizá-lo no espaço de baixa dimensão, ou seja, em 2D ou 3D. Para cada ponto de dados, agora você terá uma posição em seu gráfico 3D, que pode ser comparada com outros pontos de dados para entender o quão próximos ou distantes estão um do outro.

Para o conjunto de dados do MNIST Digits, podemos ver que algumas vezes existem alguns dígitos agrupados, com alguns outliers provavelmente causados ​​por falta de letra. Quando isso acontecer, clique nas imagens para ver que tipo de problema pode ter; se nenhum, talvez tente alterar os parâmetros de entrada para encontrar um cluster mais apertado.

### Escolhendo os Parâmetros Corretos

1230/5000
A qualidade de uma visualização t-SNE depende muito dos parâmetros de entrada quando você treina o algoritmo. Cada parâmetro tem um grande impacto em quão bem cada grupo de dados será agrupado. Aqui está o que você deve saber para cada um deles:

- **Número de iterações:** Esta é a quantidade de etapas que você deseja executar o algoritmo. Um número maior de iterações geralmente oferece melhores visualizações, mas mais tempo para treinar.
- **Perplexidade:** Este é um valor que influencia o número de vizinhos que são levados em consideração durante o treinamento. De acordo com o [artigo original](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf), o valor deve estar entre 5 e 50.
- **Taxa de aprendizado:** Este valor determina quanto peso devemos atribuir às atualizações fornecidas pelo algoritmo em cada etapa. Geralmente, está entre 10 e 1000.
- **Dimensões iniciais do PCA:** Como o número de dimensões dos dados originais pode ser muito grande, usamos outra [técnica de redução de dimensionalidade chamada PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) para primeiro reduza o conjunto de dados para um espaço menor e aplique t-SNE nesse espaço. As dimensões iniciais do PCA de 50 mostraram bons resultados experimentais no artigo original.