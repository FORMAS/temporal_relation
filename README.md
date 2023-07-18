# Identificação de Tipos de Relações Temporais *Event-Time* em Português: Uma Abordagem Baseada em Regras com Classificação Associativa

![Badge em Desenvolvimento](https://img.shields.io/static/v1?label=STATUS&message=FINISHED&color=blue&style=for-the-badge)


Nesta dissertação, apresentamos um método computacional para identificar tipos de relações temporais entre eventos e expressões temporais em textos em português. Utilizando uma abordagem baseada em regras e conjuntos de <i>features</i> relevantes, desenvolvemos conjuntos de regras utilizando algoritmos de aprendizagem de regras, além de regras manuais específicas para o idioma. Os experimentos no <i>corpus</i> TimeBankPT[^timebankpt] demonstraram a eficácia do nosso método, superando o <i>baseline</i>[^baseline] em termos de acurácia e <i>F1-score</i>. Esta pesquisa apresenta aplicações práticas no campo do resumo de documentos, compreensão de histórias e análise de notícias. Por meio do uso de regras explicáveis, possibilita uma compreensão aprimorada do tempo em textos.

# Conjunto de <i>Features</i>
Apresentamos nosso [Conjunto de *Features*](conjunto_de_features.md) organizadas por tipo de informações linguísticas.


# Conjuntos de Regras
Apresentamos nossos [Conjuntos de Regras](rules) desenvolvidos para a tarefa de identificar tipos de relações temporais entre evento e expressão temporal em textos em português.

Cada regra possui o seguinte formato:

> *[código da regra, 'TIPO DA RELAÇÃO TEMPORAL', ordem, "expressão lógica que representa a regra", 'algoritmo de origem', acurácia, total de acertos, número de vezes que foi acionada]*

A expressão lógica que representa a regra é composta por conjunções de condições. Cada condição é constituída por uma *`feature`* de nosso [conjunto de *features*](conjunto_de_features.md), um `operador`, que pode ser de igualdade ou desigualdade, e o `valor` da *feature*.


# Resultados Finais

Apresentamos os [resultados finais](resultado_final_setrules.ipynb) dos experimentos realizados com os conjuntos de regras, utilizando dados de treinamento e teste. Analisamos duas abordagens diferentes para aplicação das regras: a primeira regra acionada e o sistema de votação. Além disso, fornecemos as [significâncias estatísticas](estatistica_resultados_experimentos.ipynb) dos resultados.

<br>

# Esse Projeto Contém Também:

**Código Fonte**
- [`parse/ParseTimebankPT`](parse): Código fonte das classes que tratam os dados do *corpus* e identificam tipos de relações temporais.

**Exemplo de Uso das Classes**
- [`demonstracao_sistema`](demonstracao_sistema.ipynb): Demonstração de uso das principais funções do código fonte.

**Corpus**
- [`TimebankPT`](TimeBankPT): Esse *corpus* consiste em artigos de notícias anotados contendo referências temporais em lingua portuguesa[^timebankpt].


## Execução Local

Copie todo código fonte para uma pasta local e instale os pré-requisitos abaixo em um ambiente onde o `Jupyter Notebook` está instalado.

### Pré-requisitos

- `spacy`
- `matplotlib`
- `seaborn`
- `pandas`
- `numpy`
- `scikit_learn`
- `tabulate`
- `treelib`



### Instalar Pacotes

Dentro da pasta local execute comando a seguir para instalar os pacotes requeridos.

```sh
$ pip install -r requirements.txt

```


# Autor

[<img src="https://avatars.githubusercontent.com/u/39890631?v=4" width=125><br><sub>Dárcio Santos Rocha</sub>](https://github.com/darciorocha)

    
# Referências

[^timebankpt]: COSTA, Francisco; BRANCO, António. TimeBankPT: A TimeML Annotated Corpus of Portuguese. In: LREC. 2012. p. 3727-3734.
[^baseline]: COSTA, Francisco Nuno Quintiliano Mendonça Carapeto. Processing Temporal Information in Unstructured Documents. 2012. Tese de Doutorado. Universidade de Lisboa (Portugal).
