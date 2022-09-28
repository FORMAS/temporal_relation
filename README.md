# OpenTemporalRelation

![Badge em Desenvolvimento](http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge)


Esse repositório contém código fonte de meu trabalho de pesquisa ainda em andamento no Mestrado em Ciências da Computação pela Universidade Federal da Bahia, com título **Identificação de Relações Temporais em Português com Abordagem Baseada em Regras**.

Tem como objetivo principal desenvolver um método computacional para a tarefa de identificar de forma automática relações temporais entre pares das entidades evento/expressão temporal escritos em língua portuguesa.

Nossa abordagem propõe uma arquitetura de sistema baseada em regras, utilizamos como base inicial as regras desenvolvidas para o idioma inglês disponibilizadas por Jennifer[^jennifer], onde realizamos a tradução e adaptação para o português. 

Utilizamos dados do corpus TimebankPT[^timebankpt] que faz uso de um conjunto de 3 rótulos principais de relações temporais: `BEFORE`, `AFTER` e `OVERLAP`.

Neste trabalho, identificar relações temporais pressupõe a existência das entidades eventos e expressões temporais, assumindo que estas entidades são dadas.

Um evento é uma ocorrência específica envolvendo participantes, algo que acontece e pode frequentemente ser descrito como uma mudança de estado.

Já as expressões temporais representam vários fenômenos relacionados ao tempo, por exemplo, horários, datas e períodos.


## Esse Projeto Contém

**Python**
- `parse/ParseTimebankPT`: Código fonte das classes que tratam os dados do corpus e as relações temporais

**Jupter Notebook**
- `DocParseTimebankPT`: Documentação das classes
- `ResultadosArtigo01`: Definição das regras para extração de relações temporais, processamento e resultados dos experimentos
- `Demonstracoes`: Demonstra as principais funções das classes 

**Modelo Treinado**
- `dtree.model`: Modelo treinado utilizado para classificar de pares candidatos à predição com maior probabilidade de serem anotados no corpus

**Corpus**
- `TimebankPT`: Consiste em artigos anotados contendo referências temporais no idioma português[^timebankpt].


## Execusão Local

Copie todo código fonte para uma pasta local e instale os pré-requisitos abaixo em um ambiente onde o `Jupyter Notebook` está instalado.

### Pré-requisitos

- `spacy`
- `matplotlib`
- `seaborn`
- `pandas`
- `scikit_learn`
- `tabulate`
- `treelib`


### Instalar Pacotes

No terminal, dentro da pasta local execute o comando a seguir para instalar os pacotes requeridos.

```sh
$ pip install -r requirements.txt

```
Ainda no terminal, faça donwload do modelo de linguagem em português do spyCy:

```sh
$ python -m spacy download pt_core_news_lg

```

### Como Executar

Abra uns dos arquivos do `Jupyter Notebook` e execute o código.

## Autor

[<img src="https://avatars.githubusercontent.com/u/39890631?v=4" width=125><br><sub>Dárcio Santos Rocha</sub>](https://github.com/darciorocha)

    
## Referências

[^jennifer]: Jennifer D'Souza. (2015). Extracting Time and Space Relations from Natural Language Text. https://doi.org/10.13140/RG.2.2.20018.89288    
[^timebankpt]: Francisco Costa and António Branco. 2012. Timebankpt: A timeml annotated corpus of portuguese. In LREC, volume 12, pages 3727–3734.
