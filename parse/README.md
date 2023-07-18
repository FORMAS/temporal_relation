## DOCUMENTAÇÃO DAS CLASSES
IDENTIFICAÇÃO DE TIPOS DE RELAÇÕES TEMPORAIS


```python
from parse.ParseTimebankPT import TimebankPT

path_tml = r'TimeBankPT\train\ABC*.tml'
tb = TimebankPT(path_tml)
```

    Arquivo 'dataset/corpus.pickle' encontrado. Acionado carregamento rápido do corpus.
    Arquivo 'dataset/data_pipe_tb.pickle' encontrado. 
    Acionado carregamento rápido dos dados do TimebankPT para pipeline do spaCy.
    SEQUÊNCIA PIPELINE: PORTUGUÊS
       1 -> tok2vec
       2 -> morphologizer
       3 -> parser
       4 -> lemmatizer
       5 -> attribute_ruler
       6 -> ner
       7 -> pipe_timebankpt
       8 -> merge_entities
    
    
    

## TimebankPT


```python
help(tb)
```

    Help on TimebankPT in module parse.ParseTimebankPT object:
    
    class TimebankPT(Functions)
     |  TimebankPT(path_tml, add_timebank=True, lang='pt', dev: bool = False, ignore_load_corpus=False)
     |  
     |  Importa dados do corpus TimebankPT e fornece vários métodos para manipular o conteúdo do corpus.
     |  Se existir o arquivo 'dataset/corpus.pickle', o carregamento rápido do corpus é acionado.
     |  Se não existir, o corpus existente na pasta 'path_tml' é processado, o carregamento é mais lento.
     |  O arquivo 'dataset/corpus.pickle' é salvo pelo método: 
     |      TimebankPT.df.save_corpus(tb.path_corpus_pickle).
     |  
     |  Se existir o arquivo 'dataset/data_pipe_tb.pickle', o carregamento rápido dos dados do pipeline é acionado.
     |  Se não existir, os dados para o pipeline são processados na inicialização de Timebank.df.
     |  O arquivo 'dataset/data_pipe_tb.pickle' é salvo pelo método: 
     |      TimebankPT.save_data_pipe_tb(tb.path_data_pipe_pickle).
     |  
     |  Args:
     |      path_tml: caminho do corpus TimebankPT no formato: 'c:\diretorio\*\*.txt'
     |      add_timebank: adiciona tags (EVENT, TIMEX3 e TLINK) do corpus TimebankPT ao pipeline do spaCy. Default é True
     |      dev: Se True, os dados de treino são divididos em 'train' e 'train_test'. 'test' não deve ser utilizado.
     |          Se False, todo dado de treino é 'train' e 'test' é utilizado.
     |      ignore_load_corpus: Se True, carrega o corpus previamente salvo.
     |          Se False, processa o carragamento do curpus a partir dos arquivo .tml
     |  
     |  Method resolution order:
     |      TimebankPT
     |      Functions
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, path_tml, add_timebank=True, lang='pt', dev: bool = False, ignore_load_corpus=False)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __str__(self)
     |      Exibe as quantidades dos objetos do TimebankPT
     |  
     |  add_pipe_timebank(self)
     |      Adiciona o pipe que adiciona tags dos timebankPT ao Doc no spaCy
     |  
     |  check_filename(self, filename: str, extensao: str = '', check_if_exist: bool = False) -> str
     |      Retorna nome do arquivo com a extensão padrão .'extensão' se não for informada em filename. Pode verificar se o arquivo existe conforme 'check_if_exist'.
     |      
     |      Args:
     |          filename: nome do arquivo. Extensão no nome do arquivo tem prevalência sobre 'extensão'.
     |          extensao: extensão padrão do arquivo
     |          check_if_exist: Se True, verifica se o arquivo existe
     |  
     |  eh_nome_doc(self, nome_doc: str)
     |      Verifica se o nome_doc pertence aos arquivos do corpus TimeBankPT
     |      
     |      Args:
     |          nome_doc: Nome de arquivo do TimebankPT
     |  
     |  get_dct_doc(self, nome_doc=None)
     |      Retorna o value da Data de Criação do Documento(DCT) da id_sentença setada em set_id_sentenca() ou str da id_sentenca do parametro
     |      
     |      Args:
     |          nome_doc: se informado, sobrepõe as id_sentenca atribuída em set_id_sentenca()
     |      
     |      Return:
     |          value do DCT
     |  
     |  get_dct_doc_helper(self, nome_doc=None, retorno='dct')
     |      Retorna lista com dados da Data de Criação do Documento(DCT) da id_sentença setada em set_id_sentenca() ou de 'nome_doc' do parametro
     |      
     |      Args:
     |          nome_doc: se informado, sobrepõe as id_sentenca atribuída em set_id_sentenca()
     |          retorno: pode retornar 
     |              'dct': value da data de criação do documento
     |              'type': um dos tipos de TIMEX3 (DATE, TIME, DURATION, SET)
     |              'tid': id do TIMEX3 que é o DCT
     |  
     |  get_dct_doc_tid(self, nome_doc=None)
     |      Retorna o tid da Data de Criação do Documento(DCT) da id_sentença setada em set_id_sentenca() ou str da id_sentenca do parametro
     |      
     |      Args:
     |          nome_doc: se informado, sobrepõe as id_sentenca atribuída em set_id_sentenca()
     |      
     |      Return:
     |          tid do DCT
     |  
     |  get_dct_doc_type(self, nome_doc=None)
     |      Retorna o type da Data de Criação do Documento(DCT) da id_sentença setada em set_id_sentenca() ou str da id_sentenca do parametro
     |      
     |      Args:
     |          nome_doc: se informado, sobrepõe as id_sentenca atribuída em set_id_sentenca()
     |      
     |      Return:
     |          type do DCT
     |  
     |  get_doc(self, id_sentenca=None)
     |      Retorna lista de objetos Doc do spaCy
     |      
     |      Return:
     |          Lista de Docs
     |  
     |  get_doc_root(self, doc: spacy.tokens.doc.Doc = None) -> list
     |      Retorna lista de roots do Doc.
     |  
     |  get_doc_unico(self)
     |      Retorna o primeiro Doc da lista de Docs (self.doc).
     |      
     |      Return:
     |          Doc
     |  
     |  get_eventID(self, id_sentenca=None)
     |      Retorna lista de eventID de id_sentenca
     |  
     |  get_id_sentenca(self, texto_sentenca=None)
     |      Retorna as id_sentenca setadas em set_id_sentenca() se texto_sentenca não for informado
     |      
     |      Se texto_sentenca for informado:      
     |          Busca a id_sentenca correspondente à texto_sentenca no TimeBankPt, e a retorna.
     |      
     |      Args: 
     |          texto_sentenca: Texto da sentença a ser procurada em TimeBankPt
     |          
     |      Return:
     |          list id_sentenca
     |  
     |  get_id_sentenca_do_doc(self, id: str, nome_documento: str) -> int
     |      Retorna id_sentenca do doc e eid/tid.
     |      
     |      Args:
     |          id: pode ser o id do evento ou do timex3
     |          nome_documento: nome do arquivo que representa um documento do corpus
     |  
     |  get_id_sentenca_unica(self)
     |      Retorna lista com a primeira id_sentenca da lista de sentenças setadas em set_id_sentenca()
     |  
     |  get_id_sentencas_dep(self)
     |      Retorna também lista de id_sentenca dependentes da sentenca do documento atual.
     |  
     |  get_id_sentencas_doc(self)
     |      Retorna lista de todas id_sentenca do primeiro documento da lista de documentos atual.
     |      O nome do documento é o nome do arquivo do TimeBankPT.
     |  
     |  get_nome_doc(self, id_sentenca=None)
     |      Retorna lista com nome do documento da id_sentença setada em set_id_sentenca() ou str da id_sentenca do parametro
     |      
     |      Args:
     |          id_sentenca: se informado, sobrepõe id_sentenca atribuido em set_id_sentenca()
     |      
     |      Return:
     |          Nome de documentos. É o nome do arquivo do TimebankPT
     |  
     |  get_nome_doc_unico(self)
     |      Retorna primeiro nome de documento da lista de nomes de documentos (self.nome_doc).
     |      
     |      Return:
     |          string
     |  
     |  get_nome_documento(self, id_sentenca=None)
     |      Retorna nome do documento de 'id_sentenca'.
     |      
     |      Args:
     |          id_sentenca: pode ser list ou int
     |              se não for informado, retorna o nome do documento de id_sentenca da classe.
     |              se lista, retorno o documento do primeiro id_sentenca
     |  
     |  get_sentenca_texto(self, id_sentenca=None)
     |      Retorna lista contendo texto da sentença com as tag TimeML, conforme id_sentenca setadas em set_id_sentenca() ou informado no parâmetro id_sentenca
     |  
     |  get_sentenca_texto_doc(self)
     |      Retorna lista de texto de todas as sentenças do documento da primeira sentença de get_id_sentenca().
     |  
     |  get_sentenca_texto_tag(self, id_sentenca=None)
     |      Retorna lista contendo texto da sentença com as tag TimeML, conforme id_sentenca setadas em set_id_sentenca().
     |  
     |  get_train_test(self, nome_doc: str) -> Literal['train', 'train_test', 'test']
     |      Retorna qual o grupo de desenvolvimento que o documento pertence.
     |      
     |      Args:
     |          nome_doc: nome do documento
     |      
     |      Return:
     |          train:      Se é de treino.
     |          train_test: Se é de teste para o conjunto de treino (Dev).
     |          test:       Se é de teste global. Utilizado apenas no trabalho final.
     |  
     |  id_sentencas_task(self, task: str)
     |      Retorna id_sentenca contempladas pela tarefa 'task'
     |      
     |      Args:
     |          task: filtrar conforme task
     |          filter
     |  
     |  load_data_pipe_tb(self, nome_arquivo: str) -> dict
     |      Retorna objeto que representa os dados do pipe salvo pelo método 'save_data_pipe_tb(nome_arquivo)'.
     |      Os objetos retornados são: event, timex3, tlink, sentenca, documento
     |  
     |  pesquisa_id_sentenca(self, lista_termos, formato_dataframe=False)
     |      Retorna DataFrame com resultado pesquisa dos termos
     |      
     |      Args:
     |          lista_termos: lista de palavras a ser pesquisada em sentenças
     |          
     |          formato_dataframe: se True, retorna o dataframe filtrado por lista_termos, se não, retorna lista de id_sentenca que atendem ao critério de pesquisa
     |  
     |  pesquisa_sentenca_texto(self, lista_termos='', formato_dataframe=False)
     |      Retorna DataFrame com resultado pesquisa dos termos
     |      
     |      Args:
     |          lista_termos: lista de palavras a ser pesquisada em sentenças
     |          
     |          formato_dataframe: se True, retorna o dataframe filtrado por lista_termos, se não, retorna lista de sentenças que atendem ao critério de pesquisa
     |  
     |  print_pipes(self)
     |      Imprime a sequência dos pipelines executados
     |  
     |  query_filtro_task(self, task: str)
     |      Retorna query que filtra sentenças conforme task
     |  
     |  remove_pipe_timebank(self)
     |      Remove o pipe_timabankpt. Retira as tag dos timabankpt (EVENT e TIMEX3)
     |  
     |  save_data_pipe_tb(self, nome_arquivo: str)
     |      Salva dados do corpus carregado em arquivo físico.
     |  
     |  set_id_sentenca(self, *id_sentenca)
     |      Atribui as id_sentenca para as instâncias da classe e atribui valores a campos que dependem de id_sentenca
     |      
     |      Args:
     |          id_sentenca: Lista de id_sentença. O id_senteca não conta nos arquivos TimeML, foram criados na função timeml_to_df para facilitar o acesso.
     |          id_sentenca pode ser vários inteiros, várias strings, lista de ambos ou strings separadas por virgulas.
     |  
     |  set_sentenca_texto(self, sentenca_texto)
     |      Permite atribuir sentenças que não estão no TimebankPT para submetê-las ao pipeline do spaCy.
     |      Caso a sentenca passada exista no TimebankPT, atribui a id_sentenca à classe com set_id_sentenca()
     |      
     |      Args:
     |          sentenca_texto: Lista de sentenças ou sentença única.
     |  
     |  trata_lista(self, *dados, tipo_lista=<class 'int'>)
     |      Retorna dados convertido em lista de strings.
     |      
     |      Args:
     |          dados: Pode vários inteiros, várias strings, lista de ambos ou strings separadas por virgulas.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties defined here:
     |  
     |  dados_pipe
     |  
     |  dct_doc
     |      Retorna o value da Data de Criação do Documento(DCT) da id_sentença setada em set_id_sentenca() ou str da id_sentenca do parametro
     |      
     |      Args:
     |          nome_doc: se informado, sobrepõe as id_sentenca atribuída em set_id_sentenca()
     |      
     |      Return:
     |          value do DCT
     |  
     |  doc
     |      Retorna lista de objetos Doc do spaCy
     |      
     |      Return:
     |          Lista de Docs
     |  
     |  doc_root
     |      Retorna lista de roots do Doc.
     |  
     |  doc_unico
     |      Retorna o primeiro Doc da lista de Docs (self.doc).
     |      
     |      Return:
     |          Doc
     |  
     |  id_sentenca_unica
     |      Retorna lista com a primeira id_sentenca da lista de sentenças setadas em set_id_sentenca()
     |  
     |  id_sentencas_dep
     |      Retorna também lista de id_sentenca dependentes da sentenca do documento atual.
     |  
     |  id_sentencas_doc
     |      Retorna lista de todas id_sentenca do primeiro documento da lista de documentos atual.
     |      O nome do documento é o nome do arquivo do TimeBankPT.
     |  
     |  nome_doc
     |      Retorna lista com nome do documento da id_sentença setada em set_id_sentenca() ou str da id_sentenca do parametro
     |      
     |      Args:
     |          id_sentenca: se informado, sobrepõe id_sentenca atribuido em set_id_sentenca()
     |      
     |      Return:
     |          Nome de documentos. É o nome do arquivo do TimebankPT
     |  
     |  nome_doc_unico
     |      Retorna primeiro nome de documento da lista de nomes de documentos (self.nome_doc).
     |      
     |      Return:
     |          string
     |  
     |  sentenca_texto_doc
     |      Retorna lista de texto de todas as sentenças do documento da primeira sentença de get_id_sentenca().
     |  
     |  sentenca_texto_tag
     |      Retorna lista contendo texto da sentença com as tag TimeML, conforme id_sentenca setadas em set_id_sentenca().
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  id_sentenca
     |      Retorna as id_sentenca setadas em set_id_sentenca() se texto_sentenca não for informado
     |      
     |      Se texto_sentenca for informado:      
     |          Busca a id_sentenca correspondente à texto_sentenca no TimeBankPt, e a retorna.
     |      
     |      Args: 
     |          texto_sentenca: Texto da sentença a ser procurada em TimeBankPt
     |          
     |      Return:
     |          list id_sentenca
     |  
     |  sentenca_texto
     |      Retorna lista contendo texto da sentença com as tag TimeML, conforme id_sentenca setadas em set_id_sentenca() ou informado no parâmetro id_sentenca
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  Df = <class 'parse.ParseTimebankPT.TimebankPT.Df'>
     |      Cria DataFrame para os diversos elementos do corpus TimebankPT: EVENT, TIMEX3, TLINK, Sentenças, Nome do Documento e Data de Criação do Documento (DCT)
     |      
     |      Args:
     |          tb: Recebe instancia da classe TimebankPT
     |  
     |  
     |  MyTlink = <class 'parse.ParseTimebankPT.TimebankPT.MyTlink'>
     |      Estrutura de dados para as Relações Temporais previstas pelo método aqui proposto.
     |      Fornece impressão gráfica das relações.
     |      
     |      Args:
     |          tb: Instancia da classe TimebankPT.
     |  
     |  
     |  Print = <class 'parse.ParseTimebankPT.TimebankPT.Print'>
     |      Formata para impressão em tela os elementos do Timebank e recursos do spaCy como Entidades, POS, Morph, árvore de dependência.
     |      
     |      Args:
     |          tb: Instancia da classe TimebankPT.
     |  
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Functions:
     |  
     |  explicar_spacy(self, elemento)
     |      Retorna descrição explicativa sobre elementos POS e DEP do spaCy.
     |  
     |  get_class_list(self, obj=None)
     |      Retorna lista com as propriedades, funções e tipos presentes no objeto atual.
     |  
     |  nbor(self, token: spacy.tokens.token.Token, n: int) -> spacy.tokens.token.Token
     |      Retorna o token n vizinho de 'token'.
     |      n negativo: vizinho a esquerda. 
     |      n positivos: vizinho a direita.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from Functions:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    


```python
help(tb.df)
```

    Help on Df in module parse.ParseTimebankPT object:
    
    class Df(builtins.object)
     |  Df(tb: 'TimebankPT')
     |  
     |  Cria DataFrame para os diversos elementos do corpus TimebankPT: EVENT, TIMEX3, TLINK, Sentenças, Nome do Documento e Data de Criação do Documento (DCT)
     |  
     |  Args:
     |      tb: Recebe instancia da classe TimebankPT
     |  
     |  Methods defined here:
     |  
     |  __init__(self, tb: 'TimebankPT')
     |      Processa carregamento dos dados do corpus a partir dos arquivos .tml para Dataframes.
     |  
     |  atualizar_filtros(self)
     |      Carrega os DataFrames filtrados conforme parâmetros.
     |      É chamada sempre que uma propriedade da classe é alterado, por exemplo, set_id_sentenca, set_sentenca_texto, recursivo.
     |  
     |  dataset(self, dados: Literal['train', 'test', 'all'] = 'all')
     |      Dataset de dados anotados do corpus
     |  
     |  lista_arquivos(self)
     |      Retorna lista de arquivos do path.
     |  
     |  load_corpus(self, nome_arquivo: str)
     |      Retorna objeto que representa o corpus salvo pelo método 'save_corpus(nome_arquivo)'.
     |      Os objetos retornados são: event, timex3, tlink, sentenca, documento
     |      
     |      nome_arquivo: caminho e nome do arquivo formato .pickle
     |  
     |  save_corpus(self, nome_arquivo: str)
     |      Salva dados do corpus em arquivo físico (.pickle).
     |  
     |  split_isentencas_kfolds(self, df: pandas.core.frame.DataFrame, k: int) -> List[list]
     |      Retorna 'k' listas de id_sentenca do 'df' (k-folds). 
     |      Não considera a proporção das classes (não estratificado).
     |      
     |      Args:
     |          df: Dataframe. Ex: tb.df.dataset('test')
     |          k: quantidade de folds
     |  
     |  split_train_isentencas_kfolds(self, df, k: int, size_test: int = 1)
     |      Retorna 'k' listas de id_sentenca dos dados de treino dos k-folds. 
     |      Considera a proporção de cada classe (estratificado).
     |      Ex: k = 5, size_test = 2
     |          Os conjuntos de treino de onde as id_sentencas serão retiradas:
     |          [2,3,4], [0,3,4], [0,1,4], [0,1,2], [1,2,3]
     |      
     |      Args:
     |          df: Dataframe. Ex: tb.df.dataset('test')
     |          k: quantidade de folds
     |          size_test: Quantidade de folds para os dados de teste
     |  
     |  split_train_test_kfolds(self, df, k: int, size_test: int = 1)
     |      Retorna lista com 'k' dataframes com dados de treino e de teste, já com os k-folds alternados, conforme size_test.
     |      Ex: k = 5, size_test = 2
     |          Os conjuntos de treino serão:
     |          [2,3,4], [0,3,4], [0,1,4], [0,1,2], [1,2,3]
     |      
     |      Args:
     |          df: Dataframe. Ex: tb.df.dataset('test')
     |          k: quantidade de folds
     |          size_test: Quantidade de folds para os dados de teste
     |      
     |      Return:
     |          list_train: Lista de dataframes dos dados de treino com a classe
     |          list_test: Lista de dataframes dos dados de teste com a classe
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties defined here:
     |  
     |  X_test
     |      Dados de teste sem a classe
     |  
     |  X_train
     |      Dados de treino sem a classe
     |  
     |  dados_pipe
     |      Retorna dicionário contendo os dados necessários para o processamento do pipeline do spaCy: pipe_timebankpt.
     |      
     |      Return:
     |          {
     |              'texto da sentença': 
     |              {
     |                  'isentenca': 'id sentença',
     |                  'doc': 'nome do arquivo doc',
     |                  'dct': 'data de criação do documento',
     |                  'lista_event': [[], [], []],
     |                  'lista_timex3': [[], []],
     |                  'lista_tlink': [[], []]
     |              },
     |              
     |              'Repetidamente, ele resiste.':
     |              {  
     |                  'isentenca': '254', 
     |                  'doc': 'ABC19980120.1830.0957', 
     |                  'dct': '1998-01-20', 
     |                  'lista_event': [['EVENT', '2', 'e1', 'previram', 14, 22, 'I_ACTION', 'NONE'], ['EVENT', '2', 'e86', 'queda', 29, 34, 'OCCURRENCE', 'NONE']], 
     |                  'lista_timex3': [['TIMEX3', '10', 't94', 'quase quarenta anos', 8.0, 27.0, 'DURATION', 'P40Y']],
     |                  'lista_tlink': [['TLINK', 'l3', 'B', 'AFTER', 5, 'e11', '', 't93', '', '']]
     |              }
     |          }
     |  
     |  documento
     |      Retorna DataFrame contendo atributos do documento da sentença informada para a class.
     |  
     |  documento_completo
     |      Retorna DataFrame contendo atributos de todos os documento do corpus.
     |  
     |  event
     |      Retorna DataFrame contendo todos atributos de EVENT, porém apenas das sentenças informada para a class.
     |  
     |  event_completo
     |      Retorna DataFrame contendo todos atributos de EVENT de todas as sentenças do corpus.
     |  
     |  event_doc
     |      Retorna DataFrame contendo todos atributos de EVENT, porém apenas as sentenças do documento atual.
     |  
     |  quant_doc
     |  
     |  quant_doc_total
     |  
     |  quant_event
     |  
     |  quant_event_total
     |  
     |  quant_sentenca
     |  
     |  quant_sentenca_total
     |  
     |  quant_timex3
     |  
     |  quant_timex3_total
     |  
     |  quant_tlink
     |  
     |  quant_tlink_total
     |  
     |  sentenca
     |      Retorna DataFrame contendo atributos das sentenças informadas para a class.
     |  
     |  sentenca_completo
     |      Retorna DataFrame contendo atributos de todas as sentenças do corpus.
     |  
     |  sentenca_doc
     |      Retorna DataFrame contendo todos atributos da sentença, porém apenas as sentenças do documento atual.
     |  
     |  timex3
     |      Retorna DataFrame contendo todos atributos de TIMEX3, porém apenas das sentenças informada para a class.
     |  
     |  timex3_completo
     |      Retorna DataFrame contendo todos atributos de TIMEX3 de todas as sentenças do corpus.
     |  
     |  timex3_doc
     |      Retorna DataFrame contendo todos atributos de TIMEX3, porém apenas as sentenças do documento atual.
     |  
     |  tlink
     |      Retorna DataFrame contendo todos atributos de TLINK, porém apenas das sentenças informadas para a class.
     |  
     |  tlink_completo
     |      Retorna DataFrame contendo todos atributos de TLINK de todas as sentenças do corpus.
     |  
     |  tlink_doc
     |      Retorna DataFrame contendo todos atributos de TLINK, porém apenas os registros do documento atual.
     |  
     |  tlink_join
     |      Retorna DataFrame de TLink unido com os campos das chaves estrangeira.
     |      Dar uma visão mais global dos campos de TLink.
     |  
     |  tlink_join_completo
     |      Retorna DataFrame de TLink completo unido com os principais campos das chaves estrangeira.
     |      Exibe todos os registros de TLink.
     |  
     |  tlink_join_doc
     |      Retorna DataFrame contendo todos atributos de TLINK e suas chaves estrangeiras, porém apenas os registros do documento atual.
     |  
     |  y_test
     |      Dados de teste somente a classe
     |  
     |  y_train
     |      Dados de treino somente a classe
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  recursivo
     |      Propriedade booleana da class que indica se as sentenças dependentes serão também exibidas.
    
    


```python
help(tb.print)
```

    Help on Print in module parse.ParseTimebankPT object:
    
    class Print(builtins.object)
     |  Print(tb: 'TimebankPT')
     |  
     |  Formata para impressão em tela os elementos do Timebank e recursos do spaCy como Entidades, POS, Morph, árvore de dependência.
     |  
     |  Args:
     |      tb: Instancia da classe TimebankPT.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, tb: 'TimebankPT')
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ent(self, id_sentenca=None)
     |      Imprime Entidades Nomeadas, inclusive as tags EVENT e TIMEX3 do TimebankPT se o seu pipeline estiver adicionado ao spaCy.
     |  
     |  filhos(self, id_sentenca=None)
     |      Imprime também os dependentes (filhos) de cada token.
     |  
     |  graph(self, id_sentenca=None, size='m', compact=True, punct=False)
     |      Imprime gráfico de análise de dependência.
     |      
     |      Args:
     |          id_sentenca: se fornecida lista de id_sentenca
     |          size: 'p', 'm', 'g' representa a distancia entre os tokens
     |          punct: se True, mostra as pontuações no grafo.
     |  
     |  graph_dfs(self, id_sentenca=None, mostrar_mais=True)
     |      Imprime árvore sintática utilizando a Busca por Profundidade
     |  
     |  graph_tlink(self, id_sentenca=None)
     |      Imprime gráfico das relações temporais anotadas no corpus entre eventos e expressões temporais (Task A).
     |  
     |  graph_treelib(self, id_sentenca=None)
     |      Imprime arvore sintática utilizando a Busca por Profundidade com a biblioteca treelib
     |      from treelib import Node, Tree
     |  
     |  imprimir_campos(self, func_campos, id_sentenca=None)
     |      Função genérica que recebe dados de tokens (em func_campos) para imprimi-los. 
     |      Os dados contem classes gramaticais (Part Of Speech), análise morfológica, análise de dependência e tags do corpus TimebankPT.
     |      
     |      Args:
     |          func_campos: Função que retorna dicionário contendo dados dos tokens que serão impressos. Os dados estão em funções iniciadas por '__campos_', ex: __campos_timebank, __campos_morph ...
     |  
     |  morph(self, id_sentenca=None)
     |      Imprime Classes gramaticais (Part Of Speech) e análise morfológica.
     |  
     |  pais(self, id_sentenca=None)
     |      Imprime também os ancestrais de cada token.
     |  
     |  timebank(self, id_sentenca=None)
     |      Imprime tags do timebank.
     |  
     |  tokens(self, id_sentenca=None)
     |      Imprime tags POS.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    


```python
help(tb.my_tlink)
```

    Help on MyTlink in module parse.ParseTimebankPT object:
    
    class MyTlink(builtins.object)
     |  MyTlink(tb: 'TimebankPT')
     |  
     |  Estrutura de dados para as Relações Temporais previstas pelo método aqui proposto.
     |  Fornece impressão gráfica das relações.
     |  
     |  Args:
     |      tb: Instancia da classe TimebankPT.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, tb: 'TimebankPT')
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  add(self, relType, eventID, relatedTo, task, isentenca, doc, rule, lid=None)
     |      Adiciona tags Tlink descoberta pelo método proposto à estrutura de dados armazenada em to_list.
     |      
     |      Args:
     |          relType: Tipo da relação temporal predita
     |          eventID: ID do EVENT 
     |          relatedTo: Pode ser relatedToTime ou relatedToEvent, é inferida através de task
     |          task: Tipo da tarefa do TempEval. 
     |              A. EVENT-TIMEX3 (maioria intra-sentença)
     |              B. EVENT-DCT  
     |              C. EVENT-EVENT (inter-sentença)
     |          isentenca: id_sentenca
     |          doc: nome do arquivo do corpus, representa um documento
     |          rule: código da regra que previu o tipo de relação
     |          lid: ID do TLINK. Se não for fornecido, é calculado automaticamente, último + 1.
     |  
     |  clear(self)
     |      Limpa todas as tags TLink adicionadas.
     |  
     |  graph_rt(self, compact=True, punct=False)
     |      Exibe as relações temporais em forma gráfica.
     |  
     |  idtag_to_token(self, id_tag: str) -> spacy.tokens.token.Token
     |  
     |  idtag_to_token_next(self, id_tag: str) -> spacy.tokens.token.Token
     |  
     |  lista_id_timebank(self, task)
     |      Retorna dicionário contendo pares conforme task.
     |      
     |      Args:
     |          task:   A. EVENT-TIMEX3 (intra-sentença)
     |                  B. EVENT-DCT 
     |                  C. EVENT-EVENT (inter-sentença consecutivas)
     |  
     |  load_from_file(self, file_tlink, modo='w')
     |      ######### PROVAVELMENTE ESTE MÉTODO SERÁ EXCLUIDO: ANALISAR ISSO DEPOIS
     |      
     |      Carrega dados do arquivo salvo dados pelo método save_to_file()
     |      
     |      Args:
     |          file_tlink: Arquivo tml contendo tags TLINK criado pelo método save_to_file()
     |      
     |          modo:   se 'w' (write), limpa as carga anterior de self.to_list, sobrescreve conteúdo já carregado.
     |                  se 'a' (append), adiciona a carga atual no final da carga existente.
     |  
     |  remove(self, relType, eventID, relatedTo, task, isentenca, doc, rule, lid=None)
     |      Remove TLink da estrutura de dados.
     |      Busca par eventID e relatedTo e apaga pelo lid encontrado.
     |  
     |  save_to_file(self, file_tlink, sobrescrever=False)
     |      Salva as tags TLINK em arquivo.
     |      
     |      Args:
     |          file_tlink: Nome do arquivo tml que conterá tags TLINK.
     |      
     |          sobrescrever:   Se True, sobrescreve o arquivo file_tlink se ele existir, se não existir, cria-o.
     |                          Se False, se o arquivo existir, não sobrescreve, não faz nada. Se o arquivo não existir, cria-o.
     |  
     |  tabela_id_timebank(self)
     |      Retorna tabela contendo todas entidades EVENT e TIMEX3 da sentença e seus respectivos IDs.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties defined here:
     |  
     |  to_df
     |      Lista tags TLink extraídas em formato DataFrame
     |  
     |  to_df_join
     |      Retorna DataFrame de MyTlink contendo os dados principais das chaves estrangeiras.
     |  
     |  to_list
     |      Estrutura de dados utilizada para armazenar as tags TLink.
     |      Utilizada para criar DataFrames e alimentar impressão gráfica das relações temporais.
     |  
     |  to_txt
     |      Lista tags TLink em formato padrão das tags TLINK dos arquivos do corpus.
     |      Utilizada para salvar em arquivo.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    


```python
help(tb.features)
```

    Help on FeaturesToDataset in module parse.ParseTimebankPT object:
    
    class FeaturesToDataset(builtins.object)
     |  FeaturesToDataset(tb: parse.ParseTimebankPT.TimebankPT)
     |  
     |  Retorna todos os pares event-time, anotados e não anotados, e todas as features utilizadas para gerar regras
     |  
     |  Args:
     |      tb: instancia da class TimebankPT
     |  
     |  Methods defined here:
     |  
     |  __init__(self, tb: parse.ParseTimebankPT.TimebankPT)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  create_dataset(self)
     |      Processa todos os pares event-time e alimenta 'dataset' com as informações linguísticas (features).
     |  
     |  dataset(self)
     |      Retorna dataframe com todas as features. Anotadas e não anotadas; train, train_test e test.
     |      Não processa novamente.
     |  
     |  dataset_teste(self, so_anotados: bool = True) -> pandas.core.frame.DataFrame
     |      Retorna dataframe com os dados de teste. Inclui a classe.
     |      
     |      Args:
     |          so_anotados: Se True, retorna apenas os dados em que o tipo da relação temporal é anotado.
     |                      Se False, retorna os anotados e os não anotados.
     |  
     |  dataset_treino(self, so_anotados: bool = True) -> pandas.core.frame.DataFrame
     |      Retorna dataframe com os dados de treino. Inclui a classe.
     |      
     |      Args:
     |          so_anotados: Se True, retorna apenas os dados em que o tipo da relação temporal é anotado.
     |                      Se False, retorna os anotados e os não anotados
     |  
     |  df_encoder(self, df: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame
     |      Aplica LabelEncoder em df
     |  
     |  generate_params_functions(self) -> dict
     |      Gera dicionário com o nome das funções que implementam as features e seus parâmetros. 
     |      É necessário gerar o dataset com as features ('TimebankPT.features.create_dataset()') ou 
     |          carregar dataset salvo ('TimebankPT.features.load_dataset(nome_arquivo)').
     |  
     |  load_dataset(self, nome_arquivo: str)
     |      Carrega arquivo contendo dataset de features salvo pelo método 'save_dataset()'.
     |      Utiliza ext .parquet, mas não é necessário informá-la em nome_arquivo
     |  
     |  save_dataset(self, nome_arquivo: str)
     |      Salva arquivo contendo dataset de features completo processado pelo método 'create_dataset()'.
     |      Utiliza ext .parquet, mas não é necessário informá-la em nome_arquivo
     |  
     |  select_best_features(self, k: int = None, metodo: str = 'rfe', cv: int = 5) -> pandas.core.indexes.base.Index
     |      Retorna as melhores features calculadas sobre os dados de treino. 
     |      Se método 'chi2' as melhores segundo cálculo do chi2 (qui-quadrado)
     |      Se 'rfe' segundo técnica de Eliminação Recursiva de Features com Validação Cruzada (RFECV)
     |      
     |      Args:
     |          k:  Quantidades de features retornadas.
     |              Se k não informado: 
     |                  . retorna todas as features em ordem de importância, se metodo for 'chi2'.
     |                  . retorna as features selecionadas pelo algoritmo, se método for 'rfe'.
     |          metodo: 'chi2' - Estatística qui-quadrado
     |                  'rfe'  - Eliminação Recursiva de Features com Validação Cruzada (RFECV) (default)
     |          cv: Quantidade de folds (default = 5).
     |  
     |  to_csv(self, nome_arquivo: str, so_anotados: bool = True)
     |      Dataset contendo features em formato csv.
     |      Salva dois arquivos. Um de treino e outro de testes conforme divisão pre-existente do corpus.
     |      
     |      Args:
     |          nome_arquivo: nome do arquivo. Salva nome_arquivo_train.csv e nome_arquivo_test.csv
     |          so_anotados: Se True, salva apenas dados onde o tipo da relação temporal (relType) está anotado.
     |                      Se False, salva todos.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties defined here:
     |  
     |  X_test
     |  
     |  X_train
     |  
     |  X_train_encoder
     |  
     |  y_test
     |  
     |  y_train
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  filtra_sentenca_sem_predicao
     |      Se True, filtrar dataset exibindo apenas as sentenças que não houve predição.
     |      Se False, exibe dataset completo.
    
    

## Relações Temporais


```python
help(tb.tr)
```

    Help on TemporalRelation in module parse.ParseTimebankPT object:
    
    class TemporalRelation(builtins.object)
     |  TemporalRelation(tb: parse.ParseTimebankPT.TimebankPT)
     |  
     |  Identifica tipos de relações temporais em sentenças em português.
     |  
     |  Args:
     |      tb: instancia da class TimebankPT.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, tb: parse.ParseTimebankPT.TimebankPT)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  add_rule_class_default(self, class_default: str)
     |      Adicionar regra para classe default em TemporalRelation.rules
     |  
     |  add_setRules(self, rules: List[list], reset_cod_regra: bool = True, reset_order: bool = False)
     |      Adiciona conjunto de regras provenientes dos objetos TemporalRelation.rules ou TemporalRelation.setRules.rules
     |      Para adicionar de arquivos em formato texto, use TemporalRelation.load_rules(nome_arquivo)
     |      
     |      Args:
     |          setRules: conjunto de regras no formato lista de listas dos objetos:
     |              TemporalRelation.rules ou 
     |              TemporalRelation.setRules.rules
     |  
     |  calculate_metrics(self, setRules: parse.ParseTimebankPT.SetRules = None)
     |      Calcula métricas do conjunto de regras.
     |      
     |      Args:
     |          setRules: Instância do objeto SetRules (opcional).
     |              Se informado calcula métricas de TemporalRelation.SetRules.rules e de TemporalRelation.rules.
     |              Se não informado calcula apenas de TemporalRelation.rules.
     |  
     |  check_rules(self, rules: list) -> bool
     |      Checa a consistências da lista de regras. 
     |      
     |      Args: 
     |          rules: lista que contem as regras.
     |  
     |  cm(self)
     |      Matrix de Confusão das predições considerando o contexto:
     |      self.show_result, self.processing_type, self.active_tlink_transitive
     |  
     |  convert_id_to_token(self, eventID: str, relatedTo: str) -> tuple
     |      Converte ids de EVENT e TIMEX3 para Token.
     |      
     |      Args:
     |          eventID: ID do EVENT no corpus
     |          relatedTo: ID do TIMEX3 ou EVENT no corpus
     |      
     |      Return
     |          Tupla contendo tokens que representam a entidade EVENT e/ou TIMEX3.
     |  
     |  ct(self)
     |      Cross Tab dos resultados do processamento das regras considerando o contexto:
     |      self.show_result, self.processing_type, self.active_tlink_transitive
     |  
     |  df_real_predict_id_sentenca(self, id_sentenca=None, extra: bool = False)
     |      Une dados real do corpus com os dados previsto.
     |      
     |      Args:
     |          id_sentenca: Se informada, filtra por id_sentenca.
     |          extra: Se True, exibe as previsões que não estão anotadas no corpus, se False (default), exibe apenas as previsões que há correspondência no corpus.
     |  
     |  filter_rules_acuracia(self, value: float)
     |      Filtra conjunto de regras atual por 'acurácia' maior que 'value'
     |  
     |  filter_rules_ordem(self, value: float)
     |      Filtra conjunto de regras atual por 'ordem' menor que 'value'
     |  
     |  filter_rules_primeiras(self, x: int)
     |      Filtra conjunto de regras atual pelas primeiras 'X' regras por tipo de origem
     |  
     |  get_rules(self, cod_rules: list)
     |      Retorna lista de regras que possuem 'cod_rules'.
     |      
     |      Args:
     |          cod_rules: lista de códigos de regras
     |  
     |  get_sort_reverse(self)
     |      Obtém a informação sobre se a ordem das regras está ascendente ou descendente.
     |  
     |  get_sort_rules(self) -> str
     |      Obtém a ordem atual das regras.
     |  
     |  graph_pred(self, id_sentenca=None)
     |      Exibe as relações temporais de forma gráfica da predições realizadas por este sistema.
     |  
     |  has_rule_class_default(self) -> str
     |      Verifica se há regras com classe default.
     |      Se houver, retorna a regra de classe default.
     |  
     |  id_sentencas_show_result(self, task: str)
     |      Retorna id_sentenca contempladas pela tarefa 'task' e filtrada conforme TemporalRelation.show_result 
     |      que consiste em selecionar dados de traino ou de teste para exibir o resultado
     |      
     |      Args:
     |          task: filtrar conforme task
     |  
     |  load_results(self, nome_arquivo: str)
     |      Retorna dataset de resultados salvo pelo método 'save_results()'.
     |      Utiliza ext .parquet, mas não é necessário informá-la em nome_arquivo
     |  
     |  load_rules(self, nome_arquivo: str)
     |      Carrega conjunto de regras em formato .pickle ou .txt.
     |      Se .pickle, deve ter sido salvas pelo método TemporalRelation.save_rules(nome_arquivo).
     |      Se .txt, deve ter sido salvas pelo método TemporalRelation.save_rules_to_txt(nome_arquivo).
     |      
     |      Args:
     |          nome_arquivo: o arquivo pode estar em formato pickle ou txt.
     |  
     |  metrics(self)
     |      Resultado do processamento das regras considerando o contexto:
     |      self.show_result, self.processing_type, self.active_tlink_transitive
     |  
     |  process_resume(self)
     |      <<Precisa desbagunçar isso aqui>>
     |      
     |      Exibe resultado do processamentos do processamento das regras
     |  
     |  process_rules(self, id_sentencas=None)
     |      Realiza predição de relações temporais em todos os pares entre EVENT e TIMEX3 de todas as sentenças, conforme a tarefa.
     |      Preenche estrutura 'self.__tb.my_tlink' com dados da predição.
     |      
     |      Args:
     |          id_sentencas: Se não for informado, processa todas as sentenças cobertas pela task atual.
     |  
     |  relTypePar(self, tokenE: spacy.tokens.token.Token, tokenT: spacy.tokens.token.Token) -> str
     |      Retorna tipo da relação temporal anotada no Corpus entre tokenE e tokenT, se houver.
     |      Busca informação do relType gravada no tokenE.
     |      Só recebe pares dos dados de treino.
     |  
     |  relations_incorrect_class(self, train_test: Literal['train', 'test'], *df_real_predicts: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame
     |      Recebe vários DataFrame df_real_predict, cada um representando um resultado de setRules diferente.
     |      Retorna DataFrame com a quantidade de todas as relações incorretas.
     |      Só considera dados rotulados.
     |      
     |      Args:
     |          train_test: informar quais dados serão exibidos: 'train' ou 'test'.
     |          df_real_predict: Vários DataFrames TimebankPT.TemporalRelation.df_real_predict contendo resultados do processamento das regras.
     |  
     |  remove_rule_class_default(self)
     |      Remove todas as regras de classe default, se houver.
     |  
     |  rules_filter_list_cods(self, lista_cod_regras: list)
     |      Filtra as regras a serem processadas pelo códigos das regras informadas.
     |      Esta função altera as self.rules. Para desfazer o filtro, é necessário atribuir as regras novamente: self.rules = listas_das_regras.
     |      
     |      Args: 
     |          lista_cod_regras: lista de códigos das regras.
     |  
     |  save_results(self, nome_arquivo: str)
     |      Salva arquivo contendo dataset de resultados TemporalRelation.df_real_predict processado pelo método 'process_rules()'.
     |      Utiliza ext .parquet, mas não é necessário informá-la em nome_arquivo
     |  
     |  save_rules(self, nome_arquivo: str)
     |      Salva em formato pickle conjunto de regras carregado em TemporalRelation.rules
     |  
     |  save_rules_to_txt(self, nome_arquivo: str)
     |      Salva TemporalRelation.rules em arquivo em formato de texto.
     |      Pode ser carregado pelo método TemporalRelation.load_rules(nome_arquivo)
     |  
     |  setRules_start(self, nome_arquivo: str = '', create_dataset: bool = False)
     |      Inicia conjunto de regras.
     |      Para verificar os nomes e parâmetros das funções, é necessário carregar dataset de features.
     |      
     |      Args:
     |          create_dataset: Se True, usa o método 'TimebankPT.features.create_dataset()' para criar dataset de features do zero. Mais lento.
     |                          Se False, usa o método 'TimebankPT.features.load_dataset()' para carregar um dataset já salvo pelo método TimebankPT.features.save_dataset(nome_arquivo).
     |          nome_arquivo:   Se create_dataset for False, então nome_arquivo é obrigatório.
     |                          Representa o caminho do dataset de features salvo.
     |  
     |  sort_rules(self, order: str, reverse=False)
     |      Ordena as regras.
     |      A ordem atual pode ser obtida pelo método 'get_sort_rules()'
     |      
     |      Args:
     |          order: 'cod_regra', 'relType', 'ordem', 'random', 'origem', 'acuracia', 'acertos', 'acionamentos', 'ordem_origem'
     |                  se as regras forem processadas, permite que seja uma lista de cod_regras
     |              Se 'order' for 'ordem_origem': 
     |                  Intercala origem conforme sua ordem. Útil quando filtrado com filter_rules_primeiras_x().
     |                  Ex: CBA, CBA, CBA, IDS, IDS, IDS  --> CBA, IDS, CBA, IDS, CBA, IDS
     |  
     |  status(self)
     |      Exibe o status do processamento das regras.
     |  
     |  status_resumido(self)
     |      Exibe o status resumido do processamento das regras.
     |  
     |  status_resumido_str(self)
     |      Exibe o status resumido do processamento das regras.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties defined here:
     |  
     |  accuracy
     |      Resultado do processamento das regras considerando o contexto:
     |      self.show_result, self.processing_type, self.active_tlink_transitive
     |  
     |  df_features
     |      DESCONTINUADO.
     |      
     |      Dataframe contendo features dos tokens EVENT e TIMEX3 que poderão ser TLinks.
     |      Utilizado para análises manuais na seleção de TLinkCandidate 
     |      e para alimentar dados para treinamento do modelo de machine learning
     |  
     |  df_pred
     |      Exibe em DataFrame as predição de relações temporais processadas pelo método process_rules().
     |  
     |  df_real
     |      Sentenças anotadas no corpus que atendem aos critérios de cada task.
     |  
     |  df_real_predict
     |      Retorna dataframe com dados real do corpus e com dados previstos unidos.
     |      Apenas aquelas previsões em que há correspondência no corpus.
     |  
     |  df_real_predict_extra
     |      Retorna dataframe com dados real do corpus e com dados previstos unidos.
     |      Inclui também as previsões que não estão anotadas no corpus.
     |  
     |  df_resultado_geral
     |      Retorna resumo do resultado geral.
     |  
     |  df_resultado_por_classe
     |  
     |  df_resultado_por_documento
     |  
     |  df_resultado_por_regras
     |  
     |  df_resultado_por_sentenca
     |  
     |  df_resultado_por_task
     |  
     |  df_resultado_regras_por_classe
     |  
     |  df_rules
     |      Exibe as regras ativas em formato de tabela.
     |  
     |  f1_score
     |      Resultado do processamento das regras considerando o contexto:
     |      self.show_result, self.processing_type, self.active_tlink_transitive
     |  
     |  id_sentencas_sem_predicao
     |      Lista de id_sentenca que ainda não houve predição.
     |  
     |  pct_acerto
     |      Taxa de classificações corretas, considerando também dados não rotulados
     |  
     |  pct_acerto_anotado
     |      ACURÁCIA: Taxa de classificações corretas, considerando apenas dados rotulados
     |  
     |  pct_cobertura
     |      Taxa de cobertura
     |  
     |  pct_erro
     |      Taxa de classificações incorretas, considerando apenas dados rotulados
     |  
     |  precision_score
     |      Resultado do processamento das regras considerando o contexto:
     |      self.show_result, self.processing_type, self.active_tlink_transitive
     |  
     |  quant_acerto
     |      Total de classificações corretas, considerando apenas dados rotulados
     |  
     |  quant_anotado
     |      Quantidade de predições, considerando apenas dados rotulados
     |  
     |  quant_erro
     |      Total de classificações incorretas, considerando apenas dados rotulados
     |  
     |  quant_nao_anotado
     |      Quantidade de predições, considerando apenas dados não rotulados
     |  
     |  recall_score
     |      Resultado do processamento das regras considerando o contexto:
     |      self.show_result, self.processing_type, self.active_tlink_transitive
     |  
     |  support
     |      Resultado do processamento das regras considerando o contexto:
     |      self.show_result, self.processing_type, self.active_tlink_transitive
     |  
     |  y
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  active_tlink_candidate
     |      DESCONTINUADO.
     |      
     |      Propriedade booleana que informa à classe se serão ou não selecionados EVENTs com maior probabilidade se estarem anotados no corpus para posterior geração de TLINKs.
     |  
     |  active_tlink_transitive
     |      Propriedade booleana que informa à classe se serão adicionados TLINKs transitivos aos TLINKs preditos pelo sistema.
     |  
     |  processing_type
     |      Propriedade que define a forma como as regras serão processadas.
     |      Se igual a  'votacao' -> Todas as regras são processadas para todos os pares de relação. A relação temporal mais frequente é retornada. (default)
     |                  'primeira_regra' -> A relação temporal do par é retornada na primeira regra acionada. Para o par atual, as outras regras não são processadas.
     |  
     |  rules
     |      Lista de regras que serão passadas à instancia da classe para processamento através do método process_rules().
     |      
     |      Args:
     |          rules: lista de listas, no formato: 
     |              [[código regra: float, 
     |                  tipo de relação temporal: str, 
     |                  ordem de execução: float, 
     |                  expressão lógica que representa a regra: str, 
     |                  origem: algoritmos gerador,
     |                  acuracia: float,
     |                  acertos: int,
     |                  acionamentos: int
     |              ]]
     |              As funções que compõe as regras estão em TemporalRelation.f (são acessadas geralmente com o prefixo 'self.f.'). 
     |              Ex: [249, "OVERLAP", 2, "self.f.is_dependencyType(tokenT, tokenE, 'conj')", 'RIPPER', 0, 0, 0]
     |              
     |          #A 'ordem de execução' com números negativos torna a regra inativa.
     |  
     |  show_extras
     |      Determina se os resultados das predições de dados não rotulados (extras) serão considerados.
     |  
     |  show_result
     |      Determina qual resultado será exibido: treinamento ou teste.
     |  
     |  task
     |      Define o tipo de tarefa que será processada.
     |      Pode ser: 'A', 'B', 'C'
    
    

### Funções que implementam as features que compõe as regras que identificam RT


```python
help(tb.tr.f)
```

    Help on RulesFunctions in module parse.ParseTimebankPT object:
    
    class RulesFunctions(builtins.object)
     |  RulesFunctions(tb: parse.ParseTimebankPT.TimebankPT)
     |  
     |  Funções que implementam as features que identificam Relações Temporais, 
     |  além de funções auxiliarem que podem compor regras manuais que identificam Relações Temporais.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, tb: parse.ParseTimebankPT.TimebankPT)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ancestors_between_filho_e_pai(self, Pai: spacy.tokens.token.Token, Filho: spacy.tokens.token.Token) -> List[spacy.tokens.token.Token]
     |      Retorna lista de tokens contendo todos os ancestrais entre Filho e Pai. 
     |      O primeiro elemento da lista é o 'Filho' e o Último é o anterior ao 'Pai' (i.e. exclui o Pai).
     |  
     |  aspect_progressive(self, token: spacy.tokens.token.Token) -> bool
     |      Verifica se o aspecto verbal de token é progressivo.
     |      Essa aspecto é formado por o verbo estar conjugado + o gerúndio do verbo principal (token)
     |      
     |      Args:
     |          token: token que representa o verbo ou o EVENT
     |  
     |  children(self, tPai: spacy.tokens.token.Token) -> list
     |      Retorna lista contendo todos descendentes de tPai.
     |      Usa busca em profundidade para verificar todos os filhos e filhos dos filhos.
     |      
     |      Args:
     |          tPai: Token que deseja conhecer os descendentes
     |  
     |  classe(self, token: spacy.tokens.token.Token, classe: list) -> bool
     |      Verifica se 'token' é uma das classes de eventos da lista 'classe'.
     |      
     |      Args:
     |          token: Token
     |          classe: lista de classes do evento
     |  
     |  closelyFollowing(self, token1: spacy.tokens.token.Token, token2: spacy.tokens.token.Token, distancia=10) -> bool
     |      Retorna True se os token estiverem a uma distancia de no máximo 'distancia' tokens.
     |  
     |  closest_to_token(self, tokenPrecede: spacy.tokens.token.Token, token: spacy.tokens.token.Token, tokenFollow: spacy.tokens.token.Token)
     |      Retorna o token mais mais próximo de 'token'.
     |      Ou tokenPrecede ou tokenFollow.
     |  
     |  contextBy(self, token: spacy.tokens.token.Token, tipo: Literal['str', 'str_lemma', 'token', 'digito', 'pos', 'dep', 'morph'], valor=None, distancia='max', contexto: Literal['antes', 'depois'] = None) -> bool
     |      Procura elementos na sentença conforme o tipo e a partir do 'token' na direção do contexto.
     |  
     |  dep(self, token: spacy.tokens.token.Token, dep: list) -> bool
     |      Verifica se 'token' possui um dos tipos de dependência da lista 'dep'.
     |      
     |      Args:
     |          token: Token
     |          dep: lista de tipo de dependências.
     |  
     |  dependencyType(self, tokenPai: spacy.tokens.token.Token, tokenFilho: spacy.tokens.token.Token) -> str
     |      Retorna a relação de dependência entre 'tokenPai' e 'tokenFilho'.
     |      
     |      Args:
     |          tokenPai: governor
     |          tokenFilho: dependent
     |      
     |      Retorna
     |          String que representa o tipo de dependência de tokenFilho para tokenPai.
     |  
     |  distance_tokens(self, token1: spacy.tokens.token.Token, token2: spacy.tokens.token.Token)
     |  
     |  event_aspect(self, E: spacy.tokens.token.Token)
     |      Aspecto de EVENT
     |  
     |  event_between_order(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Se há outro EVENT entre EVENT e TIMEX
     |  
     |  event_class(self, E: spacy.tokens.token.Token) -> str
     |      Class de EVENT
     |  
     |  event_closest_to_event_class(self, E: spacy.tokens.token.Token)
     |      Class do EVENT mais próximo do EVENT do par da relação
     |  
     |  event_closest_to_event_equal_class(self, E: spacy.tokens.token.Token)
     |      Class do EVENT da relação em consideração == CLASS do EVENT mais próximo a ele
     |  
     |  event_closest_to_event_equal_lemma(self, E: spacy.tokens.token.Token)
     |      LEMMA de EVENT da relação em consideração == LEMMA de EVENT mais próximo a ele
     |  
     |  event_closest_to_event_equal_pos(self, E: spacy.tokens.token.Token)
     |      POS de EVENT da relação em consideração == POS de EVENT mais próximo a ele
     |  
     |  event_closest_to_event_equal_tense(self, E: spacy.tokens.token.Token)
     |      Tense do EVENT da relação em consideração == Tense do EVENT mais próximo a ele
     |  
     |  event_closest_to_event_pos(self, E: spacy.tokens.token.Token)
     |      POS de EVENT mais próximo do EVENT da relação em consideração
     |  
     |  event_closest_to_event_temporal_direction(self, E: spacy.tokens.token.Token)
     |      Direção temporal de EVENT mais próximo de EVENT da relação temporal em consideração
     |  
     |  event_closest_to_event_tense(self, E: spacy.tokens.token.Token)
     |      Tense de EVENT mais próximo do EVENT da relação em consideração
     |  
     |  event_closest_to_timex3_equal_pos(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      POS de EVENT da relação em consideração == POS de EVENT mais próximo do TIMEX da relação em consideração
     |  
     |  event_closest_to_timex3_pos(self, T: spacy.tokens.token.Token)
     |      POS do EVENT mais próximo do TIMEX da relação temporal em consideração
     |  
     |  event_closest_to_token(self, token: spacy.tokens.token.Token) -> spacy.tokens.token.Token
     |      Retorna o token do evento mais próximo (da esquerda ou da direita) do evento da relação.
     |  
     |  event_closest_to_token_follow(self, token: spacy.tokens.token.Token) -> spacy.tokens.token.Token
     |      Retorna o token do evento mais próximo à direita do evento da relação.
     |  
     |  event_closest_to_token_precede(self, token: spacy.tokens.token.Token) -> spacy.tokens.token.Token
     |      Retorna o token do evento mais próximo à esquerda do evento da relação.
     |  
     |  event_closest_to_token_resource(self, token: spacy.tokens.token.Token, resource: Literal['class', 'pos', 'tense', 'temporal_direction'])
     |      Recurso do EVENT mais próximo do EVENT do par da relação
     |  
     |  event_conjunction_closest_follow(self, E: spacy.tokens.token.Token) -> str
     |      Conjunção mais próxima após o evento da relação processada.
     |  
     |  event_conjunction_closest_precede(self, E: spacy.tokens.token.Token) -> str
     |      Conjunção mais próxima antes do evento da relação processada.
     |  
     |  event_dep(self, E: spacy.tokens.token.Token)
     |      DEP de Event com seu pai
     |  
     |  event_first_order(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Se EVENT precede textualmente TIMEX na relação em consideração
     |  
     |  event_gov_verb(self, E: spacy.tokens.token.Token) -> str
     |      Verbo que rege o EVENT (Para eventos que são verbos, essa feature é o próprio evento)
     |      NÃO USAR? BASEADO EM PALAVRAS
     |  
     |  event_gov_verb_aspect(self, E: spacy.tokens.token.Token)
     |      Aspecto de EVENT não verbal: Se EVENT não for verbo, o aspecto é estimado pelo seu verbo governante (event.ancestor) com base em sua relação de dependência. 
     |      Se evento é verbo, o valor é do próprio evento
     |  
     |  event_gov_verb_tense(self, E: spacy.tokens.token.Token)
     |      Tense de EVENT não verbal: Se EVENT não for verbo, o tempo verbal é estimado pelo seu verbo governante (event.ancestor) com base em sua relação de dependência. 
     |      Se evento é verbo, o valor é do próprio evento. (Tense do verbo que rege o EVENT)
     |  
     |  event_has_modal_verb_precede(self, E: spacy.tokens.token.Token) -> bool
     |      Se EVENT tem auxiliares modais antes dele
     |  
     |  event_head_is_root(self, E: spacy.tokens.token.Token) -> bool
     |      O Event modifica diretamente a raiz? (ex: Event é um filho direto da raiz?)
     |  
     |  event_head_pos(self, E: spacy.tokens.token.Token)
     |      POS do pai de EVENT
     |  
     |  event_intervening_following_tense(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token)
     |      Tense de EVENT que está entre o EVENT e TIMEX, nesta ordem, da relação em consideração e está mais próximo do TIMEX. 
     |      Ex: EVENT -------- event.tense -- TIMEX
     |  
     |  event_intervening_preceding_class(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token)
     |      Class do EVENT que está entre o TIMEX e EVENT, nesta ordem, da relação em consideração e está mais próximo do TIMEX. 
     |      Ex: TIMEX -- event.class -------- EVENT
     |  
     |  event_is_ancestor_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      EVENT é a entidade regente na relação?
     |  
     |  event_is_child_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      EVENT é a entidade dependente na relação?
     |  
     |  event_is_pai_direto_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Verifica se EVENT é pai direto de TIMEX
     |  
     |  event_modal_verb(self, E: spacy.tokens.token.Token) -> str
     |      Verbo modal antes de EVENT
     |  
     |  event_polarity(self, E: spacy.tokens.token.Token) -> str
     |      Polaridade de EVENT
     |  
     |  event_pos(self, E: spacy.tokens.token.Token) -> str
     |      POS de EVENT
     |  
     |  event_pos_token_1_follow(self, E: spacy.tokens.token.Token)
     |      POS do 1º token após o EVENT
     |  
     |  event_pos_token_1_precede(self, E: spacy.tokens.token.Token)
     |      POS do 1º token antes o EVENT
     |  
     |  event_pos_token_2_follow(self, E: spacy.tokens.token.Token)
     |      POS do 2º token depois o EVENT
     |  
     |  event_pos_token_2_precede(self, E: spacy.tokens.token.Token)
     |      POS do 2º token antes o EVENT
     |  
     |  event_pos_token_3_follow(self, E: spacy.tokens.token.Token)
     |      POS do 3º token depois o EVENT
     |  
     |  event_pos_token_3_precede(self, E: spacy.tokens.token.Token)
     |      POS do 3º token antes o EVENT
     |  
     |  event_preposition_gov(self, E: spacy.tokens.token.Token) -> spacy.tokens.token.Token
     |      preposição que rege sintaticamente o EVENT
     |  
     |  event_preposition_precede(self, E: spacy.tokens.token.Token)
     |      Preposições que precedem um EVENT, ou NONE se essa palavra não for uma preposição.
     |  
     |  event_root(self, E: spacy.tokens.token.Token) -> bool
     |      EVENT é a raiz da sentença? (bool)
     |  
     |  event_temporal_direction(self, E: spacy.tokens.token.Token)
     |      Mapeamento manual entre EVENTs e a relação temporal esperada com seu complemento.
     |  
     |  event_tense(self, E: spacy.tokens.token.Token) -> str
     |      Tense de EVENT
     |  
     |  event_timex3_dep(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> str
     |      DEP entre EVENT/TIMEX e TIMEX/EVENT, se houver
     |  
     |  event_timex3_distance(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token)
     |      A distância entre EVENT e TIMEX (número de tokens categorizado)
     |      Escala: “perto” até 4 tokens, “distancia_media”: 5 a 9, “longe”: 10 a 14 e "muito_longe": 14+
     |  
     |  event_timex3_no_between_order(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      True se não houver EVENT ou TIMEX entre o par EVENT/TIMEX da relação que está sendo processada. 
     |      (é verdadeiro se e somente se timex3-between-order e event-between-order são falsos)
     |  
     |  followedBy(self, token: spacy.tokens.token.Token, tipo: Literal['str', 'str_lemma', 'token', 'digito', 'pos', 'dep', 'morph'], valor=None, distancia='max') -> bool
     |      Procura elementos na sentença depois do 'token', conforme o 'tipo'.
     |      
     |      Args:
     |          token: objeto Token do spaCy.
     |          tipo: string e pode ser:
     |              str     -> verifica se existe palavras especificada em valor 
     |                      -> valor: str ou list; PODE SER OMITIDO.
     |              str_lemma -> verifica se existe palavras lematizadas especificada em valor 
     |                      -> valor: str ou list;
     |              token   -> verifica se 'token' vem depois do outro token especificado em valor 
     |                      -> valor: Token; PODE SER OMITIDO.
     |              digito  -> verifica se há dígitos ou pos = 'NUM' -> Não tem valor;
     |                      -> Se valor for informado, ele será a distância.
     |              pos     -> verifica se há a classe gramatical especificada em valor 
     |                      -> valor: list, ex: ['VERB', 'NOUM'];
     |              dep     -> verifica se há na árvore de dependência o elemento especificado em valor 
     |                      -> valor: list, ex: ['nsubj', 'nmod'];
     |              morph   -> verifica se há na análise morfológica o elemento especificado em valor 
     |                      -> valor: tuple (key, value), ex: ('Tense', 'Fut'). Só é permitido uma tupla.
     |          valor: valor do elemento que será procurado, conforme o tipo.
     |          distancia:  Se inteiro, é quantidade de tokens depois de 'token' onde a pesquisa será realizada.
     |                      Se string, a pesquisa será realizada em todos os tokens que vem depois 'token'.
     |  
     |  governVerb(self, token) -> spacy.tokens.token.Token
     |      Retorna o verbo pai mais próximo na hierarquia da árvore sintática.
     |  
     |  hasDepInBetween(self, token1: spacy.tokens.token.Token, token2: spacy.tokens.token.Token, dep: list)
     |      Verifica se existe a dependência 'dep' entre os dois tokens.
     |  
     |  hasDepInContext(self, token, dep: list, distancia=5, contexto=None) -> bool
     |      Verifica se existe a dependência 'dep' a uma distância de até 5 tokens, observando o contexto (antes ou depois do token).
     |  
     |  hasDepInContextFollow(self, token: spacy.tokens.token.Token, dep: list, distancia=5) -> bool
     |      Verifica se existe a dependência 'dep' a uma distância de até 5 tokens depois do 'token'.
     |      
     |      Args:
     |          token: objeto Token
     |          distancia:  Quantidade de tokens depois de 'token'. 
     |                      Se 'max' (ou qualquer string), estende até o fim da sentença, conforme contexto.
     |  
     |  hasDepInContextPrecede(self, token: spacy.tokens.token.Token, dep: list, distancia=5) -> bool
     |      Verifica se existe a dependência 'dep' a uma distância de até 5 tokens antes do 'token'.
     |      
     |      Args:
     |          token: objeto Token
     |          distancia:  Quantidade de tokens antes de 'token'. 
     |                      Se 'max' (ou qualquer string), estende até o início da sentença, conforme contexto.
     |  
     |  hasMorphInBetween(self, token1: spacy.tokens.token.Token, token2: spacy.tokens.token.Token, keyvalue: tuple)
     |      Verifica se existe na análise morfológica o 'key' de valor 'value' entre os dois tokens.
     |      
     |      Args:
     |          token: objeto Token do spaCy.
     |          key: chave que representa a morfologia que deseja o valor, ex: Tense, VerbForm
     |          value: valor da key, ex: Fut, Inf
     |  
     |  hasMorphInContext(self, token: spacy.tokens.token.Token, keyvalue: tuple, distancia=5, contexto=None)
     |      Verifica se existe na análise morfológica o 'key' de valor 'value' a uma distância de até 5 tokens, observando o contexto (antes ou depois do token).
     |  
     |  hasMorphInContextFollow(self, token: spacy.tokens.token.Token, keyvalue: tuple, distancia=5)
     |      Verifica se existe na análise morfológica o 'key' de valor 'value' a uma distância de até 5 tokens depois de 'token'.
     |      
     |      Args:
     |          token: objeto Token do spaCy.
     |          keyvalue:   Tuple que representa o par da morfologia que deseja (key, value). 
     |                      Ex:  (Tense, Fut) ou (VerbForm, Inf).
     |          distancia:  Quantidade de tokens depois de 'token'. 
     |                      Se 'max' (ou qualquer string), estende até o fim da sentença, conforme contexto.
     |  
     |  hasMorphInContextPrecede(self, token: spacy.tokens.token.Token, keyvalue: tuple, distancia=5)
     |      Verifica se existe na análise morfológica o 'key' de valor 'value' a uma distância de até 5 tokens antes de 'token'.
     |      
     |      Args:
     |          token: objeto Token do spaCy.
     |          keyvalue:   Tuple que representa o par da morfologia que deseja (key, value). 
     |                      Ex:  (Tense, Fut) ou (VerbForm, Inf).
     |          distancia:  Quantidade de tokens antes de 'token'. 
     |                      Se 'max' (ou qualquer string), estende até o início da sentença, conforme contexto.
     |  
     |  hasNoVerbInBetween(self, token1: spacy.tokens.token.Token, token2: spacy.tokens.token.Token) -> bool
     |      Verifica se não há verbos entre os tokens de início e de fim.
     |  
     |  hasNoVerbInContext(self, token: spacy.tokens.token.Token, distancia=5, contexto=None) -> bool
     |      Checks for verb within entity_context of 5 words
     |      Verifica se existe VERBO a uma distância de até 5 tokens, observando o contexto (antes ou depois do token).
     |  
     |  hasNoVerbInContextFollow(self, token: spacy.tokens.token.Token, distancia=5) -> bool
     |      Verifica se existe VERBO a uma distância de até 5 tokens depois do 'token'.
     |      
     |      Args:
     |          token: objeto Token
     |          distancia:  Quantidade de tokens depois de 'token'. 
     |                      Se 'max' (ou qualquer string), estende até o fim da sentença, conforme contexto.
     |  
     |  hasNoVerbInContextPrecede(self, token: spacy.tokens.token.Token, distancia=5) -> bool
     |      Verifica se existe VERBO a uma distância de até 5 tokens antes do 'token'.
     |      
     |      Args:
     |          token: objeto Token
     |          distancia:  Quantidade de tokens antes de 'token'. 
     |                      Se 'max' (ou qualquer string), estende até o início da sentença, conforme contexto.
     |  
     |  hasPastParticipleInContext(self, token: spacy.tokens.token.Token, distancia=5, contexto: Literal['antes', 'depois'] = None)
     |      Verifica se há particípio passado a uma distância de até 5 tokens, observando o contexto (antes ou depois do token).
     |  
     |  hasPastParticipleInContextFollow(self, token: spacy.tokens.token.Token, distancia=5)
     |      Verifica se há particípio passado a uma distância de até 5 tokens depois do 'token'.
     |  
     |  hasPastParticipleInContextPrecede(self, token: spacy.tokens.token.Token, distancia=5)
     |      Verifica se há particípio passado a uma distância de até 5 tokens antes do 'token'.
     |  
     |  hasPosInBetween(self, token1: spacy.tokens.token.Token, token2: spacy.tokens.token.Token, pos: list) -> bool
     |      Verifica se existe a classe gramatical 'pos' entre os dois tokens.
     |  
     |  hasPosInContext(self, token, pos: list, distancia=5, contexto=None) -> bool
     |      Verifica se existe a classe gramatical 'pos' a uma distância de até 5 tokens, observando o contexto (antes ou depois do token).
     |  
     |  hasPosInContextFollow(self, token: spacy.tokens.token.Token, pos: list, distancia=5) -> bool
     |      Verifica se existe a classe gramatical 'pos' a uma distância de até 5 tokens depois do 'token'.
     |      
     |      Args:
     |          token: Objeto Token do spaCy.
     |          pos: Classe gramatical - POS Tag.
     |          distancia:  Quantidade de tokens depois de 'token'. 
     |                      Se 'max' (ou qualquer string), estende até o fim da sentença, conforme contexto.
     |  
     |  hasPosInContextPrecede(self, token: spacy.tokens.token.Token, pos: list, distancia=5) -> bool
     |      Verifica se existe a classe gramatical 'pos' a uma distância de até 5 tokens antes do 'token'.
     |      
     |      Args:
     |          token: Objeto Token do spaCy.
     |          pos: Classe gramatical - POS Tag.
     |          distancia: Quantidade de tokens antes de 'token'. 
     |                  Se 'max' (ou qualquer string), estende até o início da sentença, conforme contexto.
     |  
     |  hasTenseVerbInBetween(self, token1: spacy.tokens.token.Token, token2: spacy.tokens.token.Token, tense)
     |      Verifica se tem VERB com tempo 'tense' entre dos dois tokens.
     |  
     |  hasWordInBetween(self, token1: spacy.tokens.token.Token, token2: spacy.tokens.token.Token, palavras, lemma: bool = False) -> bool
     |      Verifica se há 'palavras' entre a entidade 1 e entidade 2.
     |      
     |      Args:
     |          token1 e token2: objeto Token do spaCy.
     |          palavras: expressão que deseja encontrar. Pode ser string, lista de strings, Token ou expressão regular.
     |                  Pesquisa por palavras inteiras.
     |                  Regex gerado: (\W|^)(palavras_separadas_por_pipe)(\W|$)
     |          lemma: Se True, lematiza palavras e contexto.
     |  
     |  hasWordInContext(self, token: spacy.tokens.token.Token, palavras: list, distancia=5, contexto=None, lemma: bool = False) -> bool
     |      Verifica se existe uma das 'palavras' a uma certa distância do token, observando o contexto (antes ou depois do token).
     |  
     |  hasWordInContextFollow(self, token: spacy.tokens.token.Token, palavras: list, distancia=5, lemma: bool = False) -> bool
     |      Verifica se existe 'palavra' a uma distância de até 5 tokens depois do 'token'.
     |      
     |      Args:
     |          token: Objeto Token do spaCy
     |          palavras: Expressão que deseja encontrar. Pode ser string, lista de strings, Token ou expressão regular.
     |                  Pesquisa por palavras inteiras.
     |                  Regex gerado: (\W|^)(palavras_separadas_por_pipe)(\W|$)
     |          distancia: Quantidade de tokens antes ou depois do token. 
     |                  Se 'max' (ou qualquer string), estende até o final ou o início da sentença, dependendo do contexto.
     |          lemma: Se True, lematiza palavras e contexto.
     |  
     |  hasWordInContextPrecede(self, token: spacy.tokens.token.Token, palavras: list, distancia=5, lemma: bool = False) -> bool
     |      Verifica se existe 'palavras' a uma distância de até 5 tokens antes do 'token'.
     |      
     |      Args:
     |          token: Objeto Token do spaCy
     |          palavras: Expressão que deseja encontrar. Pode ser string, lista de strings, Token ou expressão regular.
     |                  Pesquisa por palavras inteiras.
     |                  Regex gerado: (\W|^)(palavras_separadas_por_pipe)(\W|$)
     |          distancia: Quantidade de tokens antes ou depois do token. 
     |                  Se 'max' (ou qualquer string), estende até o final ou o início da sentença, dependendo do contexto.
     |          lemma: Se True, lematiza palavras e contexto.
     |  
     |  has_dep_list_token(self, list_tokens: List[spacy.tokens.token.Token], dep: list) -> bool
     |      Verifica de há a dependência sintática 'dep' na lista de tokens
     |  
     |  identicalHead(self, token1, token2) -> bool
     |      Verifica se ambos os tokens possuem a mesma palavra principal (head).
     |  
     |  isEvent(self, token)
     |      Verifica se o token é um EVENT
     |  
     |  isTimex3(self, token)
     |      Verifica se o token é um TIMEX3
     |  
     |  is_child(self, tPai: spacy.tokens.token.Token, tFilho: spacy.tokens.token.Token) -> bool
     |      Retorna True se tFilho for descendente de tPai
     |      
     |      Args:
     |          tPai: Token pai
     |          tFilho: Token filho
     |  
     |  is_dependencyType(self, tokenPai: spacy.tokens.token.Token, tokenFilho: spacy.tokens.token.Token, tipo_dep: str) -> bool
     |      Checks for type(token1=governor, token2=dependent)
     |      Verifica se a relação de dependência entre 'tokenPai' e 'tokenFilho' é 'tipo_dep'.
     |      
     |      Args:
     |          tokenPai: governor
     |          tokenFilho: dependent
     |          tipo_dep: String que representa o tipo de dependência de tokenFilho para tokenPai.
     |  
     |  is_entity(self, token: spacy.tokens.token.Token, entidade: Literal['EVENT', 'TIMEX']) -> bool
     |      Verifica de o token é Event ou Timex conforme valor de 'entidade'
     |  
     |  is_equal(self, valor1, valor2) -> bool
     |      Verifica se dois valores são iguais.
     |      Se ambos foram 'NONE', não são iguais
     |  
     |  lengthInBetween(self, token1: spacy.tokens.token.Token, token2: spacy.tokens.token.Token) -> int
     |      Retorna distância entre os tokens.
     |  
     |  list_combination(self, lista: list) -> list
     |      Retorna lista com a combinação r = de 2 a len(lista)
     |  
     |  list_modal_verbs(self) -> list
     |      Retorna lista contendo os principais verbos modais em português
     |  
     |  list_temporal_signal(self, lemma=False) -> list
     |      Retorna lista contendo os sinais temporais
     |  
     |  list_to_str(self, lista, delimitador: str = ' ') -> str
     |      Converte 'lista' em string minúsculas.
     |      
     |      Args:
     |          lista: Pode ser: str, list, Doc, Span, Token
     |          delimitador: separador de cada palavra na string quando 'lista' for do tipo list.
     |      
     |      Return:
     |          Se 'lista' for do tipo list, converte-a em string minúsculas separadas por 'delimitador'. 
     |          Se 'lista' for do tipo Doc, Span ou Token, converte em string minúsculas.
     |          Se não, retorna string minúsculas.
     |  
     |  lista_palavras(self, texto: str) -> list
     |      Converte texto em uma lista de palavras
     |  
     |  mood(self, token: spacy.tokens.token.Token) -> str
     |      Retorna o modo verbal do token
     |  
     |  mood_check(self, token: spacy.tokens.token.Token, mood: list) -> bool
     |      Verifica se 'token' possui modo verbal 'mood'.
     |      
     |      Args:
     |          token: Token
     |          mood: str ou list
     |              mood válidos: 'Cnd', 'Imp', 'Ind', 'Sub'
     |  
     |  morph(self, token: spacy.tokens.token.Token, morph: tuple) -> bool
     |      Verifica se 'token' possui o elemento morph representado pela tupla (key, value) da análise morfológica.
     |      
     |      Args:
     |          token: Token
     |          morph: tupla (key, value), ex: ('Tense', 'Fut')
     |  
     |  nbor(self, token: spacy.tokens.token.Token, n: int) -> spacy.tokens.token.Token
     |  
     |  pos(self, token: spacy.tokens.token.Token, pos: list) -> bool
     |      Verifica se 'token' é uma das classes gramaticais da lista 'pos'.
     |      
     |      Args:
     |          token: Token
     |          pos: lista de classes gramaticais
     |  
     |  precededBy(self, token: spacy.tokens.token.Token, tipo: Literal['str', 'str_lemma', 'token', 'digito', 'pos', 'dep', 'morph'], valor=None, distancia='max') -> bool
     |      Procura elementos na sentença antes do 'token', conforme o 'tipo'.
     |      
     |      Args:
     |          token: objeto Token do spaCy.
     |          tipo: string e pode ser:
     |              str     -> verifica se existe palavras especificada em valor 
     |                      -> valor: str ou list; PODE SER OMITIDO.
     |              str_lemma -> verifica se existe palavras lematizadas especificada em valor 
     |                      -> valor: str ou list;
     |              token   -> verifica se 'token' precede o outro token especificado em valor 
     |                      -> valor: Token; PODE SER OMITIDO.
     |              digito  -> verifica se há dígitos ou pos = 'NUM' -> Não tem valor;
     |                      -> Se valor for informado, ele será a distância.
     |              pos     -> verifica se há a classe gramatical especificada em valor 
     |                      -> valor: list, ex: ['VERB', 'NOUM'];
     |              dep     -> verifica se há na árvore de dependência o elemento especificado em valor 
     |                      -> valor: list, ex: ['nsubj', 'nmod'];
     |              morph   -> verifica se há na análise morfológica o elemento especificado em valor. 
     |                      -> valor: tuple (key, value), ex: ('Tense', 'Fut'). Só é permitido uma tupla.
     |          valor: valor do elemento que será procurado, conforme o tipo.
     |          distancia:  Se inteiro, é quantidade de tokens antes de 'token' onde a pesquisa será realizada.
     |                      Se string, a pesquisa será realizada em todos os tokens que precedem 'token'.
     |  
     |  reichenbach_direct_modification(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Modificação direta: O TIMEX modifica diretamente o EVENT? (TIMEX é filho de EVENT?)
     |  
     |  reichenbach_temporal_mod_function(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Função de modificação temporal: Existe uma relação tmod (usaremos nmod) no caminho de dependência do EVENT ao TIMEX? 
     |      (verificar se nmod ajuda, não há tmod em português)
     |  
     |  reichenbach_tense(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> str
     |      Se POS do EVENT é VERB e TIMEX modifica o EVENT (Timex é filho) e TIMEX.tipo é DATE ou TIME então TIMEX = momento de referência (R).
     |      Se TIMEX = R, então o valor da feature assume os valores: anterior, simples ou posterior conforme o tempo verbal de EVENT equivalente na tabela de Reichenbach.
     |      O tipo da relação é entre E/R => tempos anteriores: E < R, tempos simples: E = R, tempos posteriores: E > R
     |  
     |  sameTense(self, token1: spacy.tokens.token.Token, token2: spacy.tokens.token.Token) -> bool
     |      Verificar se os tokens possuem o mesmo tempo verbal.
     |  
     |  search(self, palavras, frase, lemma: bool = False)
     |      Verifica se 'palavras' inteiras estão presentes em 'frase'.
     |      
     |      Args:
     |          palavras:   Pode ser list, str, Token ou expressão regular.
     |                      Pesquisa por palavras inteiras.
     |                      Regex gerado: (\W|^)(palavras_separadas_por_pipe)(\W|$)
     |          frase:  Texto onde as 'palavras' serão encontradas.
     |                  Pode ser list, str, Doc, Span e Token
     |          lemma:  Se True, lematiza palavras e frase.
     |  
     |  signal_closest_span(self, span: spacy.tokens.span.Span) -> spacy.tokens.token.Token
     |      Retorna o primeiro token que representa o sinal temporal, portanto, o mais próximo conforme contexto
     |  
     |  signal_context_token(self, token: spacy.tokens.token.Token, context: Literal['antes', 'depois']) -> spacy.tokens.token.Token
     |      Retorna o sinal temporal (Token) mais próximo do contexto de 'token'.
     |  
     |  signal_follow_event_ancestor_event(self, E: spacy.tokens.token.Token) -> bool
     |      Sinal que segue event domina sintaticamente o Event?
     |  
     |  signal_follow_event_ancestor_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue event domina sintaticamente o Timex?
     |  
     |  signal_follow_event_child_event(self, E: spacy.tokens.token.Token) -> bool
     |      Sinal que segue event é um filho do Event?
     |  
     |  signal_follow_event_child_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue event é um filho de Timex?
     |  
     |  signal_follow_event_comma_between_event(self, E: spacy.tokens.token.Token) -> bool
     |      Há uma vírgula entre o Sinal que segue event e o Event?
     |  
     |  signal_follow_event_comma_between_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Há uma vírgula entre o Sinal que segue event e o Timex?
     |  
     |  signal_follow_event_dep_advmod_advcl_event(self, E: spacy.tokens.token.Token) -> bool
     |      Sinal que segue event está diretamente relacionado ao Event com advmod ou advcl?
     |  
     |  signal_follow_event_dep_advmod_advcl_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue event está diretamente relacionado ao Timex com advmod ou advcl?
     |  
     |  signal_follow_event_dep_if_child_event(self, E: spacy.tokens.token.Token) -> bool
     |      DEP do Sinal que segue event, se ele for um filho do Event
     |  
     |  signal_follow_event_dep_if_child_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      DEP do Sinal que segue event, se ele for um filho do Timex
     |  
     |  signal_follow_event_distance_event(self, E: spacy.tokens.token.Token) -> str
     |      Distância do Sinal que segue event até Event.
     |  
     |  signal_follow_event_distance_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> str
     |      Distância do Sinal que segue event até Timex
     |  
     |  signal_follow_event_head_is_event(self, E: spacy.tokens.token.Token) -> bool
     |      O Sinal que segue event  modifica o Event diretamente? (ex: signal é um filho direto do Event?)
     |  
     |  signal_follow_event_head_is_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      O Sinal que segue event  modifica o Timex diretamente? (ex: signal é um filho direto do Timex?)
     |  
     |  signal_follow_event_is_event_head(self, E: spacy.tokens.token.Token) -> bool
     |      Sinal que segue event  é um pai direto do Event?
     |  
     |  signal_follow_event_is_timex3_head(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue event é um pai direto do Timex?
     |  
     |  signal_follow_event_pos(self, E: spacy.tokens.token.Token) -> str
     |      POS do Sinal que segue event.
     |  
     |  signal_follow_event_text(self, E: spacy.tokens.token.Token) -> str
     |      Texto do Sinal que segue event.
     |  
     |  signal_follow_timex3_ancestor_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue timex domina sintaticamente o Event?
     |  
     |  signal_follow_timex3_ancestor_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue timex domina sintaticamente o Timex?
     |  
     |  signal_follow_timex3_child_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue timex é um filho do Event?
     |  
     |  signal_follow_timex3_child_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue timex é um filho de Timex?
     |  
     |  signal_follow_timex3_comma_between_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Há uma vírgula entre o Sinal que segue timex e o Event?
     |  
     |  signal_follow_timex3_comma_between_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      Há uma vírgula entre o Sinal que segue timex e o Timex?
     |  
     |  signal_follow_timex3_dep_advmod_advcl_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue timex está diretamente relacionado ao Event com advmod ou advcl?
     |  
     |  signal_follow_timex3_dep_advmod_advcl_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue timex está diretamente relacionado ao Timex com advmod ou advcl?
     |  
     |  signal_follow_timex3_dep_if_child_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      DEP do Sinal que segue timex, se ele for um filho do Event
     |  
     |  signal_follow_timex3_dep_if_child_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      DEP do Sinal que segue timex, se ele for um filho do Timex
     |  
     |  signal_follow_timex3_distance_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> str
     |      Distância do Sinal que segue timex até Event
     |  
     |  signal_follow_timex3_distance_timex3(self, T: spacy.tokens.token.Token) -> str
     |      Distância do Sinal que segue timex até Timex
     |  
     |  signal_follow_timex3_head_is_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      O Sinal que segue timex  modifica o Event diretamente? (ex: signal é um filho direto do Event?)
     |  
     |  signal_follow_timex3_head_is_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      O Sinal que segue timex  modifica o Timex diretamente? (ex: signal é um filho direto do Timex?)
     |  
     |  signal_follow_timex3_is_event_head(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue timex  é um pai direto do Event?
     |  
     |  signal_follow_timex3_is_timex3_head(self, T: spacy.tokens.token.Token) -> bool
     |      Sinal que segue timex é um pai direto do Timex?
     |  
     |  signal_follow_timex3_pos(self, T: spacy.tokens.token.Token) -> str
     |      POS do Sinal que segue timex
     |  
     |  signal_follow_timex3_text(self, T: spacy.tokens.token.Token) -> str
     |      Texto do Sinal que segue timex
     |  
     |  signal_has_comma_token(self, token: spacy.tokens.token.Token, signal: spacy.tokens.token.Token) -> bool
     |      Verifica de há vírgula entre o 'token' e o sinal temporal
     |  
     |  signal_precede_event_ancestor_event(self, E: spacy.tokens.token.Token) -> bool
     |      Sinal que precede event domina sintaticamente o Event?
     |  
     |  signal_precede_event_ancestor_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede event domina sintaticamente o Timex?
     |  
     |  signal_precede_event_child_event(self, E: spacy.tokens.token.Token) -> bool
     |      Sinal que precede event é um filho do Event?
     |  
     |  signal_precede_event_child_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede event é um filho de Timex?
     |  
     |  signal_precede_event_comma_between_event(self, E: spacy.tokens.token.Token) -> bool
     |      Há uma vírgula entre o Sinal que precede event e o Event?
     |  
     |  signal_precede_event_comma_between_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Há uma vírgula entre o Sinal que precede event e o Timex?
     |  
     |  signal_precede_event_dep_advmod_advcl_event(self, E: spacy.tokens.token.Token) -> bool
     |      Sinal que precede event está diretamente relacionado ao Event com advmod ou advcl?
     |  
     |  signal_precede_event_dep_advmod_advcl_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede event está diretamente relacionado ao Timex com advmod ou advcl?
     |  
     |  signal_precede_event_dep_if_child_event(self, E: spacy.tokens.token.Token) -> bool
     |      DEP do Sinal que precede event, se ele for um filho do Event
     |  
     |  signal_precede_event_dep_if_child_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      DEP do Sinal que precede event, se ele for um filho do Timex
     |  
     |  signal_precede_event_distance_event(self, E: spacy.tokens.token.Token) -> str
     |      Distância do Sinal que precede event até Event
     |  
     |  signal_precede_event_distance_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> str
     |      Distância do Sinal que precede event até Timex
     |  
     |  signal_precede_event_head_is_event(self, E: spacy.tokens.token.Token) -> bool
     |      O Sinal que precede event  modifica o Event diretamente? (ex: signal é um filho direto do Event?)
     |  
     |  signal_precede_event_head_is_timex3(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      O Sinal que precede event  modifica o Timex diretamente? (ex: signal é um filho direto do Timex?)
     |  
     |  signal_precede_event_is_event_head(self, E: spacy.tokens.token.Token) -> bool
     |      Sinal que precede event  é um pai direto do Event?
     |  
     |  signal_precede_event_is_timex3_head(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede event é um pai direto do Timex?
     |  
     |  signal_precede_event_pos(self, E: spacy.tokens.token.Token) -> str
     |      POS do Sinal que precede event
     |  
     |  signal_precede_event_text(self, E: spacy.tokens.token.Token) -> str
     |      Texto do Sinal que precede event
     |  
     |  signal_precede_timex3_ancestor_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede timex domina sintaticamente o Event?
     |  
     |  signal_precede_timex3_ancestor_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede timex domina sintaticamente o Timex?
     |  
     |  signal_precede_timex3_child_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede timex é um filho do Event?
     |  
     |  signal_precede_timex3_child_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede timex é um filho de Timex?
     |  
     |  signal_precede_timex3_comma_between_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Há uma vírgula entre o Sinal que precede timex e o Event?
     |  
     |  signal_precede_timex3_comma_between_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      Há uma vírgula entre o Sinal que precede timex e o Timex?
     |  
     |  signal_precede_timex3_dep_advmod_advcl_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede timex está diretamente relacionado ao Event com advmod ou advcl?
     |  
     |  signal_precede_timex3_dep_advmod_advcl_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede timex está diretamente relacionado ao Timex com advmod ou advcl?
     |  
     |  signal_precede_timex3_dep_if_child_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      DEP do Sinal que precede timex, se ele for um filho do Event
     |  
     |  signal_precede_timex3_dep_if_child_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      DEP do Sinal que precede timex, se ele for um filho do Timex
     |  
     |  signal_precede_timex3_distance_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> str
     |      Distância do Sinal que precede timex até Event
     |  
     |  signal_precede_timex3_distance_timex3(self, T: spacy.tokens.token.Token) -> str
     |      Distância do Sinal que precede timex até Timex
     |  
     |  signal_precede_timex3_head_is_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      O Sinal que precede timex  modifica o Event diretamente? (ex: signal é um filho direto do Event?)
     |  
     |  signal_precede_timex3_head_is_timex3(self, T: spacy.tokens.token.Token) -> bool
     |      O Sinal que precede timex  modifica o Timex diretamente? (ex: signal é um filho direto do Timex?)
     |  
     |  signal_precede_timex3_is_event_head(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede timex  é um pai direto do Event?
     |  
     |  signal_precede_timex3_is_timex3_head(self, T: spacy.tokens.token.Token) -> bool
     |      Sinal que precede timex é um pai direto do Timex?
     |  
     |  signal_precede_timex3_pos(self, T: spacy.tokens.token.Token) -> str
     |      POS do Sinal que precede timex
     |  
     |  signal_precede_timex3_text(self, T: spacy.tokens.token.Token) -> str
     |      Texto do Sinal que precede timex
     |  
     |  signal_span_context(self, token: spacy.tokens.token.Token, context: Literal['antes', 'depois']) -> spacy.tokens.span.Span
     |      Retorna Span entre 'token' e a próxima entity (Event ou Timex), conforme o contexto.
     |  
     |  spanBetween(self, token1, token2) -> list
     |      Retorna pedaço da sentença com a quantidade de token 'distancia', conforme contexto (antes ou depois de 'token')
     |      
     |      Return: Lista de Span
     |  
     |  spanContext(self, token, distancia=5, contexto=None) -> list
     |      Retorna pedaço da sentença com a quantidade de token 'distancia', conforme contexto (antes ou depois de 'token')
     |      
     |      Return: Lista de Span
     |  
     |  spanFollow(self, token, distancia=5) -> spacy.tokens.span.Span
     |      Retorna pedaço da sentença do tamanho da 'distancia' depois de 'token'.
     |      
     |      Args:
     |          token: Objeto Token.
     |          distancia:  Se inteiro, é quantidade de tokens depois de 'token'.
     |                      Se string, a pesquisa será realizada em todos os tokens que vem depois de 'token'.
     |      
     |      Return: Span
     |  
     |  spanPrecede(self, token, distancia=5) -> spacy.tokens.span.Span
     |      Retorna pedaço da sentença do tamanho da 'distancia' antes de 'token'.
     |      
     |      Args:
     |          token: Objeto Token.
     |          distancia:  Se inteiro, é quantidade de tokens antes de 'token'.
     |                      Se string, a pesquisa será realizada em todos os tokens que vem antes de 'token'.
     |      
     |      Return: Span
     |  
     |  spanSomeMatch(self, span1: spacy.tokens.span.Span, span2: spacy.tokens.span.Span) -> bool
     |      Verifica se há interseção entre dois pedaços da sentenças.
     |      
     |      Args:
     |          span1: parte 1 da sentença.
     |          span2: parte 2 da sentenca.
     |  
     |  str_to_list(self, palavra: str) -> list
     |      Se 'palavra' for string, converte em lista unitária de string maiúsculas.
     |      Se não, converte em maiúsculas.
     |  
     |  t1AfterT2(self, t1: spacy.tokens.token.Token, t2: spacy.tokens.token.Token) -> bool
     |      Retorna True se t1 estiver posicionado na frase depois de t2. Senão retorna False.
     |  
     |  t1BeforeT2(self, t1: spacy.tokens.token.Token, t2: spacy.tokens.token.Token) -> bool
     |      Retorna True se t1 estiver posicionado na frase antes de t2. Senão retorna False.
     |  
     |  temporal_direction(self, word_pt: str) -> str
     |      Retorna o tipo de relação temporal mais provável para o evento composto por palavras presente no arquivo temporal_direction.txt
     |      Retirado do Apêndice III - LX-TimeAnalyzer
     |      
     |      Args:
     |          word_pt: Texto do evento em português
     |  
     |  temporal_signal_interssection_token(self, token: spacy.tokens.token.Token) -> str
     |      Retorna a intersecção de palavras consecutivas do token que estão na lista de temporal signals
     |  
     |  tense(self, token: spacy.tokens.token.Token) -> str
     |      Retorna o tempo verbal de 'token'
     |      Function tenseVerb(token, list_tense) pode ser usada para comparações
     |  
     |  tenseVerb(self, token: spacy.tokens.token.Token, tense: list) -> bool
     |      Verifica se 'token' possui tempo verbal 'tense'.
     |      
     |      Args:
     |          token: Token
     |          tense:  str ou list
     |                  tempo verbal válidos: 'FUT', 'IMP', 'PAST', 'PQP', 'PRES'
     |  
     |  tense_compound(self, token: spacy.tokens.token.Token) -> str
     |      Retorna o tempo verbal composto do token. Por enquanto, apenas do modo indicativo.
     |      
     |      Returns:
     |          . PRETPC    = pretérito perfeito composto do indicativo
     |          . FPRESC    = futuro do presente composto do indicativo
     |          . PRETMQPC  = pretérito mais-que-perfeito composto do indicativo
     |          . FPRETC    = futuro do pretérito composto do indicativo
     |  
     |  tense_compound_check(self, token: spacy.tokens.token.Token, tense_compound: list) -> bool
     |      Verifica se 'token' possui tempo verbal composto 'tense_compound'.
     |      
     |      Args:
     |          token: Token
     |          tense_compound: str ou list
     |              tempos verbais compostos: 'PRETPC', 'FPRESC', 'PRETMQPC', 'FPRETC'
     |  
     |  timex3_between_order(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Se há outro TIMEX entre EVENT e TIMEX
     |  
     |  timex3_between_quant(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> int
     |      A quantidade de TIMEX entre o par EVENT e TIMEX
     |  
     |  timex3_dep(self, T: spacy.tokens.token.Token)
     |      Relação final: DEP de TIMEX com seu pai
     |  
     |  timex3_gov_verb(self, T: spacy.tokens.token.Token) -> spacy.tokens.token.Token
     |      Verbo que rege o TIMEX
     |      NÃO USAR? BASEADO EM PALAVRAS
     |  
     |  timex3_gov_verb_tense(self, T: spacy.tokens.token.Token)
     |      Tense do verbo que rege o TIMEX
     |  
     |  timex3_head_is_root(self, T: spacy.tokens.token.Token) -> bool
     |      O Timex modifica diretamente a raiz? (ex: Timex é um filho direto da raiz?)
     |  
     |  timex3_head_pos(self, T: spacy.tokens.token.Token)
     |      POS do pai de TIMEX
     |  
     |  timex3_is_ancestor_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      TIMEX é a entidade regente na relação?
     |  
     |  timex3_is_child_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      TIMEX é a entidade dependente na relação?
     |  
     |  timex3_is_pai_direto_event(self, E: spacy.tokens.token.Token, T: spacy.tokens.token.Token) -> bool
     |      Verifica se TIMEX é pai direto de EVENT
     |  
     |  timex3_pos(self, T: spacy.tokens.token.Token)
     |      POS de TIMEX
     |  
     |  timex3_pos_token_1_follow(self, T: spacy.tokens.token.Token)
     |      POS do 1º token depois o TIMEX
     |  
     |  timex3_pos_token_1_precede(self, T: spacy.tokens.token.Token)
     |      POS do 1º token antes o TIMEX
     |  
     |  timex3_pos_token_2_follow(self, T: spacy.tokens.token.Token)
     |      POS do 2º token depois o TIMEX
     |  
     |  timex3_pos_token_2_precede(self, T: spacy.tokens.token.Token)
     |      POS do 2º token antes o TIMEX
     |  
     |  timex3_pos_token_3_follow(self, T: spacy.tokens.token.Token)
     |      POS do 3º token depois o TIMEX
     |  
     |  timex3_pos_token_3_precede(self, T: spacy.tokens.token.Token)
     |      POS do 3º token antes o TIMEX
     |  
     |  timex3_preposition_gov(self, T: spacy.tokens.token.Token) -> spacy.tokens.token.Token
     |      preposição que rege sintaticamente o TIMEX
     |  
     |  timex3_preposition_precede(self, T: spacy.tokens.token.Token)
     |      Preposição antes do TIMEX, ou NONE se essa palavra não for uma preposição.
     |  
     |  timex3_relevant_lemmas(self, T: spacy.tokens.token.Token)
     |      Se o lemma do TIMEX está contida em LISTA de palavras que têm algum conteúdo temporal.
     |  
     |  timex3_root(self, T: spacy.tokens.token.Token) -> bool
     |      TIMEX é a raiz da sentença? (bool)
     |  
     |  timex3_temporalfunction(self, T: spacy.tokens.token.Token)
     |      temporalFunction de TIMEX (bool)
     |  
     |  timex3_type(self, T: spacy.tokens.token.Token)
     |      Tipo de TIMEX
     |  
     |  tipo(self, token: spacy.tokens.token.Token, tipo: list) -> bool
     |      Verifica se 'token' é uma dos tipos de timex3 da lista 'tipo'.
     |      
     |      Args:
     |          token: Token
     |          tipo: lista de tipos de timex
     |  
     |  token1_token2_dep(self, t1: spacy.tokens.token.Token, t2: spacy.tokens.token.Token) -> str
     |      DEP entre token1/token2 e token2/token1, se houver
     |  
     |  token_resource(self, token: spacy.tokens.token.Token, resource: Literal['class', 'pos', 'tense', 'polarity', 'aspect', 'dep', 'type', 'temporalfunction'])
     |  
     |  verbGerundio(self, token: spacy.tokens.token.Token) -> bool
     |      Verifica se é gerúndio.
     |  
     |  verbform(self, token: spacy.tokens.token.Token) -> str
     |      Retorna a forma verbal. Ex: Inf, Fin.
     |  
     |  verbform_check(self, token: spacy.tokens.token.Token, verbform: list) -> bool
     |      Verifica se 'token' possui modo verbal 'mood'.
     |      
     |      Args:
     |          token: Token
     |          mood: str ou list
     |              mood válidos: 'Fin', 'Ger', 'Inf', 'Part'
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    


```python
tb.tr.setRules_start('../dataset/dataset_features')
help(tb.tr.setRules)
```

    O arquivo ../dataset/dataset_features possui mais de um ponto. Certifique-se que haja uma extensão.
    Dataset ../dataset/dataset_features.parquet carregado com sucesso.
    Help on SetRules in module parse.ParseTimebankPT object:
    
    class SetRules(builtins.object)
     |  SetRules(params_functions: dict = None)
     |  
     |  Implementa estrutura de dados de Conjunto de Regras.
     |  Classe independente que processa apenas regras originadas do dataset de features (TimebankPT.FeaturesToDataset).
     |  Para que as regras da classe sejam processadas, é necessários atribuí-las a TemporalRelation.rules.
     |  Por exemplo: TemporalRelation.rules = temporalRelation.SetRules.rules
     |  
     |  args:
     |      class_features: String que representa a classe onde estão as funções que implementam as features.
     |              As funções estão na classe TemporalRelation.f, geralmente instanciada como tr.f ou self.f
     |          
     |      params_functions: Dicionário de funções e seus parâmetros. Deve ser passado ao instanciar a classe.
     |              Ex: {'event_class': ['E'], 'event_timex3_dep': ['E', 'T']}
     |  
     |  Methods defined here:
     |  
     |  __init__(self, params_functions: dict = None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __repr__(self)
     |      Return repr(self).
     |  
     |  __str__(self)
     |      Return str(self).
     |  
     |  add_rule(self, relType: str, cod_regra: float = None, ordem: int = None, origem: str = '', acuracia: float = 0, acertos: int = 0, acionamentos: int = 0)
     |      Adiciona novas regras ao conjunto de regras
     |  
     |  add_rule_class_default(self, class_default: str)
     |      Adicionar regra para classe default em TemporalRelation.SetRules.rules
     |  
     |  add_setRules(self, nome_arquivo: str, algoritmo: str = '', reset_cod_regra: bool = False, reset_order: bool = False)
     |      Adiciona conjunto de regras à instancia atual.
     |      
     |      Args:
     |          nome_arquivo:   Nome do arquivo de origem das regras.
     |                          Pode ser arquivos .pickle salvos pelo método 'save_rules(nome_arquivo)'
     |                          Pode ser arquivos .txt contendo regras geradas por um dos algoritmos válido.
     |                          Pode ser arquivos .txt salvo pelo método 'save_rules_to_txt(nome_arquivo)'
     |                              neste caso, o algoritmo deve ser 'my_rules'.
     |      
     |          algoritmo:  Obrigatório se for arquivo .txt ou qualquer outro arquivo texto contendo regras.
     |                      Serve para identificar a origem da regras para utilizar o parser apropriado.
     |                      Valores válidos 'jrip_weka', 'cn2_orange', 'CN2' e 'my_rules'. Sendo 'cn2_orange' = 'CN2'.
     |                      Não é usado quando o arquivo é .pickle
     |      
     |          reset_cod_regra: Se algoritmo for 'my_rules', então reset_cod_regra determina se o código da regra que está sendo 
     |                      adicionada será levado em consideração. Se True, o código da regra do arquivo será desconsiderado.
     |          
     |          reset_order: Se algoritmo for 'my_rules', então reset_order determina se a ordem da regra será levado em consideração. 
     |                      Se True, a ordem da regra do arquivo será desconsiderado.
     |  
     |  add_setRules_ojb(self, algoritmo: str, setRules_obj, relType: str = '', verbosy: bool = True)
     |      Adicionar regras no setRules atual conforme algoritmo gerador de regras.
     |      
     |      Args:
     |          algoritmo: são válidos 'RIPPER', 'CBA', 'IDS', 'MY_RULES', 'SELF'.
     |      
     |          setRules_obj: Objeto que representa o conjunto de regras gerado por 'algoritmo'.
     |              Se RIPPER: classificador.ruleset_.rules
     |              Se CBA: classificador.rules
     |              Se IDS: classificador.clf.rules
     |              Se MY_RULES: TemporalRelation.rules
     |              Se SELF: TemporalRelation.SetRules.rules
     |      
     |          relType: Se o algoritmo for RIPPLE, é obrigatório informar o tipo da relação temporal das regras. 
     |                  Este algoritmo é de classificação binária, só gera regras para uma classe de cada vez.
     |  
     |  check_filename(self, filename: str, check_if_exist: bool = False) -> str
     |      Retorna nome do arquivo com a extensão padrão .pickle se não for informada e pode verificar se o arquivo existe.
     |  
     |  clear(self)
     |      Apaga todas as regras do conjunto de regras para iniciar um novo conjunto de regras.
     |  
     |  copy(self)
     |  
     |  filter_clear(self)
     |      Limpa todos os filtros aplicados à regras
     |  
     |  filter_rules(self, field: str, value: list, verbosy=True) -> List[list]
     |      Filtra regras conforme o campo e valor especificado
     |      
     |      Args:
     |          field = campo da regra que deseja filtrar. 
     |                  Valores válidos: ['cod_regra', 'relType', 'ordem', 'origem', 'acuracia', 'acertos', 'acionamentos']
     |          value = pode ser lista ou valor único. 
     |                  Se field for acuracia, acertos ou acionamentos, exibe as regras cuja valor é maior que 'value'.
     |                  Se field for ordem, exibe as regras cuja ordem é menor que 'value'.
     |  
     |  get_max_cod_rule(self) -> int
     |  
     |  get_max_order_rule(self, origem: str) -> int
     |      Retorna dicionário com a ordem máxima de cada origem
     |  
     |  get_rule(self, cod_regra: list) -> list
     |      Retorna a regra ou lista de regras conforme 'cod_regra' especificado.
     |      
     |      Args:
     |          cod_regra: Pode ser numérico ou lista de números.
     |              Se for informado um número retorna uma regra do tipo Rule, ou Rule vazio se 'cod_regra' não existir
     |              Se for informado uma lista de números, retorna uma lista de regras.
     |  
     |  has_rule_class_default(self) -> str
     |      Verifica se há regras com classe default.
     |      Se houver, retorna a regra de classe default.
     |  
     |  remove_rule(self, field: str, value, verbosy=True) -> bool
     |      Remove regras do conjunto de regras conforme campo e valor especificado.
     |      
     |      Args:
     |          field: Especifica o campo que deseja filtrar o valor.
     |              Campos válidos: ['cod_regra', 'relType', 'ordem', 'origem', 'acuracia', 'acertos', 'acionamentos']
     |          valor: valor do campo 'field' que deseja remover.
     |              Pode ser lista ou valor único. 
     |              Se field for acuracia, acertos, acionamentos, remove os valores menores que 'value'.
     |              Se field for ordem, remove 'ordem' maiores que 'value'.
     |  
     |  remove_rule_class_default(self)
     |      Remove todas as regras de classe default, se houver.
     |  
     |  remove_rules_duplicate(self)
     |      Remove regras duplicadas. A primeira da lista de duplicados permanece.
     |      Considera apenas a lista de predicados para comparar a regra.
     |      Não leva em conta a ordem dos predicados.
     |      
     |      Return:
     |          lista de regras duplicadas excluídas
     |  
     |  save_object(self, name_object, nome_arquivo: str)
     |      Salva name_object em arquivo físico .pickle.
     |      
     |      Args:
     |          name_object: objeto serializável.
     |          nome_arquivo: arquivo .pickle
     |  
     |  save_rules(self, nome_arquivo: str)
     |      Salva objeto SetRules.rules, que representa o conjunto de regras atual, em arquivo físico .pickle.
     |  
     |  save_rules_to_txt(self, nome_arquivo: str)
     |      Salva conjunto de regras em formato de texto em arquivo.
     |      Para carregar, use o método add_setRules(nome_arquivo, 'my_rules'). 
     |          O parametro 'algoritmo' deve ser 'my_rules'.
     |  
     |  sort_clear(self)
     |      Retorna à ordem inicial. Limpa ordenação
     |  
     |  sort_rules(self, fields: list, reverse=False)
     |      Ordena as regras conforme os campos especificados. 
     |      
     |      Args:
     |          field = pode ser um campo ou lista de campos da regra que deseja filtrar.
     |                  valores válidos: 'cod_regra', 'relType', 'ordem', 'origem', 'acuracia', 'acertos', 'acionamentos'
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties defined here:
     |  
     |  df_features_counts
     |      Retorna DataFrame contendo a quantidade de ocorrências de cada feature no conjunto de regras por tipo de relação temporal.
     |      Se o setRules conter mais de uma origem, será gravada apenas a primeira, que deverá ser ignorada.
     |  
     |  df_rules
     |      Exibe as regras ativas em formato de tabela.
     |  
     |  is_filter
     |  
     |  rules_str
     |      Retorna o conjunto de regras em formato de lista de listas
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  class_features
     |  
     |  rule
     |      Retorna objeto Rule
     |  
     |  rules
     |      Retorna conjunto de regras em formato de lista de objetos Rule
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  Rule = <class 'parse.ParseTimebankPT.SetRules.Rule'>
     |      Estrutura de dados de uma Regra.
     |  
     |  
     |  params_functions = {'event_between_order': ['E', 'T'], 'event_class': ...
    
    

### Pipeline spaCy para TimebankPT


```python
from parse.ParseTimebankPT import PipeTimebankPT
help(PipeTimebankPT)
```

    Help on class PipeTimebankPT in module parse.ParseTimebankPT:
    
    class PipeTimebankPT(builtins.object)
     |  PipeTimebankPT(tb_dict: dict, nlp: spacy.language.Language)
     |  
     |  Implementa pipeline do spaCy que processa anotações do corpus TimeBankPT trazendo para a estrutura do spaCy as tag EVENT e TIMEX3.
     |  Adicionar antes do pipe nlp.add_pipe("merge_entities")
     |  
     |  Arg:
     |      tb_dict: Dicionário de dados contendo dados do TimebankPT necessários para este pipeline do spaCy.
     |  
     |  Return: 
     |      Objeto Doc do spaCy processado contendo as tag do TimebankPT (EVENT e TIMEX3).
     |  
     |  Methods defined here:
     |  
     |  __call__(self, doc: spacy.tokens.doc.Doc) -> spacy.tokens.doc.Doc
     |      Retorna Doc processado contendo informações do TimebankPT.
     |  
     |  __init__(self, tb_dict: dict, nlp: spacy.language.Language)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    
