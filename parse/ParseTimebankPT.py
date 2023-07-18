'''
Autor: Dárcio Santos Rocha® 
e-mail: darcio.rocha@ufba.br
Mestrando em Ciências da Computação - UFBA
Agosto/2021

Importa tags do corpus TimebankPT para a estrutura do spaCy.
Identifica tipos de relações temporais.
'''

#----------------------------------------------------------
#INSTALAÇÕES

#conda install -c conda-forge spacy
#conda install tabulate
#python -m spacy download pt_core_news_lg 
# ou  
#spacy.cli.download("pt_core_news_lg")
#!python -m spacy info
#----------------------------------------------------------


'''
TODO:
    . Verificar a importância de noun_chunks: [(p.text, p.label_) for p in tb.doc[0].noun_chunks]
    
    . verificar se \b é melhor que \W na func search -> reg = "(\W|^)(" + tratar_palavras_busca(palavras) + ")(\W|$)"
    
    . The last feature in this category is the Temporal Relation between the Document Creation Time and the Temporal Expression in the target sentence. 
    . The value of this feature could be “greater than”, “less than”, “equal”, or “none”. 
    
    . ele primeiro ordena expressões temporais anotadas de acordo com seu valor normalizado (por exemplo, a data 1989-09-29  é ordenada como precedendo  1989-10-02). 
        Ou seja, exploramos as anotações timex3 a fim de enriquecer o conjunto de relações temporais com as quais trabalhamos, e mais especificamente fazemos uso do atributo de valor dos elementos TIMEX3.  

. TRABALHOS FUTUROS
    . implementar junção das relações frequentemente classificadas como incorretas com o dataset de features, verificar padrões
    . Experimentar utilizar Tempo Verbal e POS das anotações do corpus ao invés do spaCy.
    . Falar sobre as relações que nenhum algoritmo acertou (tb.tr.relations_incorrect_class)
    . Falar sobre as 40 regras duplicadas geradas por mais de um algoritmo, 2 ou mais algoritmos geraram a mesma regra.
    . Rotular 55% das relações event-time não rotuladas no Corpus
    . Implementar task B: relações entre eventos e DT
    . implementar taxk C: relações entre pares de eventos

. CONCLUSÕES PARA A DISSERTAÇÃO:
    . Regras geradas por CN2 obteve alta cobertura, o fechamento temporal não geral nenhum par event-time que já não houvesse sido predito pelas regras.
    . As regras RIPPER e IDS geraram muitas regras repetidas durante os loops

TODO IDS: 
    . Cria regras de alta qualidade, mas baixa cobertura.
    . [ok] Retirar instâncias cobertas, e aplicar algoritmo nas instâncias descobertas 
    . [ok] Adicionar novo conjunto de regras abaixo do conjunto existente
    . [ok] Testar outros algoritmos de optimização além do SLS, testar o DSL
    . [ok] Testar outros lambdas (dinâmico)
    . [ok] Gerar novamente regras com classes raras

TODO RIPPER:
    . [ok] Semelhante ao IDS, repetir processamento nas instâncias não cobertas
    . experimentar em ordem de relType, classe menor frequente na frente, mais frequente no final
    . [ok] tentar gerar regras para as classes: BEFORE-OR-OVERLAP, OVERLAP-OR-AFTER e VAGUE
    . tentar GridSearch em cada iteração

TODO CN2:
    . [ok] Testar retirando classificação default da classe majoritária
    . [ok] Submeter as instância não classificadas ao mesmo algoritmos (PIOROU)
    


#SALVAR SAIDA DO DISPLACY EM ARQUIVO HTML
html = spacy.displacy.render(doc, style='ent', jupyter=False, page=True)
f = open("teste.html", "w", encoding='utf-8')
f.write(html)
f.close()
'''

verbosy = False
import time
if verbosy: ini_total = time.time()
if verbosy: ini = time.time()
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
#from pandas import Series
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)
#pd.set_option('max_rows', 40)
#pd.set_option("colheader_justify", "left")  # Não funcionou
if verbosy: print((time.time() - ini) / 60, 'min -> numpy, pandas')

if verbosy: ini = time.time()
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
if verbosy: print((time.time() - ini) / 60, 'min -> plots')

if verbosy: ini = time.time()
import spacy
from spacy.language import Language
from spacy.tokens import Token, Span, Doc
if verbosy: print((time.time() - ini) / 60, 'min -> spaCy')

if verbosy: ini = time.time()
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pickle  #save model

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2 #, mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

#Visualização Decision Tree
from IPython.display import Image  
from sklearn.tree import export_graphviz
#!pip install pydotplus
#!conda install graphviz
import pydotplus
from tabulate import tabulate
#!pip install treelib
from treelib import Node, Tree
if verbosy: print((time.time() - ini) / 60, 'min -> sklearn')

if verbosy: ini = time.time()
import os
import io
import math
import random
import xml.etree.ElementTree as et 
import glob
import re
import html
from itertools import product, combinations, groupby
from collections import defaultdict, Counter
import types
from typing import List, Literal, Dict, Tuple, Union
import inspect
import copy
from IPython.display import display
if verbosy: print((time.time() - ini) / 60, 'min -> outros módulos')
if verbosy: print((time.time() - ini_total) / 60, 'min -> Tempo total')

class Functions:
    '''
    Funções genéricas utilizadas em outras classes
    
    '''
    def explicar_spacy(self, elemento):
        '''
        Retorna descrição explicativa sobre elementos POS e DEP do spaCy.
        '''
        __explain_outros = {'pass': 'passive voice', 'foreign': 'foreign name', 'name': 'proper name'}
        
        if not elemento:
            return ''
        
        __list_elem = elemento.split(':')
        __list_elem_explain = [spacy.explain(e) for e in elemento.split(':')]
        
        if len(__list_elem) > 1:
            if __list_elem_explain[1] == None:
                if __list_elem[1] in __explain_outros.keys():
                    __list_elem_explain[1] = __explain_outros[__list_elem[1]]
                else:
                    __list_elem_explain[1] = 'None'

        if __list_elem_explain[0] == None:
            return ''
        
        return ', '.join(__list_elem_explain)
    
    
    def nbor(self, token: Token, n: int) -> Token:
        '''
        Retorna o token n vizinho de 'token'.
        n negativo: vizinho a esquerda. 
        n positivos: vizinho a direita.
        '''
        min = -token.i
        max = len(token.doc) - token.i - 1
        n = 0 if n < min else n
        n = 0 if n > max else n
        return None if n == 0 else token.nbor(n)
    
    #-------------------------------
    def __get_class(self, tipo, obj = None):
        if obj:
            self = obj
        classname = self.__class__
        components = dir(classname)
        features = list(filter(lambda attr: (type(getattr(classname, attr)) is tipo) and attr[0] != '_', components))
        return features

    def __get_properties(self, obj = None):
        '''
        Retorna propriedades de um objeto.
        '''
        return self.__get_class(property, obj)
    
    def __get_functions(self, obj = None):
        '''
        Retorna funções de um objeto.
        '''
        return self.__get_class(types.FunctionType, obj)
        
    def __get_types(self, obj = None):
        '''
        Retorna tipos de um objeto.
        '''
        return self.__get_class(type, obj)
    
    def get_class_list(self, obj = None):
        '''
        Retorna lista com as propriedades, funções e tipos presentes no objeto atual.
        '''
        return {'properties': self.__get_properties(obj), 'functions': self.__get_functions(obj), 'types': self.__get_types(obj)}
    #----------------------------------------------

    


# '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
#  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
#  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
#  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
#  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
#  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
#  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
# ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #

#---------------------------------------------------------------------
#     CLASSE SIGLAS
#--------------------------------------------------------------------

class Siglas(Functions):
    '''
    Fornece estrutura de dados contendo as siglas de POS, DEP, CLASSE DE EVENT e TIPO DO TIMEX3
    '''
    def __init__(self):
        '''
        pos    : classe gramatical
        dep    : árvore de dependência
        classe : classe do evento <EVENT> anotada no corpus
        tipo   : tipo da expressão temporal <TIME3> anotada no corpus
        
        '''
        self.siglas = {
            'pos': {0:'adj', 1:'adp', 2:'adv', 3:'aux', 4:'cconj', 5:'det', 6:'intj', 7:'noun', 8:'num', 9:'pron', 10:'propn', 11:'punct', 12:'sconj', 13:'sym', 14:'verb', 15:'x'},
            'dep': {0:'acl', 1:'acl:relcl', 2:'advcl', 3:'advmod', 4:'amod', 5:'appos', 6:'aux', 7:'aux:pass', 8:'case', 9:'cc', 10:'ccomp', 11:'compound', 12:'conj', 13:'cop', 14:'csubj', 15:'dep', 16:'det', 17:'discourse', 18:'expl', 19:'fixed', 20:'flat', 21:'flat:foreign', 22:'flat:name', 23:'iobj', 24:'mark', 25:'nmod', 26:'nsubj', 27:'nsubj:pass', 28:'nummod', 29:'obj', 30:'obl', 31:'obl:agent', 32:'parataxis', 33:'punct', 34:'root', 35:'xcomp'},
            'classe': {0:'aspectual', 1:'i_action', 2:'i_state', 3:'occurrence', 4:'perception', 5:'reporting', 6:'state'},
            'morph_key': {0:'case', 1:'definite', 2:'degree', 3:'foreign', 4:'gender', 5:'numtype', 6:'polarity', 7:'prontype', 8:'reflex', 9:'tense', 10:'mood', 11:'verbform', 12:'person', 13:'number', 14:'voice'},
            'tipo': {0: 'date', 1:'time', 2:'duration', 3:'set'},
        }
        
        #Features em ordem de frequência para testar o formato da árvore
        #self.siglas = {
        #    'pos': {0: 'verb', 1: 'noun', 2: 'adp', 3: 'adj', 4: 'adv', 5: 'num', 6: 'aux', 7: 'cconj', 8: 'det', 9: 'intj', 10: 'pron', 11: 'propn', 12: 'punct', 13: 'sconj', 14: 'sym', 15: 'x'},
        #    'dep': {0: 'root', 1: 'obj', 2: 'ccomp', 3: 'advcl', 4: 'xcomp', 5: 'obl', 6: 'nmod', 7: 'acl:relcl', 8: 'conj', 9: 'acl', 10: 'nsubj', 11: 'advmod', 12: 'amod', 13: 'appos', 14: 'aux', 15: 'aux:pass', 16: 'case', 17: 'cc', 18: 'compound', 19: 'cop', 20: 'csubj', 21: 'dep', 22: 'det', 23: 'discourse', 24: 'expl', 25: 'fixed', 26: 'flat', 27: 'flat:foreign', 28: 'flat:name', 29: 'iobj', 30: 'mark', 31: 'nsubj:pass', 32: 'nummod', 33: 'obl:agent', 34: 'parataxis', 35: 'punct'},
        #    'classe': {0: 'occurrence', 1: 'reporting', 2: 'state', 3: 'i_state', 4: 'i_action', 5: 'aspectual', 6: 'perception'},
        #}
    
    def __valida_tipo_siglas(self, tipo_siglas: str) -> bool:
        '''
        Verifica se o tipo da sigla é válido
        
        Args:
            tipo_siglas: 'pos', 'dep', 'classe', morph_key e tipo (timex3)
            
        Return:
            bool
            
        '''
        if tipo_siglas in self.get_tipo_siglas():
            return True
        else:
            print("Tipo de siglas válidos: " + str(self.get_tipo_siglas()))
            return False

    def __list_tipo_siglas(self, tipo_siglas: str, valor: float, sinal: str):
        
        if not self.__valida_tipo_siglas(tipo_siglas):
            return []
        
        lista = []
        siglas = self.siglas[tipo_siglas]
        
        for k in siglas:
            if eval(str(k) + sinal + str(valor)):
                lista.append(siglas[k].upper())
        return lista
        
    
    def __le(self, tipo_siglas: str, valor: float):
        '''
        le = menor ou igual a
        '''
        return self.__list_tipo_siglas(tipo_siglas, valor, '<=')
    
    def __gt(self, tipo_siglas: str, valor: float):
        '''
        gt = maior que
        '''
        return self.__list_tipo_siglas(tipo_siglas, valor, '>')
    
    
    def pos_le(self, valor: float):
        '''
        Return POS tag menor ou igual a 'valor' da classe Siglas().
        '''
        return self.__le('pos', valor)
    
    def dep_le(self, valor: float):
        '''
        Return DEP menor ou igual a 'valor' da classe Siglas().
        '''
        return self.__le('dep', valor)

    def classe_le(self, valor: float):
        '''
        Return Classe do evento menor ou igual a 'valor' da classe Siglas().
        '''
        return self.__le('classe', valor)
    
    
    def pos_gt(self, valor: float):
        '''
        Return POS tag maior que 'valor' da classe Siglas().
        '''
        return self.__gt('pos', valor)
    
    def dep_gt(self, valor: float):
        '''
        Return DEP maior que 'valor' da classe Siglas().
        '''
        return self.__gt('dep', valor)

    def classe_gt(self, valor: float):
        '''
        Return Classe do evento maior que 'valor' da classe Siglas().
        '''
        return self.__gt('classe', valor)
    

    def get_key(self, valor: str, tipo_siglas: str):
        '''
        Obtém a chave do dicionário 'tipo_siglas' passando o valor como argumento.
        
        Args:
            valor: valor da chave que deseja buscar
            tipo_siglas: 'pos', 'dep', 'classe', morph_key e tipo (timex3)
        '''
        valor = valor.lower()
        tipo_siglas = tipo_siglas.lower()
        if not self.__valida_tipo_siglas(tipo_siglas):
            return False

        for key, value in self.siglas[tipo_siglas].items():
            if valor == value:
                return key
        return False

    def get_value(self, key: int, tipo_siglas: str):
        '''
        Obtém o valor da chave do dicionário 'tipo_siglas' passando a chave como argumento.
        
        Args:
            key: número inteiro que representa a chave
            tipo_siglas: pode ser 'pos', 'dep', 'classe', morph_key e tipo (timex3)
        '''
        tipo_siglas = tipo_siglas.lower()
        if not self.__valida_tipo_siglas(tipo_siglas):
            return False
        if key not in self.siglas[tipo_siglas].keys():
            return False
        return self.siglas[tipo_siglas][key]
    
    def get_tipo_siglas(self) -> list:
        '''
        Retorna os tipos de dicionários de siglas disponíveis
        '''
        return list(self.siglas.keys())
    
    def get_desc(self, valor: str):
        '''
        Retorna a descrição de cada DEP ou POS disponíveis no spaCy.
        
        Args:
            valor: pode ser valores de POS e DEP.
        '''
        return self.explicar_spacy(valor)
    
    def print_dep(self):
        '''
        Imprime todos DEP na tela com suas descrições.
        '''
        siglas_desc = {}
        
        for key, sigla in self.siglas['dep'].items():
            siglas_desc[sigla.upper()] = self.explicar_spacy(sigla)
                
        return siglas_desc
        
    def print_pos(self):
        '''
        Imprime todos POS na tela e suas descrições.
        '''
        siglas_desc = {}
        
        for key, sigla in self.siglas['pos'].items():
            siglas_desc[key] = sigla.upper() + ': ' + self.explicar_spacy(sigla.upper())
                
        return siglas_desc
    
    def print_classe(self):
        '''
        Imprime todas as classes de EVENT.
        '''
        siglas_desc = {}
        
        for key, sigla in self.siglas['classe'].items():
            siglas_desc[key] = sigla.upper()
                
        return siglas_desc
    
    def get_list(self, tipo_siglas: str):
        '''
        Retorna 'tipo_siglas' em formato de lista.
        
        Args:
            tipo_siglas: 'pos', 'dep', 'classe', morph_key e tipo (timex3)
            
        '''
        if not self.__valida_tipo_siglas(tipo_siglas):
            return False
        
        lista = self.siglas[tipo_siglas].values()
        return list(map(str.upper, lista))
        
#---------------------------------------------------------------------
#     FIM CLASSE SIGLAS
#--------------------------------------------------------------------

    
#=========================================================================================================================
#=========================================================================================================================    
# '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
#  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
#  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
#  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
#  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
#  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
#  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
# ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #
#=========================================================================================================================
#=========================================================================================================================
    
class TimebankPT(Functions):
    '''
    Importa dados do corpus TimebankPT e fornece vários métodos para manipular o conteúdo do corpus.
    Se existir o arquivo 'dataset/corpus.pickle', o carregamento rápido do corpus é acionado.
    Se não existir, o corpus existente na pasta 'path_tml' é processado, o carregamento é mais lento.
    O arquivo 'dataset/corpus.pickle' é salvo pelo método: 
        TimebankPT.df.save_corpus(tb.path_corpus_pickle).
    
    Se existir o arquivo 'dataset/data_pipe_tb.pickle', o carregamento rápido dos dados do pipeline é acionado.
    Se não existir, os dados para o pipeline são processados na inicialização de Timebank.df.
    O arquivo 'dataset/data_pipe_tb.pickle' é salvo pelo método: 
        TimebankPT.save_data_pipe_tb(tb.path_data_pipe_pickle).
    
    Args:
        path_tml: caminho do corpus TimebankPT no formato: 'c:\diretorio\*\*.txt'
        add_timebank: adiciona tags (EVENT, TIMEX3 e TLINK) do corpus TimebankPT ao pipeline do spaCy. Default é True
        dev: Se True, os dados de treino são divididos em 'train' e 'train_test'. 'test' não deve ser utilizado.
            Se False, todo dado de treino é 'train' e 'test' é utilizado.
        ignore_load_corpus: Se True, carrega o corpus previamente salvo.
            Se False, processa o carragamento do curpus a partir dos arquivo .tml
    '''
    
    def __init__(self, path_tml, add_timebank = True, lang = 'pt', dev: bool = False, ignore_load_corpus = False):
        
        self.path_corpus_pickle = 'dataset/corpus.pickle'
        self.path_data_pipe_pickle = 'dataset/data_pipe_tb.pickle'
        self.ignore_load_corpus = ignore_load_corpus

        if not os.path.exists(self.path_corpus_pickle):
            if not os.path.exists(os.path.dirname(path_tml).replace('*', '')):
                print('ERROR: Path dos arquivos .tml não existe.\n' + path_tml)
                return
        
        self.path_tml = path_tml
        
        self.dev = dev
        
        #O self do parâmetro é para passar a class TimebankPT para Df, Print e MyTlink
        self.df = self.Df(self)
        self.print = self.Print(self)
        self.my_tlink = self.MyTlink(self)
        
        #TemporalRelation deve ser anterior a FeaturesToDataset
        self.tr = TemporalRelation(self) 
        self.features = FeaturesToDataset(self)
        
        self.siglas = Siglas()

        self.__dados_pipe = None
        self.__id_sentenca = []
        self.__sentenca_texto = []
        self.__nome_doc = None
        self.__dct_doc = None
        self.__lingua = 'PORTUGUÊS'
        #objeto Doc do spaCy
        self.__doc = None
        
        #Carrega linguagem pt do spaCy
        if lang == 'pt':
            self.nlp = spacy.load('pt_core_news_lg')
        else:
            self.nlp = spacy.load('en_core_web_sm')
        
        if add_timebank and lang == 'pt':
            #Adiciona o pipe_timebank antes do merge_entities
            self.add_pipe_timebank()
        
        #Junta em um unico token entidades com mais de um token
        if not self.nlp.has_pipe("merge_entities"):
            self.nlp.add_pipe("merge_entities")

        #Verificar se este pipe é útil para alguma situação. Junta substantivos em pequenas frases.
        #self.nlp.add_pipe("merge_noun_chunks")
        
        #define variável com nome da linguagem utilizada
        if lang == 'pt':
            self.__lingua = 'PORTUGUÊS'
        else:
            self.__lingua = 'INGLÊS'

        
    def __str__(self):
        '''
        Exibe as quantidades dos objetos do TimebankPT
        
        '''
        return "QUANTIDADES:\n   . Sentenças: " + str(self.df.quant_sentenca_total) + \
                "\n   . Documentos: " + str(self.df.quant_doc_total) + \
                "\n   . Events: " + str(self.df.quant_event_total) + \
                "\n   . Timex3: " + str(self.df.quant_timex3_total) + \
                "\n   . TLink: " + str(self.df.quant_tlink_total ) + \
                "\n\nSENTENÇAS: " + str(self.df.sentenca_completo.value_counts('train_test'))
    
    
    #----------------------------------------
    #-----  PÚBLICO  -  class Timebank  -----
    #----------------------------------------
    
    def print_pipes(self):
        '''
        Imprime a sequência dos pipelines executados
        '''
        print('SEQUÊNCIA PIPELINE: ' + self.__lingua)
        for i, pipe in enumerate(self.nlp.pipe_names):
            print('   ' + str(i + 1) + ' -> ' + pipe)
        print('\n')

    @property
    def dados_pipe(self):
        return self.__dados_pipe
    
    def add_pipe_timebank(self):
        '''
        Adiciona o pipe que adiciona tags dos timebankPT ao Doc no spaCy
        
        '''
        if not self.nlp.has_pipe("pipe_timebankpt"):
            insere_antes = 'merge_entities'
            if not self.nlp.has_pipe(insere_antes):
                self.nlp.add_pipe(insere_antes)
            
            #Recupera dados do TimebankPT para serem fornecidos ao pipeline 
            if os.path.exists(self.path_data_pipe_pickle) and not self.ignore_load_corpus:
                print(f"Arquivo '{self.path_data_pipe_pickle}' encontrado. \nAcionado carregamento rápido dos dados do TimebankPT para pipeline do spaCy.")
                self.__dados_pipe = self.load_data_pipe_tb(self.path_data_pipe_pickle)
            else:
                if not self.__dados_pipe:
                    self.__dados_pipe = self.df.dados_pipe
            
            self.nlp.add_pipe("pipe_timebankpt", before = insere_antes, config={'tb_dict': self.__dados_pipe})
            
            #atualiza Doc com tags do timebankpt (EVENT e TIMEX3)
            self.__set_doc()
            
        else:
            print('pipe_timebankpt já está adicionado.')
        
        self.print_pipes()
        
    def remove_pipe_timebank(self):
        '''
        Remove o pipe_timabankpt. Retira as tag dos timabankpt (EVENT e TIMEX3)
        '''
        if self.nlp.has_pipe("pipe_timebankpt"):
            self.nlp.remove_pipe('pipe_timebankpt')
            
            #atualiza Doc retirando as tags do timebankpt (EVENT e TIMEX3)
            self.__set_doc()
            
        else:
            print('pipe_timebankpt ainda não foi adicionado.')
            
        self.print_pipes()


    def save_data_pipe_tb(self, nome_arquivo: str):
        '''
        Salva dados do corpus carregado em arquivo físico.
        '''
        nome_arquivo = self.check_filename(nome_arquivo, 'pickle')
        try:
            if not self.__dados_pipe:
                self.__dados_pipe = self.df.dados_pipe
            with open(nome_arquivo, 'wb') as arquivo:
                pickle.dump(self.__dados_pipe, arquivo)
        except Exception as e:
            print(f'Erro ao salvar dados do pipe no arquivo {nome_arquivo}. \nERRO: {e}')
        else:
            print(f'Dados do pipe salvo com sucesso no arquivo {nome_arquivo}.')

    def load_data_pipe_tb(self, nome_arquivo: str) -> dict:
        '''
        Retorna objeto que representa os dados do pipe salvo pelo método 'save_data_pipe_tb(nome_arquivo)'.
        Os objetos retornados são: event, timex3, tlink, sentenca, documento
        '''
        nome_arquivo = self.check_filename(nome_arquivo, 'pickle', check_if_exist=True)
        with open(nome_arquivo, 'rb') as arquivo:
            return pickle.load(arquivo)
    
    #---------------------------------------
    # DESCRITORES
    def set_id_sentenca(self, *id_sentenca):
        '''
        Atribui as id_sentenca para as instâncias da classe e atribui valores a campos que dependem de id_sentenca
        
        Args:
            id_sentenca: Lista de id_sentença. O id_senteca não conta nos arquivos TimeML, foram criados na função timeml_to_df para facilitar o acesso.
            id_sentenca pode ser vários inteiros, várias strings, lista de ambos ou strings separadas por virgulas.
            
        '''
        lista_id_sentenca = []
        lista_nome_doc = []
        lista_dct_doc = []
        lista_sentenca_texto = []
        
        id_sentencas = self.trata_lista(*id_sentenca)
        
        if id_sentencas:
            for id_sent in id_sentencas:
                #Considera apenas id_sentenca existentes
                if self.__is_id_sentenca(id_sent):
                    lista_id_sentenca.append(id_sent)

                    nome_doc = self.get_nome_doc(id_sent)
                    lista_nome_doc.append(nome_doc)

                    lista_dct_doc.append(self.get_dct_doc(nome_doc))
                    lista_sentenca_texto.extend(self.get_sentenca_texto(id_sent))
                    
        
        self.__id_sentenca = lista_id_sentenca

        self.__nome_doc = lista_nome_doc
        self.__dct_doc = lista_dct_doc
        self.__sentenca_texto = lista_sentenca_texto
        
        self.__set_doc()
        self.df.atualizar_filtros()

        
    def set_sentenca_texto(self, sentenca_texto):
        '''
        Permite atribuir sentenças que não estão no TimebankPT para submetê-las ao pipeline do spaCy.
        Caso a sentenca passada exista no TimebankPT, atribui a id_sentenca à classe com set_id_sentenca()
        
        Args:
            sentenca_texto: Lista de sentenças ou sentença única.
        
        '''
        self.id_sentenca = []
        self.__nome_doc = []
        self.__dct_doc = []
        
        if type(sentenca_texto) == int:
            sentenca_texto = str(sentenca_texto)
            
        #se sentenca_texto for str, converte em lista de um elemento
        #if type(sentenca_texto) == str:
        #    tmp = []
        #    tmp.append(sentenca_texto)
        #    sentenca_texto = tmp
        
        #Verifica se a sentença existe
        achou_id_sentenca = self.get_id_sentenca(sentenca_texto)
        if achou_id_sentenca: #se sim, passa ela para a instância da classe
            self.id_sentenca = achou_id_sentenca
        else:  # se não, o pipe processa a sentença passada e atualiza o objeto Doc
            self.__sentenca_texto = sentenca_texto
            self.__set_doc()
            self.df.atualizar_filtros()
    
    
    def __set_doc(self):
        '''
        Atribui objeto Doc do spaCy baseado na id_sentenca passada para a instancia da classe
        
        Return:
            list de Doc
        '''
        if len(self.id_sentenca) == 0:
            self.__doc = []
        
        sentenca_texto = self.sentenca_texto    #self.sentenca_texto é atribuido em set_id_sentenca() ou em set_sentenca_texto()
        if type(sentenca_texto) == list:
            self.__doc = list(self.nlp.pipe(sentenca_texto))
        elif type(sentenca_texto) == str:
            doc = self.nlp(sentenca_texto)
            lista = []
            lista.append(doc)
            doc = lista
            self.__doc = doc
        else:
            print('ERRO: sentenca_texto deve ser do tipo list ou str')
    
    # FIM DESCRITORES
    #---------------------------------------
    
    
    def get_id_sentenca(self, texto_sentenca = None):
        '''
        Retorna as id_sentenca setadas em set_id_sentenca() se texto_sentenca não for informado
        
        Se texto_sentenca for informado:      
            Busca a id_sentenca correspondente à texto_sentenca no TimeBankPt, e a retorna.

        Args: 
            texto_sentenca: Texto da sentença a ser procurada em TimeBankPt
            
        Return:
            list id_sentenca
            
        '''
        if texto_sentenca:
            id_sentenca_list = []
            if (type(texto_sentenca) == list):
                texto_sentenca = texto_sentenca[0]
            if (type(texto_sentenca) == str):
                #escapa as aspas duplas do texto, uma vez que a string texto_sentenca está entre aspas duplas
                texto_sentenca = texto_sentenca.replace('"','\\"')
                df = self.df.sentenca_completo.query('sentenca == "' + texto_sentenca + '"')['isentenca']
                if not df.empty:
                    id_sentenca_list.append(df.tolist()[0])

            return id_sentenca_list
        else:
            return self.__id_sentenca
    
    def get_id_sentenca_unica(self):
        '''
        Retorna lista com a primeira id_sentenca da lista de sentenças setadas em set_id_sentenca()
        
        '''
        id_sentenca = self.id_sentenca
        
        if len(id_sentenca) == 0:
            return []
        
        id_sentenca_list = []
        id_sentenca_list.append(id_sentenca[0])
        id_sentenca = id_sentenca_list
        return id_sentenca
    
    def get_eventID(self, id_sentenca = None):
        '''
        Retorna lista de eventID de id_sentenca
        '''
        if id_sentenca is None:
            id_sentenca = self.id_sentenca
        else:
            id_sentenca = self.trata_lista(id_sentenca)
            
        return self.df.event_completo[['eid', 'isentenca']].query("isentenca == " + str(id_sentenca))['eid'].values.tolist()
    
    def get_nome_documento(self, id_sentenca = None):
        '''
        Retorna nome do documento de 'id_sentenca'.
        
        Args:
            id_sentenca: pode ser list ou int
                se não for informado, retorna o nome do documento de id_sentenca da classe.
                se lista, retorno o documento do primeiro id_sentenca
        '''
        if id_sentenca is None:
            id_sentenca = self.id_sentenca
        else:
            id_sentenca = self.trata_lista(id_sentenca)
            
        return self.df.sentenca_completo.query("isentenca == " + str(id_sentenca[0]))['doc'].values[0]
    
    def get_id_sentenca_do_doc(self, id: str, nome_documento: str) -> int:
        '''
        Retorna id_sentenca do doc e eid/tid.
        
        Args:
            id: pode ser o id do evento ou do timex3
            nome_documento: nome do arquivo que representa um documento do corpus
        '''
        df_event  = self.df.event_completo[['eid', 'doc', 'isentenca']].rename(columns={'eid':'id'}).query("doc == '" + nome_documento + "'")
        df_timex3 = self.df.timex3_completo[['tid', 'doc', 'isentenca']].rename(columns={'tid':'id'}).query("doc == '" + nome_documento + "'")
        df_timex3 = df_timex3[~df_timex3['isentenca'].isna()]
        df_event_timex3 = pd.concat([df_event, df_timex3])

        df_event_timex3 = df_event_timex3.query("id == '" + id + "'")

        if df_event_timex3.empty:
            return ''

        return int(df_event_timex3['isentenca'].values.tolist()[0])

    
    def get_id_sentencas_doc(self):
        '''
        Retorna lista de todas id_sentenca do primeiro documento da lista de documentos atual.
        O nome do documento é o nome do arquivo do TimeBankPT.

        '''
        
        #if self.df.sentenca_doc.empty:
        #    print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
        #    return
            
        return self.df.sentenca_doc['isentenca'].tolist()
    
    def get_id_sentencas_dep(self):
        '''
        Retorna também lista de id_sentenca dependentes da sentenca do documento atual.
        
        '''
        if not self.id_sentenca:
            print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
            return
        
        recursivo_atual = self.df.recursivo
        if recursivo_atual == False:
            self.df.recursivo = True
            
        df = self.df.sentenca['isentenca'].tolist()
        
        if self.df.recursivo != recursivo_atual:
            self.df.recursivo = recursivo_atual
        
        return df

    def query_filtro_task(self, task: str):
        '''
        Retorna query que filtra sentenças conforme task
        
        '''
        if not task:
            print('Task ainda não foi informada.')
            return 
        
        task = task.upper()
        if task == 'A':
            q = "and isentenca == isentenca_rt"
        elif task == 'B':
            q = ''
        elif task == 'C':
            q = "and isentenca != isentenca_re"
        else:
            print("ERROR: Task '{0}' inválida.".format(task))
            return ''
        return "task == '" + task + "' " + q

    
    def id_sentencas_task(self, task: str):
        '''
        Retorna id_sentenca contempladas pela tarefa 'task'
        
        Args:
            task: filtrar conforme task
            filter 
        '''
        if not task:
            print('Task ainda não foi informada.')
            return 
        
        query = self.query_filtro_task(task)
        return self.df.tlink_join_completo.query(query)['isentenca'].unique().tolist()
    
    
    def __get_sentenca_texto_helper(self, id_sentenca = None, todas_do_documento = False, com_tags = False):
        '''
        Retorna lista contendo texto da sentença, conforme id_sentenca setadas em set_id_sentenca().

        Args:
            id_sentenca: sobrepõe id_sentenca atribuída em set_id_sentenca()
            
            todas_do_documento: Se True, retorna o texto de todas as sentenças do documento de get_id_sentenca(). Se False, retorna apenas o texto da sentença de get_id_sentenca()
                Se True e id_sentenca for uma lista com sentenças pertencentes a mais de um documento, considera apenas as sentenças do documento do primeiro item da lista id_sentenca

            com_tags: Se True, retorna o texto da sentença com as tags TimeML. Se False, retorna texto puro.

        Return:
            lista de sentenças
            
        '''
        lista_sentenca = []
        
        #DOCUMENTO
        if todas_do_documento:
            id_sentenca = self.id_sentencas_doc
        else:
            if id_sentenca is None:
                id_sentenca = self.id_sentenca
            else:
                id_sentenca = self.trata_lista(id_sentenca)

        if com_tags:
            campo_retorno = 'sentenca_tag'
        else:
            campo_retorno = 'sentenca'
        
        if (id_sentenca is None) or len(id_sentenca) == 0:
            return []
        
        df = self.df.sentenca_completo.query("isentenca in " + str(id_sentenca))[campo_retorno]
        if not df.empty:
            lista_sentenca = df.tolist()
        
        return lista_sentenca
    
    def get_sentenca_texto(self, id_sentenca = None):
        '''
        Retorna lista contendo texto da sentença com as tag TimeML, conforme id_sentenca setadas em set_id_sentenca() ou informado no parâmetro id_sentenca
        
        '''
        if id_sentenca is None:
            return self.__sentenca_texto
        else:
            return self.__get_sentenca_texto_helper(id_sentenca, todas_do_documento = False, com_tags = False)
    
    
    def get_sentenca_texto_tag(self, id_sentenca = None):
        '''
        Retorna lista contendo texto da sentença com as tag TimeML, conforme id_sentenca setadas em set_id_sentenca().
        
        '''
        return self.__get_sentenca_texto_helper(id_sentenca, todas_do_documento = False, com_tags = True)
        
    def get_sentenca_texto_doc(self):
        '''
        Retorna lista de texto de todas as sentenças do documento da primeira sentença de get_id_sentenca().
        '''
        return self.__get_sentenca_texto_helper(todas_do_documento = True, com_tags = False)
    
        
    def pesquisa_sentenca_texto(self, lista_termos = '', formato_dataframe = False):
        '''
        Retorna DataFrame com resultado pesquisa dos termos
        
        Args:
            lista_termos: lista de palavras a ser pesquisada em sentenças
            
            formato_dataframe: se True, retorna o dataframe filtrado por lista_termos, se não, retorna lista de sentenças que atendem ao critério de pesquisa
        '''
        df = self.df.sentenca_completo
        lista_termos = self.trata_lista(lista_termos, tipo_lista = str)
        df_filtrado = df[df['sentenca'].str.contains('|'.join(map(re.escape, lista_termos)))]
        
        if formato_dataframe:
            return df_filtrado
        else:
            return df_filtrado['sentenca'].tolist()
        
    def pesquisa_id_sentenca(self, lista_termos, formato_dataframe = False):
        '''
        Retorna DataFrame com resultado pesquisa dos termos
        
        Args:
            lista_termos: lista de palavras a ser pesquisada em sentenças
            
            formato_dataframe: se True, retorna o dataframe filtrado por lista_termos, se não, retorna lista de id_sentenca que atendem ao critério de pesquisa
        '''
        df = self.pesquisa_sentenca_texto(lista_termos, formato_dataframe = True)
        
        if formato_dataframe:
            return df
        else:
            return df['isentenca'].tolist()
    
            
    def get_doc(self, id_sentenca = None):
        '''
        Retorna lista de objetos Doc do spaCy
        
        Return:
            Lista de Docs
            
        '''
        if id_sentenca is None:
            doc = self.__doc
        else:
            id_sentenca_anterior = self.id_sentenca
            self.id_sentenca = id_sentenca
            doc = self.__doc
            
        if id_sentenca is not None:
            self.id_sentenca = id_sentenca_anterior
            
        return doc
    
    def get_doc_unico(self):
        '''
        Retorna o primeiro Doc da lista de Docs (self.doc).
        
        Return:
            Doc
        '''
        doc = None
        
        if not self.doc:
            print('É necessário atribuir id_sentenca à instancia da class TimebankPT.')
            return
        
        if type(self.doc) == list:
            doc = self.doc[0]
            
        if type(doc) == Doc:
            return doc
        
    
    def get_doc_root(self, doc: Doc = None) -> list:
        '''
        Retorna lista de roots do Doc.
        '''
        if doc:
            if type(doc) == list:
                doc = doc[0]
            if type(doc) == Doc:
                doc_atual = doc
            else:
                print('ERROR: Argumento não é Doc do spaCy válido.')
                return
        else:
            if not self.doc_unico:
                return 
            
            doc_atual = self.doc_unico
            
        roots = [token for token in doc_atual if token.dep_ == 'ROOT']
        return roots
        
        #if len(roots) == 1:
        #    return roots[0]
        #elif 'VERB' in roots:
        #    return roots[0]
        #else:
        #    return roots[0]
        
    
    def eh_nome_doc(self, nome_doc: str):
        '''
        Verifica se o nome_doc pertence aos arquivos do corpus TimeBankPT
        
        Args:
            nome_doc: Nome de arquivo do TimebankPT
            
        '''
        if type(nome_doc) == list:
            nome_doc = nome_doc[0]
        
        if type(nome_doc) == Doc:
            nome_doc = nome_doc._.nome
        
        if len([path for path in self.df.lista_arquivos() if path.find(nome_doc.strip()) >=0]) > 0:  #se encontrar
            return True
        else:
            return False
    
    
    def get_nome_doc(self, id_sentenca = None):
        '''
        Retorna lista com nome do documento da id_sentença setada em set_id_sentenca() ou str da id_sentenca do parametro
        
        Args:
            id_sentenca: se informado, sobrepõe id_sentenca atribuido em set_id_sentenca()
        
        Return:
            Nome de documentos. É o nome do arquivo do TimebankPT
            
        '''
        if id_sentenca is None:
            if self.id_sentenca:
                return  self.__nome_doc
            else:
                #print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
                return []
        else:
            return self.df.sentenca_completo.query("isentenca == " + str(id_sentenca) + "")['doc'].tolist()[0]
    
    def get_nome_doc_unico(self):
        '''
        Retorna primeiro nome de documento da lista de nomes de documentos (self.nome_doc).
        
        Return:
            string
            
        '''
        if self.id_sentenca:
            return self.__nome_doc[0]
        else:
            #print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
            return ''
    
    
    def get_dct_doc_helper(self, nome_doc = None, retorno = 'dct'):
        '''
        Retorna lista com dados da Data de Criação do Documento(DCT) da id_sentença setada em set_id_sentenca() ou de 'nome_doc' do parametro
        
        Args:
            nome_doc: se informado, sobrepõe as id_sentenca atribuída em set_id_sentenca()
            retorno: pode retornar 
                'dct': value da data de criação do documento
                'type': um dos tipos de TIMEX3 (DATE, TIME, DURATION, SET)
                'tid': id do TIMEX3 que é o DCT
        '''
        if retorno not in ['dct', 'type', 'tid']:
            print("ERROR: parâmetro de retorno válido: 'dct', 'type', 'tid'")
            return []
            
        if nome_doc is None:
            if self.id_sentenca:
                return self.__dct_doc
            else:
                #print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
                return []
        else:
            return self.df.documento_completo.query("doc == '" + str(nome_doc) + "'")[retorno].tolist()[0]
        
    def get_dct_doc(self, nome_doc = None):
        '''
        Retorna o value da Data de Criação do Documento(DCT) da id_sentença setada em set_id_sentenca() ou str da id_sentenca do parametro
        
        Args:
            nome_doc: se informado, sobrepõe as id_sentenca atribuída em set_id_sentenca()
        
        Return:
            value do DCT
        '''
        return self.get_dct_doc_helper(nome_doc, 'dct')
    
    def get_dct_doc_tid(self, nome_doc = None):
        '''
        Retorna o tid da Data de Criação do Documento(DCT) da id_sentença setada em set_id_sentenca() ou str da id_sentenca do parametro
        
        Args:
            nome_doc: se informado, sobrepõe as id_sentenca atribuída em set_id_sentenca()
        
        Return:
            tid do DCT
        '''
        return self.get_dct_doc_helper(nome_doc, 'tid')
    
    def get_dct_doc_type(self, nome_doc = None):
        '''
        Retorna o type da Data de Criação do Documento(DCT) da id_sentença setada em set_id_sentenca() ou str da id_sentenca do parametro
        
        Args:
            nome_doc: se informado, sobrepõe as id_sentenca atribuída em set_id_sentenca()
        
        Return:
            type do DCT
        '''
        return self.get_dct_doc_helper(nome_doc, 'type')
    
        
    def get_train_test(self, nome_doc: str) -> Literal["train", "train_test", "test"]:
        '''
        Retorna qual o grupo de desenvolvimento que o documento pertence.
        
        Args:
            nome_doc: nome do documento
        
        Return:
            train:      Se é de treino.
            train_test: Se é de teste para o conjunto de treino (Dev).
            test:       Se é de teste global. Utilizado apenas no trabalho final.
            
        '''
        if type(nome_doc) != str:
            print('ERROR: nome_doc deve ser do tipo string.')
            return ''
        
        result = self.df.documento_completo.query("doc == '" + nome_doc + "'")['train_test'].tolist()[0]
        
        if not self.dev:
            if result == 'train_test':
                result = 'train'
                
        return result


    def trata_lista(self, *dados, tipo_lista = int):
        '''
        Retorna dados convertido em lista de strings.

        Args:
            dados: Pode vários inteiros, várias strings, lista de ambos ou strings separadas por virgulas.
            
        '''
        def add_com_delimitador(dado, delimitador = ','):
            if (delimitador in dado):
                lista.extend(dado.split(delimitador))
            else:
                lista.append(dado)

        lista = []
        delimitador = ','

        if not dados:
            return []

        for dado in dados:
            
            if type(dado) == list:
                list_string = list(map(str, dado))
                for item in list_string:
                    if item != '':
                        add_com_delimitador(item, delimitador)

            if type(dado) in [int, float]:
                lista.append(str(dado))

            if type(dado) == str:
                if dado != '':
                    add_com_delimitador(dado, delimitador)

        #remove os espaços de cada item da lista se tipo_lista = str
        func = None
        if tipo_lista == int:
            func = int
        elif tipo_lista == str:
            func = str #.strip
        
        return list(map(func, lista))

    
    def __is_id_sentenca(self, id_sentenca = None):
        '''
        Retorna True se id_sentenca existe em TimeBankPT.

        Args:
            id_sentenca: ID da sentença armazenada no DataFrame. O ID da sentença não conta nos arquivos TimeML, foram criados na função timeml_to_df.
                    Se for passada um lista de id_sentenca, será conciderada apenas o primeiro item da lista.
        '''
        if id_sentenca is None:
            return False
        
        df_id_sentenca = self.df.sentenca_completo.query("isentenca == " + str(id_sentenca) + "")
        return not df_id_sentenca.empty


    def check_filename(self, filename: str, extensao: str = '', check_if_exist: bool = False) -> str:
        '''
        Retorna nome do arquivo com a extensão padrão .'extensão' se não for informada em filename. Pode verificar se o arquivo existe conforme 'check_if_exist'.

        Args:
            filename: nome do arquivo. Extensão no nome do arquivo tem prevalência sobre 'extensão'.
            extensao: extensão padrão do arquivo
            check_if_exist: Se True, verifica se o arquivo existe
        '''
        if not filename:
            raise ValueError(f"É necessário informar o nome do arquivo.")
        filename, ext = os.path.splitext(filename.strip())
        
        if filename.count('.') >= 2:
            print(f'O arquivo {filename} possui mais de um ponto. Certifique-se que haja uma extensão.')
        
        if not ext:
            if not extensao:
                raise ValueError(f"É necessário informar a extensão do arquivo.")
            ext = '.' + extensao
        filename =  filename.strip() + ext
        
        if check_if_exist:
            if not os.path.exists(filename):
                raise ValueError(f"O arquivo '{filename}' não existe.")
                
        return filename
    
        
    #---------------------------------------------
    #-----  PROPRIEDADES  -  class Timebank  -----
    #---------------------------------------------

    id_sentenca        = property(get_id_sentenca, set_id_sentenca)
    sentenca_texto     = property(get_sentenca_texto, set_sentenca_texto)
    sentenca_texto_tag = property(get_sentenca_texto_tag)
    sentenca_texto_doc = property(get_sentenca_texto_doc)
    id_sentenca_unica  = property(get_id_sentenca_unica)
    id_sentencas_doc   = property(get_id_sentencas_doc)
    id_sentencas_dep   = property(get_id_sentencas_dep)
    nome_doc           = property(get_nome_doc)
    nome_doc_unico     = property(get_nome_doc_unico)
    dct_doc            = property(get_dct_doc)
    doc                = property(get_doc)
    doc_unico          = property(get_doc_unico)
    doc_root           = property(get_doc_root)

    
    
    
    
    #=========================================================================================================================
    # '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
    #  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
    #  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
    #  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
    #  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
    #  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
    #  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
    # ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #
    #=========================================================================================================================
    
    
    #------------------------------------------------
    #-----  CLASS DATAFRAMES  -  class Df  ----------
    #------------------------------------------------
    
    class Df():
        '''
        Cria DataFrame para os diversos elementos do corpus TimebankPT: EVENT, TIMEX3, TLINK, Sentenças, Nome do Documento e Data de Criação do Documento (DCT)
        
        Args:
            tb: Recebe instancia da classe TimebankPT
        '''

        def __init__(self, tb: 'TimebankPT'):  #o 'tb' é recebido no self do parâmetro da instanciação da class, ex: df = Df(self)
            '''
            Processa carregamento dos dados do corpus a partir dos arquivos .tml para Dataframes.
            '''
            self.__tb = tb
            self.__recursivo = False
            self.__load_df_full()
            
        
        @property
        def recursivo(self):
            '''
            Propriedade booleana da class que indica se as sentenças dependentes serão também exibidas.
            
            '''
            return self.__recursivo
        
        @recursivo.setter
        def recursivo(self, recursivo: bool):
            '''
            Altera o comportamento da exibição dos DataFrames.
            
            Args:
                recursivo: Se True, é exibido também as sentenças que possuem relações temporais dependentes.
                            Se False, exibe apenas as sentenças informadas.
            
            '''
            recursivo_atual = self.__recursivo
            
            if recursivo_atual != recursivo:
                self.__recursivo = recursivo
                self.atualizar_filtros()
        
        
        def __load_df_full(self):
            '''
            Carrega os DataFrames completos, sem filtros.
            '''
            if os.path.exists(self.__tb.path_corpus_pickle) and not self.__tb.ignore_load_corpus:
                print(f"Arquivo '{self.__tb.path_corpus_pickle}' encontrado. Acionado carregamento rápido do corpus.")
                self.__df_event, self.__df_timex3, self.__df_tlink, self.__df_sentenca, self.__df_doc = self.load_corpus(self.__tb.path_corpus_pickle)
            else:
                print(f"Não foi encontrado o arquivo '{self.__tb.path_corpus_pickle}'. \nProcessando corpus de {self.__tb.path_tml}.")
                self.__df_event, self.__df_timex3, self.__df_tlink, self.__df_sentenca, self.__df_doc = self.__timeml_to_df()
            
        def __check_load_df(self):
            '''Verifica se o Corpus foi carregado'''
            if self.event_completo.empty or self.timex3_completo.empty or self.tlink_completo.empty or self.sentenca_completo.empty or self.documento_completo.empty:
                return False
            else:
                return True
            
        def save_corpus(self, nome_arquivo: str):
            '''
            Salva dados do corpus em arquivo físico (.pickle).
            '''
            def carrega_corpus():
                if not self.__check_load_df():
                    raise ValueError('Corpus não carregado.')
                return self.event_completo, self.timex3_completo, self.tlink_completo, self.sentenca_completo, self.documento_completo
            
            nome_arquivo = self.__tb.check_filename(nome_arquivo, 'pickle')
            try:
                corpus = carrega_corpus()
                with open(nome_arquivo, 'wb') as arquivo:
                    pickle.dump(corpus, arquivo)
            except Exception as e:
                print(f'Erro ao salvar corpus no arquivo {nome_arquivo}. \nERRO: {e}')
            else:
                print(f'Corpus salvo com sucesso no arquivo {nome_arquivo}.')

        def load_corpus(self, nome_arquivo: str):
            '''
            Retorna objeto que representa o corpus salvo pelo método 'save_corpus(nome_arquivo)'.
            Os objetos retornados são: event, timex3, tlink, sentenca, documento

            nome_arquivo: caminho e nome do arquivo formato .pickle
            '''
            nome_arquivo = self.__tb.check_filename(nome_arquivo, 'pickle', check_if_exist=True)
            with open(nome_arquivo, 'rb') as arquivo:
                return pickle.load(arquivo)


        def atualizar_filtros(self):
            '''
            Carrega os DataFrames filtrados conforme parâmetros.
            É chamada sempre que uma propriedade da classe é alterado, por exemplo, set_id_sentenca, set_sentenca_texto, recursivo.
            
            '''
            if not self.__tb.id_sentenca:
                #print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
                return
                
            self.__df_event_filter, self.__df_timex3_filter, self.__df_tlink_filter, self.__df_sentenca_filter, self.__df_doc_filter = self.__timeml_to_df_filter(self.recursivo)

        def lista_arquivos(self):
            '''
            Retorna lista de arquivos do path.

            '''
            return list(glob.glob(self.__tb.path_tml))

        
        @property
        def event(self):
            '''
            Retorna DataFrame contendo todos atributos de EVENT, porém apenas das sentenças informada para a class.
            
            '''
            if not self.__tb.id_sentenca:
                print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
                return self.event_completo.query("doc == ''")
            
            return self.__df_event_filter
        
        @property
        def event_completo(self):
            '''
            Retorna DataFrame contendo todos atributos de EVENT de todas as sentenças do corpus.
            
            '''
            return self.__df_event
        
        @property
        def event_doc(self):
            '''
            Retorna DataFrame contendo todos atributos de EVENT, porém apenas as sentenças do documento atual.
            
            '''
            if not self.__tb.nome_doc:
                print('O atributo nome_doc ainda não foi definido. Necessário informar id_sentenca para instancia da class TimebankPT.')
                return self.event_completo.query("doc == ''")
            
            return self.event_completo.query("doc == '" + self.__tb.nome_doc_unico + "'")
        
        @property
        def timex3(self):
            '''
            Retorna DataFrame contendo todos atributos de TIMEX3, porém apenas das sentenças informada para a class.
            
            '''
            if not self.__tb.id_sentenca:
                print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
                return self.timex3_completo.query("doc == ''")
            
            return self.__df_timex3_filter
        
        @property
        def timex3_completo(self):
            '''
            Retorna DataFrame contendo todos atributos de TIMEX3 de todas as sentenças do corpus.
            
            '''
            return self.__df_timex3
        
        @property
        def timex3_doc(self):
            '''
            Retorna DataFrame contendo todos atributos de TIMEX3, porém apenas as sentenças do documento atual.
            
            '''
            if not self.__tb.nome_doc:
                print('O atributo nome_doc ainda não foi definido. Necessário informar id_sentenca para instancia da class TimebankPT.')
                return self.timex3_completo.query("doc == ''")
            
            return self.timex3_completo.query("doc == '" + self.__tb.nome_doc_unico + "'")
        
        
        @property
        def sentenca(self):
            '''
            Retorna DataFrame contendo atributos das sentenças informadas para a class.
            
            '''
            if not self.__tb.id_sentenca:
                print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
                return self.sentenca_completo.query("doc == ''")
            
            return self.__df_sentenca_filter
        
        @property
        def sentenca_completo(self):
            '''
            Retorna DataFrame contendo atributos de todas as sentenças do corpus.
            
            '''
            return self.__df_sentenca
        
        @property
        def sentenca_doc(self):
            '''
            Retorna DataFrame contendo todos atributos da sentença, porém apenas as sentenças do documento atual.
            
            '''
            if not self.__tb.nome_doc:
                print('O atributo nome_doc ainda não foi definido. Necessário informar id_sentenca para instancia da class TimebankPT.')
                return self.sentenca_completo.query("doc == ''")
            
            return self.sentenca_completo.query("doc == '" + self.__tb.nome_doc_unico + "'")
        
        @property
        def documento(self):
            '''
            Retorna DataFrame contendo atributos do documento da sentença informada para a class.
            
            '''
            if not self.__tb.id_sentenca:
                print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
                return self.documento_completo.query("doc == ''")
            
            return self.__df_doc_filter
        
        @property
        def documento_completo(self):
            '''
            Retorna DataFrame contendo atributos de todos os documento do corpus.
            
            '''
            return self.__df_doc
        
        
        #TLINK
        @property
        def tlink(self):
            '''
            Retorna DataFrame contendo todos atributos de TLINK, porém apenas das sentenças informadas para a class.
            
            '''
            if not self.__tb.id_sentenca:
                print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
                return self.tlink_completo.query("doc == ''")
            
            return self.__df_tlink_filter
        
        @property
        def tlink_completo(self):
            '''
            Retorna DataFrame contendo todos atributos de TLINK de todas as sentenças do corpus.
            
            '''
            return self.__df_tlink
        
        @property
        def tlink_doc(self):
            '''
            Retorna DataFrame contendo todos atributos de TLINK, porém apenas os registros do documento atual.
            
            '''
            if not self.__tb.nome_doc:
                print('O atributo nome_doc ainda não foi definido. Necessário informar id_sentenca para instancia da class TimebankPT.')
                return self.tlink_completo.query("doc == ''")
            
            return self.tlink_completo.query("doc == '" + self.__tb.nome_doc_unico + "'")
        
        @property
        def tlink_join(self):
            '''
            Retorna DataFrame de TLink unido com os campos das chaves estrangeira.
            Dar uma visão mais global dos campos de TLink.
            
            '''
            id_sentenca = []
            if self.recursivo:
                id_sentenca = self.__tb.id_sentencas_dep
            else:
                id_sentenca = self.__tb.id_sentenca
            
            if not self.__tb.id_sentenca:
                print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
                return self.tlink_join_completo.query("isentenca == ''")
            
            return self.tlink_join_completo.query(f"isentenca in {str(id_sentenca)} or isentenca_re in {str(id_sentenca)}")  #incluir isentenca_re -> útil para task C
        
        @property
        def tlink_join_doc(self):
            '''
            Retorna DataFrame contendo todos atributos de TLINK e suas chaves estrangeiras, porém apenas os registros do documento atual.
            
            '''
            if not self.__tb.nome_doc:
                print('O atributo nome_doc ainda não foi definido. Necessário informar id_sentenca para instancia da class TimebankPT.')
                return self.tlink_completo.query("doc == ''")
            
            return self.tlink_join_completo.query("doc == '" + self.__tb.nome_doc_unico + "'")
        
        @property
        def tlink_join_completo(self):
            '''
            Retorna DataFrame de TLink completo unido com os principais campos das chaves estrangeira.
            Exibe todos os registros de TLink.
            
            '''
            #df principais filtrados
            df_tlink  = self.tlink_completo
            df_event  = self.event_completo
            df_timex3 = self.timex3_completo
            
            #df com PK renomeado para join
            df_event_eventID = df_event.rename(columns={'eid': 'eventID'})
            df_timex3_relatedToTime = df_timex3.rename(columns={'tid': 'relatedToTime'})
            df_event_relatedToEvent = df_event.rename(columns={'eid': 'relatedToEvent'})
            df_timex3_anchorTimeID = df_timex3.drop(columns=['anchorTimeID']).rename(columns={'tid': 'anchorTimeID'}) #apaga coluna anchorTimeID original antes de renomear as PK com o mesmo nome
            
            df_event = None
            df_timex3 = None
            
            #JOIN eventID
            df_join_eventID = df_tlink.merge(df_event_eventID, on=['eventID', 'doc'], how='outer', suffixes=('_TLINK_L', '_EVENTID_R')) #left    # outer = mostra eventos que não possuem tlinks
            df_tlink = None
            
            #JOIN relatedToTime
            df_join_event_relatedToTime = df_join_eventID.merge(df_timex3_relatedToTime, on=['relatedToTime', 'doc'], how='left', suffixes=('_EVENTID_L', '_RTOTIME_R'))
            df_join_eventID = None
            
            #JOIN relatedToEvent
            df_join_event_relatedToTime_relatedToEvent = df_join_event_relatedToTime.merge(df_event_relatedToEvent, on=['relatedToEvent', 'doc'], how='left', suffixes=('_RTOTIME_L', '_RTOEVENT_R'))
            df_join_event_relatedToTime = None
            
            #JOIN anchorTimeID
            col = [ 'lid', 'relType', 'task', 'doc', 'train_test_EVENTID_R', 
                    'eventID', 'isentenca_EVENTID_L', 'text_EVENTID_L', 'class_RTOTIME_L', 'tense_RTOTIME_L', 'pos_RTOTIME_L', 'aspect_RTOTIME_L',
                    'relatedToTime', 'isentenca_RTOTIME_R', 'tag_RTOTIME_L', 'type_RTOEVENT_L', 'value_RTOEVENT_L', 'text_RTOTIME_R',
                    'relatedToEvent', 'isentenca_RTOEVENT_L', 'text_RTOEVENT_L', 'class_RTOEVENT_R', 'tense_RTOEVENT_R', 'pos_RTOEVENT_R', 'aspect_RTOEVENT_R', 
                    'anchorTimeID', 'isentenca_ANCHOR_R', 'tag', 'type_ANCHOR_R', 'value_ANCHOR_R', 'text_ANCHOR_R']
            df_join_event_relatedToTime_relatedToEvent_anchorTimeID = df_join_event_relatedToTime_relatedToEvent.merge(df_timex3_anchorTimeID, on=['anchorTimeID', 'doc'], how='left', suffixes=('_RTOEVENT_L', '_ANCHOR_R'))[col]

            #Renomeia cnome das colunas para facilitar a visualização
            col_rename = {'train_test_EVENTID_R': 'train_test', 
                    'isentenca_EVENTID_L': 'isentenca', 'text_EVENTID_L': 'text', 'class_RTOTIME_L': 'class', 'tense_RTOTIME_L': 'tense', 'pos_RTOTIME_L': 'pos', 'aspect_RTOTIME_L': 'aspect',
                    'isentenca_RTOTIME_R': 'isentenca_rt', 'tag_RTOTIME_L': 'tag_rt', 'type_RTOEVENT_L': 'type_rt', 'value_RTOEVENT_L': 'value_rt', 'text_RTOTIME_R': 'text_rt',
                    'isentenca_RTOEVENT_L': 'isentenca_re', 'text_RTOEVENT_L': 'text_re', 'class_RTOEVENT_R': 'class_re', 'tense_RTOEVENT_R': 'tense_re', 'pos_RTOEVENT_R': 'pos_re', 'aspect_RTOEVENT_R': 'aspect_re', 
                    'isentenca_ANCHOR_R': 'isentenca_at', 'tag': 'tag_at', 'type_ANCHOR_R': 'type_at', 'value_ANCHOR_R': 'value_at', 'text_ANCHOR_R': 'text_at'}
            df_join_event_relatedToTime_relatedToEvent_anchorTimeID.rename(columns=col_rename, inplace=True)
            
            col = ['lid', 'doc', 'train_test', 'relType', 'task', 'eventID', 'isentenca', 'text', 'class', 'aspect', 'relatedToTime', 'isentenca_rt', 'tag_rt', 'type_rt', 'value_rt', 'text_rt', 'relatedToEvent', 'isentenca_re', 'text_re', 'class_re', 'aspect_re', 'anchorTimeID', 'isentenca_at', 'tag_at', 'type_at','value_at', 'text_at']
            return df_join_event_relatedToTime_relatedToEvent_anchorTimeID.sort_values(['isentenca', 'task', 'eventID', 'relatedToTime', 'relatedToEvent'])[col]  
        
        #---------------------
        def dataset(self, dados: Literal["train", "test", "all"] = 'all'):
            '''Dataset de dados anotados do corpus'''
            
            if dados not in ['train', 'test', 'all']:
                raise ValueError("dados deve ser: 'train' ou 'test'")

            str_dados = ''
            if dados in ['train', 'test']:
                str_dados = " and train_test == '" + dados + "'"
                
            colunas = ['train_test', 'isentenca', 'lid', 'eventID', 'relatedToTime', 'doc', 'relType']
            return self.tlink_join_completo.query(self.__tb.query_filtro_task('A') + str_dados)[colunas]

        @property
        def X_train(self):
            '''Dados de treino sem a classe'''
            return self.dataset('train').drop(['relType'], axis=1)
        @property
        def y_train(self):
            '''Dados de treino somente a classe'''
            return self.dataset('train')['relType']
        
        @property
        def X_test(self):
            '''Dados de teste sem a classe'''
            return self.dataset('test').drop(['relType'], axis=1)
        @property
        def y_test(self):
            '''Dados de teste somente a classe'''
            return self.dataset('test')['relType']
        #-------------------------

        def __stratified_kfold_attrib(self, df, k, attrib):
            '''Retorna k dataframes estratificado por attrib'''
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            grupos = df[attrib]
            folds = []
            for _, index in skf.split(np.zeros(len(grupos)), grupos):
                fold = df.iloc[index]
                folds.append(fold)
            return folds

        def __dividir_lista(self, lista, list_index):
            if type(list_index) == int:
                list_index = [list_index]
            if type(list_index) != list:
                raise ValueError('list_index deve ser int ou List[int]')
            
            lista_dos_indices = [lista[i] for i in list_index]
            lista_complemento = [elemento for i, elemento in enumerate(lista) if i not in list_index]
            return lista_complemento, lista_dos_indices

        def __is_test(self, i, k, size_test):
            list_is = []
            for j in range(size_test):
                list_is.append((i+j) % k)
            return list_is
            
        def split_train_test_kfolds(self, df, k: int, size_test: int = 1):
            '''
            Retorna lista com 'k' dataframes com dados de treino e de teste, já com os k-folds alternados, conforme size_test.
            Ex: k = 5, size_test = 2
                Os conjuntos de treino serão:
                [2,3,4], [0,3,4], [0,1,4], [0,1,2], [1,2,3]
            
            Args:
                df: Dataframe. Ex: tb.df.dataset('test')
                k: quantidade de folds
                size_test: Quantidade de folds para os dados de teste

            Return:
                list_train: Lista de dataframes dos dados de treino com a classe
                list_test: Lista de dataframes dos dados de teste com a classe
                
            '''
            if size_test >= k:
                raise ValueError('size_test deve ser menor que k')
            
            #Lista de dataframes dividido em kfolds, gerado pelo função stratified_kfold(df, k)
            kfolds = self.__stratified_kfold_attrib(df, k, 'relType')
            
            list_train = []
            list_test = []
            for i in range(k):
                list_kfolds_train, list_kfolds_test = self.__dividir_lista(kfolds, self.__is_test(i, k, size_test))
                df_train = pd.concat(list_kfolds_train, axis=0)
                df_test = pd.concat(list_kfolds_test, axis=0)
                list_train.append(df_train)
                list_test.append(df_test)

            return list_train, list_test
        
        def split_train_isentencas_kfolds(self, df, k: int, size_test: int = 1):
            '''
            Retorna 'k' listas de id_sentenca dos dados de treino dos k-folds. 
            Considera a proporção de cada classe (estratificado).
            Ex: k = 5, size_test = 2
                Os conjuntos de treino de onde as id_sentencas serão retiradas:
                [2,3,4], [0,3,4], [0,1,4], [0,1,2], [1,2,3]
            
            Args:
                df: Dataframe. Ex: tb.df.dataset('test')
                k: quantidade de folds
                size_test: Quantidade de folds para os dados de teste
            '''
            list_train, _ = self.split_train_test_kfolds(df, k, size_test)

            list_isentencas = []
            for i in range(k):
                list_isentencas.append(sorted(list(set(list_train[i]['isentenca']))))

            return list_isentencas
        
        def split_isentencas_kfolds(self, df: DataFrame, k: int) -> List[list]:
            '''
            Retorna 'k' listas de id_sentenca do 'df' (k-folds). 
            Não considera a proporção das classes (não estratificado).
            
            Args:
                df: Dataframe. Ex: tb.df.dataset('test')
                k: quantidade de folds
            '''
            if k <= 1 or df.shape[0] < k:
                raise ValueError("Valor inválido. k deve ser maior que 1 e menor que o tamanho do dataframe.")
            
            if 'isentenca' not in df.columns:
                raise ValueError('O dataframe deve ter a coluna "isentenca".')
                
            list_sentenca = list(set(df['isentenca']))
            n = len(list_sentenca) // k  # Número de elementos em cada subconjunto
            list_isentencas = []

            for i in range(k):
                inicio = i * n  # Índice inicial do subconjunto
                fim = (i + 1) * n if i < k - 1 else len(list_sentenca)  # Índice final do subconjunto

                subconjunto = list_sentenca[:inicio] + list_sentenca[fim:]  # Criação do subconjunto excluindo a parte atual
                list_isentencas.append(subconjunto)

            return list_isentencas
        #--------------------------
        

        #Retorna a quantidade de registro do dataframe
        @property
        def quant_sentenca(self):
            return self.sentenca.shape[0]
        @property
        def quant_sentenca_total(self):
            return self.sentenca_completo.shape[0]
        
        @property
        def quant_doc(self):
            return self.documento.shape[0]
        @property
        def quant_doc_total(self):
            return self.documento_completo.shape[0]
        
        @property
        def quant_event(self):
            return self.event.shape[0]
        @property
        def quant_event_total(self):
            return self.event_completo.shape[0]
        
        @property
        def quant_timex3(self):
            return self.timex3.shape[0]
        @property
        def quant_timex3_total(self):
            return self.timex3_completo.shape[0]
        
        @property
        def quant_tlink(self):
            return self.tlink.shape[0]
        @property
        def quant_tlink_total(self):
            return self.tlink_completo.shape[0]
        
        @property
        def dados_pipe(self):
            '''
            Retorna dicionário contendo os dados necessários para o processamento do pipeline do spaCy: pipe_timebankpt.
            
            Return:
                {
                    'texto da sentença': 
                    {
                        'isentenca': 'id sentença',
                        'doc': 'nome do arquivo doc',
                        'dct': 'data de criação do documento',
                        'lista_event': [[], [], []],
                        'lista_timex3': [[], []],
                        'lista_tlink': [[], []]
                    },
                    
                    'Repetidamente, ele resiste.':
                    {  
                        'isentenca': '254', 
                        'doc': 'ABC19980120.1830.0957', 
                        'dct': '1998-01-20', 
                        'lista_event': [['EVENT', '2', 'e1', 'previram', 14, 22, 'I_ACTION', 'NONE'], ['EVENT', '2', 'e86', 'queda', 29, 34, 'OCCURRENCE', 'NONE']], 
                        'lista_timex3': [['TIMEX3', '10', 't94', 'quase quarenta anos', 8.0, 27.0, 'DURATION', 'P40Y']],
                        'lista_tlink': [['TLINK', 'l3', 'B', 'AFTER', 5, 'e11', '', 't93', '', '']]
                    }
                }

            '''
            col = ['isentenca', 'sentenca', 'doc', 'dct', 'tid', 'type', 'train_test']
            df_sentenca = self.sentenca_completo.merge(self.documento_completo, on=['doc', 'train_test'], how='left')[col]
            df_sentenca.set_index('sentenca', inplace=True)

            dict_pipe = {}
            for sentenca, row in df_sentenca.iterrows():
                dados = {   'isentenca': row['isentenca'], 'tid':row['tid'], 'doc': row['doc'], 'dct': row['dct'], 'type': row['type'], 'train_test': row['train_test'], 
                            'lista_event': self.__get_lista_event(row['isentenca']), 
                            'lista_timex3': self.__get_lista_timex3(row['isentenca']),
                            'lista_tlink': self.__get_lista_tlink(row['isentenca'])
                        }
                dict_pipe.update({sentenca: dados})

            return dict_pipe

        
        def __trata_id_sentenca_unica(self, id_sentenca):
            '''
            Trata id_sentenca e retorna o primeiro se a lista contiver mais de um id_sentença.
            '''
            if not id_sentenca:
                return []
            
            id_sentenca = self.__tb.trata_lista(id_sentenca)
            
            if len(id_sentenca) == 0:
                return []
            
            return id_sentenca[0]
        
        def __get_lista_event(self, id_sentenca):
            '''
            Retorna lista de todos os eventos (EVENT) da sentença 'id_sentenca'. 
            Considera apenas a primeira sentenca se 'id_sentenca' for uma lista.
            
            '''
            id_sentenca = self.__trata_id_sentenca_unica(id_sentenca)
            
            cols = ['tag', 'isentenca', 'eid', 'text', 'p_inicio', 'p_fim', 'class', 'aspect', 'tense', 'pos', 'polarity']
            df_event_sentenca = self.event_completo[cols].query("isentenca == " + str(id_sentenca) + "")
            
            if df_event_sentenca.empty:
                return []

            return df_event_sentenca.values.tolist()
            

        def __get_lista_timex3(self, id_sentenca):
            '''
            Retorna lista de todas as expressões temporais (TIMEX3) da sentença 'id_sentenca'. 
            Considera apenas a primeira sentenca se 'id_sentenca' for uma lista.

            '''
            id_sentenca = self.__trata_id_sentenca_unica(id_sentenca)

            cols = ['tag', 'isentenca', 'tid', 'text', 'p_inicio', 'p_fim', 'type', 'value', 'value_group', 'anchorTimeID', 'temporalFunction']
            df_timex3_sentenca = self.timex3_completo[cols].query("isentenca == " + str(id_sentenca) + "")
            df_timex3_sentenca['anchorTimeID'].fillna('', inplace = True)
            
            if df_timex3_sentenca.empty:
                return []
            
            return df_timex3_sentenca.values.tolist()
        
        
        def __get_lista_tlink(self, id_sentenca):
            '''
            Retorna lista das relações temporais (TLINK) da sentenca 'id_sentenca'.
            Considera apenas a primeira sentenca se 'id_sentenca' for uma lista.
            '''
            id_sentenca = self.__trata_id_sentenca_unica(id_sentenca)
            
            cols = ['tag', 'lid', 'task', 'relType', 'eventID', 'relatedToTime', 'relatedToEvent', 'doc']
            df_tlink_sentenca = self.tlink_completo.query("doc == '" + self.__tb.get_nome_documento(id_sentenca) + "' and eventID in " + str(self.__tb.get_eventID(id_sentenca))).fillna('')
            
            if df_tlink_sentenca.empty:
                return []
            
            return df_tlink_sentenca[cols].values.tolist()
        
        
        #-------------------------------
        # FUNÇÕES PRIVADAS DA CLASSE DF
        #-------------------------------
        
        #------- FUNÇÕES PARA ADICIONAR DO CAMPO 'value_group' em TIMEX3
        def __is_digit(self, value) -> bool:
            for c in value:
                if c not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    return False
            return True

        def __is_period(self, value) -> bool:
            if value[0] in ['p', 'P'] and value[-4:] != '_REF':
                return True
            return False

        def __is_year(self, value) -> bool:
            if (3 <= len(value) <= 4) and (self.__is_digit(value[0:2]) or value[2] == 'X' or value[3] == 'X'):
                return True
            return False

        def __is_sub_year(self, value) -> bool:
            if 7 <= len(value) <= 8 and self.__is_year(value[0:4]) and value[4] == '-':
                return True
            return False

        def __value_group(self, value: str):
            #agrupar value de timex3 em: year, sub-year, sub-month, period
            if self.__is_period(value):
                return 'period'
            if self.__is_year(value):
                return 'year'
            if self.__is_sub_year(value):
                return 'sub_year'
            if value in ['FUTURE_REF', 'PAST_REF', 'PRESENT_REF']:
                return value
            return 'sub_month'

        def __get_train_test_filename(self, nome_arquivo: str) -> Literal["train", "train_test", "test"]:
            '''
            Retorna se o arquivo é de teste ou de treino, baseado no subdiretório onde cada tipo está armazenado.

            Args:
                nome_arquivo: nome do arquivo que represente um documento do corpus.

            '''
            nome_inverso = nome_arquivo[::-1]
            ini = nome_inverso.find('\\')
            fim = nome_inverso.find('\\', ini + 1)
            tipo = nome_inverso[fim-1:ini:-1].lower()
            
            if not self.__tb.dev:
                if tipo == 'train_test':
                    tipo = 'train'
            
            return tipo
    
        def __timeml_to_df(self):
            '''
            Retorna Dataframes das tags Timex3, Event e Tlink dos arquivos TimeML
            mais as sentenças com tags e sentenças sem tags

            Return:
                df_event, df_timex3, df_tlink, df_sentenca, df_doc

            '''
            event = []
            timex3 = []
            tlink = []
            sentenca = []
            list_doc = []
            isentenca = 0

            for arq_timeml in self.lista_arquivos():
                train_test = self.__get_train_test_filename(arq_timeml)
                xtree = et.parse(arq_timeml)
                root = xtree.getroot()

                nome_arquivo = os.path.basename(arq_timeml)
                doc, ext = os.path.splitext(nome_arquivo)

                for node in list(root):
                    #DCT: há apenas um DCT para cada documento
                    if node.tag == 'TIMEX3':
                        node.attrib['tag'] = 'DCT'
                        node.attrib['text'] = node.text
                        node.attrib['doc'] = doc
                        node.attrib['train_test'] = train_test
                        timex3.append(node.attrib)
                        
                        #DCT de df_doc
                        list_doc.append({'tid': node.attrib['tid'], 'doc': doc, 'dct': node.attrib['value'], 'type': node.attrib['type'], 'train_test': train_test})

                        #print(node.attrib)

                    if node.tag == 's':
                        s_tag = et.tostring(node, encoding="unicode")
                        #retira todo conteúdo após o fechamento da tag 's'. É lixo.
                        s_tag = s_tag[:s_tag.find('</s>') + 4]
                        
                        #remove todas as tags da sentença
                        s = self.__removeTags(s_tag)
                        
                        #Retira sentenças com conteúdo inválido no formato: NYT-02-06-98 2219EST, AP-NY-03-01-98 1411EST
                        if len(re.findall(r'[A-Z]{2,3}-\w{2}-\w{2}', s)) > 0:
                            continue

                        #retira sentenças com conteúdo inválido, como poucos caracteres e entre parêntesis: (sp/eml)
                        if len(re.findall(r'^(\w|\(|\[).{0,15}(\)|\])$', s)) > 0: 
                            continue
                        
                        isentenca += 1
                        sentenca.append({'isentenca': isentenca, 'doc': doc, 'sentenca': s, 'sentenca_tag': s_tag, 'train_test': train_test})

                        i_aux = 0
                        for elem in list(node):
                            elem_text = elem.text.strip()

                            #Retira elementos não aceitável em EVENT e TIMEX3
                            retirar_de_elem = '!"#&()*+:;<=>?[\\]^`{|}~'  #dúvida se fica: /
                            elem_text = ''.join([i for i in elem_text if i not in retirar_de_elem])
                            
                            #\b = limites da palavra. Coincidir palavra inteira
                            reg = r"\b" + elem_text + r"\b"
                            #Trata valores numéricos com %. 
                            encontrou_pct = (len([i for i in elem_text if i in ["%"]])) >= 1
                            if encontrou_pct:
                                reg = r"\b" + elem_text
                                
                            encontrou = re.search(reg, s[i_aux:])
                            inicio = -1
                            if encontrou:
                                #como a pesquisa é feita na substring iniciada em i_aux, é necessário adicionar seu valor em inicio para que a posição seja sempre em referência ao primeiro caractere da sentença
                                inicio = encontrou.start() + i_aux

                            #Corrigir problemas com HÍFENS
                            #Adicionar o sufixo com hífen, se houver, ao elem_text
                            #ex: se elem_text = declarar
                            #mas na sentença = declarar-se então adiciona o sufixo em elem_text
                            #É necessário porque o TOKEN é formado também com o sufixo
                            pos_proximo_char = inicio + len(elem_text)
                            proximo_char = s[pos_proximo_char]
                            tamanho_sufixo = 0
                            if proximo_char == '-':
                                #Se o próximo char do elem for um hífen
                                #a partir do hífen, busca o próximo char que não seja letra ou hífen ou seja fim da linha
                                tamanho_sufixo = re.search(r'[^\w|-]|$', s[pos_proximo_char:]).start()

                            fim = pos_proximo_char + tamanho_sufixo
                            #atualizar i_aux para a posição de fim do elemento encontrado, isso para que na próxima iteração a busca seja a partir desse ponto.
                            if inicio >= 0: #se encontrou
                                i_aux = fim

                            elem.attrib['p_inicio'] = int(inicio)
                            elem.attrib['p_fim'] = int(fim)

                            elem.attrib['tag'] = elem.tag
                            elem.attrib['text'] = elem_text
                            elem.attrib['isentenca'] = (isentenca)
                            elem.attrib['doc'] = doc
                            elem.attrib['train_test'] = train_test

                            if elem.tag == 'EVENT':
                                event.append(elem.attrib)
                            if elem.tag == 'TIMEX3':
                                timex3.append(elem.attrib)
                            #print(elem.attrib)
                            
                    if node.tag == 'TLINK':
                        node.attrib['tag'] = node.tag
                        node.attrib['doc'] = doc
                        node.attrib['train_test'] = train_test
                        tlink.append(node.attrib)
                        #print(node.attrib)
            
            col_timex3 = ['tid', 'isentenca', 'tag', 'text', 'type', 'value', 'value_group', 'anchorTimeID', 'temporalFunction', 'functionInDocument', 'mod', 'beginPoint', 'endPoint', 'quant', 'freq', 'p_inicio', 'p_fim', 'doc', 'train_test']
            df_timex3 = pd.DataFrame(timex3, columns=col_timex3) 
            # Adiciona o campo 'value_group' que agrupo a campo value em period, year, sub_year, sub_month
            df_timex3['value_group'] = df_timex3['value'].apply(self.__value_group)
            df_timex3['temporalFunction'] = df_timex3['temporalFunction'].map({'true':True, 'false': False}) 
            
            df_timex3 = df_timex3[col_timex3]
            df_timex3 = df_timex3.sort_values(['doc', 'isentenca', 'p_inicio'])
            
            
            df_event = pd.DataFrame(event)
            df_event = df_event.astype({'isentenca': 'int'})
            df_event = df_event[['eid', 'isentenca', 'tag', 'text', 'class', 'stem', 'aspect', 'tense', 'polarity', 'pos', 'p_inicio', 'p_fim', 'doc', 'train_test']]
            df_event = df_event.sort_values(['doc', 'isentenca', 'p_inicio'])
            
            df_tlink = pd.DataFrame(tlink)
            df_tlink = df_tlink[['lid', 'tag', 'task', 'relType', 'eventID', 'relatedToTime', 'relatedToEvent', 'doc', 'train_test']]
            df_tlink = df_tlink.sort_values(['doc', 'task', 'eventID', 'relatedToTime', 'relatedToEvent'])
            
            df_sentenca = pd.DataFrame(sentenca)
            df_doc = pd.DataFrame(list_doc)
            
            return df_event, df_timex3, df_tlink, df_sentenca, df_doc
        

        def __timeml_to_df_filter(self, recursivo = True):
            '''
            Retorna dataframes filtrado conforme id_sentenca do parâmetro e todas as sentenças relacionadas a ela, contendo as tags Event, Timex3 e Tlink dos arquivos TimeML
                mais a sentença com tags e sem tags.

            Args:
                recursivo: se True, busca as sentenças que estão relacionadas com id_sentenca, bem como EVENT, TIMEX3 e TLINK. 

            Return:
                df_event_filter, df_timex3_filter, df_tlink_filter, df_sentenca_filter

            '''
            #Converte valor da sentença recebido em uma lista de strings
            id_sentenca = self.__tb.id_sentenca
            
            #Cria dataframes vazios
            df_event_vazio = self.event_completo.query("isentenca == ''")
            df_timex3_vazio = self.timex3_completo.query("isentenca == ''")
            df_tlink_vazio = self.tlink_completo.query("lid == ''")
            df_sentenca_vazio = self.sentenca_completo.query("isentenca == ''")
            df_doc_vazio = self.documento_completo.query("doc == ''")

            if len(id_sentenca) == 0:
                return df_event_vazio, df_timex3_vazio, df_tlink_vazio, df_sentenca_vazio, df_doc_vazio

            #DOCUMENTO
            doc = self.sentenca_completo.query("isentenca in " + str(id_sentenca))['doc'].unique()
            if len(doc) == 0:
                doc = ''
                print('\nÉ necessário informar sentenças válidas.\n')
                return df_event_vazio, df_timex3_vazio, df_tlink_vazio, df_sentenca_vazio, df_doc_vazio
            elif len(doc) == 1:
                doc = doc[0]
            elif len(doc) > 1:
                doc = doc[0]
                print('\nAs sentenças pertencem a mais de um documento. \nA lista de sentenças devem pertencer a um mesmo documento.\n')
                #return df_event_vazio, df_timex3_vazio, df_tlink_vazio, df_sentenca_vazio, df_doc_vazio


            #EVENT
            df_e = self.event_completo.query("isentenca in " + str(id_sentenca))

            #lista de todos event da sentença X
            list_e = df_e['eid'].unique().tolist()


            #TIMEX3
            df_t = self.timex3_completo.query("isentenca in " + str(id_sentenca))

            #lista todos Timex3 da sentença X
            list_t = df_t['tid'].dropna().unique().tolist()
            list_anchor = df_t['anchorTimeID'].dropna().unique().tolist()
            list_t.extend(list_anchor)
            list_t = list(set(list_t))


            #TLINK: Busca Event e Timex3
            q = "(doc == '" + doc + "') and (eventID in " + str(list_e) + " or relatedToEvent in " + str(list_e) + " or relatedToTime in " + str(list_t) + ")"
            df_l = self.tlink_completo.query(q)

            if recursivo:
                #lista de relatedToEvent dos event de df_event em df_tlink
                list_e_tlink = df_l['relatedToEvent'].dropna().unique().tolist()
                list_e.extend(list_e_tlink)
                list_e = list(set(list_e))

            if recursivo:
                #lista de relatedToTime dos event de df_event em df_tlink
                list_t_tlink = df_l['relatedToTime'].dropna().unique().tolist()
                list_t.extend(list_t_tlink)
                list_t = list(set(list_t))


            #Lista Timex3 apenas DCT: Geralmente cada documento contém apenas um DCT, mas deixei como lista assim mesmo
            filtro_dct = "doc == '" + doc + "' and functionInDocument == 'CREATION_TIME' and tid in " + str(list_t)
            list_t_dct = self.timex3_completo.query(filtro_dct)['tid'].dropna().unique().tolist()
            #Lista Timex3 sem DCT
            list_t_no_dct = list(set(list_t).difference(set(list_t_dct)))


            #TLINK
            q = "(doc == '" + doc + "') and (eventID in " + str(list_e) + " or relatedToEvent in " + str(list_e) + " or relatedToTime in " + str(list_t_no_dct) + ")"
            df_tlink_filter = self.tlink_completo.query(q)

            #Atualiza lista de event e timex3 adicionando os event e timex3 de TLINK
            if recursivo:
                list_eventID = df_tlink_filter['eventID'].dropna().unique().tolist()
                list_relatedToEvent = df_tlink_filter['relatedToEvent'].dropna().unique().tolist()
                list_eventID.extend(list_relatedToEvent)

                list_e.extend(list_eventID)
                list_e = list(set(list_e))


            #RETORNOS de Event e Timex3
            df_event_filter = self.event_completo.query("doc == '" + doc + "' and eid in " + str(list_e))
            df_timex3_filter = self.timex3_completo.query("doc == '" + doc + "' and tid in " + str(list_t))


            #SENTENÇAS
            list_sentenca = id_sentenca.copy()
            if recursivo:
                list_sentenca.extend(df_event_filter['isentenca'].dropna().unique().tolist())
                list_sentenca.extend(df_timex3_filter['isentenca'].dropna().unique().tolist())
            list_sentenca = list(set(list_sentenca))

            df_sentenca_filter = self.sentenca_completo.query("doc == '" + doc + "' and isentenca in " + str(list_sentenca))

            df_doc_filter = self.documento_completo.query("doc == '" + doc + "'")
            
            return df_event_filter, df_timex3_filter, df_tlink_filter, df_sentenca_filter, df_doc_filter
    
        
        def __removeTags(self, raw_tags):
            '''
            Retorna texto sem tags.

            Args:
                raw_tags: texto com tags
                
            '''
            if raw_tags is None:
                return ''
            else:
                raw_tags = html.unescape(raw_tags)
                #cleanr = re.compile('<.*?>')
                cleanre = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6});')
                cleanText = re.sub(cleanre, '', raw_tags)
                #Retira quebra de linha e espaços duplos
                str_tratada = cleanText.replace("\n", " ").replace("  ", " ").replace("  ", " ").strip()
                return str_tratada

    #------------------------------------------------
    #-----  FIM  -  class Df  -----------------------
    #------------------------------------------------
        

    #=========================================================================================================================
    # '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
    #  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
    #  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
    #  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
    #  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
    #  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
    #  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
    # ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #
    #=========================================================================================================================

        
    #-------------------------------------------
    #-----  CLASS PRINT  ----  IMPRESSÕES ------
    #-------------------------------------------

    class Print():
        '''
        Formata para impressão em tela os elementos do Timebank e recursos do spaCy como Entidades, POS, Morph, árvore de dependência.
        
        Args:
            tb: Instancia da classe TimebankPT.
        '''
        def __init__(self, tb: 'TimebankPT'):  #o 'tb' é recebido no self do parâmetro da instanciação da class, ex: df = Df(self)
            
            self.__sentenca_anterior = None
            self.__id_sentenca_anterior = None
                
            self.__tb = tb
            
        
        def __cabecalho_sentenca(self, i: int, com_texto_sentenca = True):
            '''
            Retorna cabeçalho para as impressões de dados do Spacy com TimebankPT

            '''
            dct = ''
            nome = ''
            id_sent = ''
            sentenca = ''
            texto = ''

            if self.__tb.sentenca_texto:
                
                sentenca = self.__tb.sentenca_texto
                
                if type(sentenca) == list:
                    sentenca = sentenca[i]

                if com_texto_sentenca:
                    texto = '\n\n-------------------------------------------------------------------\n{3}\n'
                else:
                    texto = '\n-------------------------------------------------------------------\n'

            if self.__tb.id_sentenca:
                if com_texto_sentenca:
                    texto = '\n-------------------------------------------------------------------\nDCT: {0}   DOC: {1}   ID_SENTENCA: {2} \n{3}\n'
                else:
                    texto = '\n\n-------------------------------------------------------------------\nDCT: {0}   DOC: {1}   ID_SENTENCA: {2} \n'

                id_sent = self.__tb.id_sentenca[i]
                dct = self.__tb.dct_doc[i]
                nome = self.__tb.nome_doc[i]

            return texto.format(dct, nome, id_sent, sentenca)
    
        def __valida_doc(self):
            '''
            Verifica se os objeto Doc do spaCy existe.

            Return:
                Retorna True se existir, False se não existir.
            '''
            if not self.__tb.doc:
                print("ERROR: Necessário inicializar objeto Doc. Atribua id_sentenca à instancia da classe TimebankPT")
                return False
            else:
                return True
            
        def __valida_id_sentenca(self):
            '''
            Verifica se a propriedade id_sentenca da class existe.
            
            Return:
                Retorna True se existir, False se não existir.
            '''
            if not self.__tb.id_sentenca:
                print("ERROR: Necessário inicializar objeto Doc. Atribua id_sentenca à instancia da classe TimebankPT")
                return False
            else:
                return True
            
        def __valida_pipe(self):
            '''
            Verifica se o pipeline 'pipe_timebankpt' foi adicionado ao processamento do spaCy.
            
            Return:
                Retorna True se foi adicionado, False se não.
            '''
            if not (self.__tb.nlp.has_pipe("pipe_timebankpt") and Doc.has_extension('id_sentenca')):
                print("ERROR: Necessário adicionar pipeline 'pipe_timebankpt' ao pipeline do spaCy: TimebankPT.add_pipe_timebank()")
                return False
            else:
                return True
            
        def __salva_sentenca_anterior(self, id_sentenca = None):
            '''
            Salvar sentença anterior para permitir informar manualmente id_sentenca nos métodos de impressão (ent, pos, graph, ent_timebank) sem perder o id_sentenca da instancia da classe.
            '''
            if id_sentenca is not None:
                self.__sentenca_anterior = self.__tb.sentenca_texto
                self.__id_sentenca_anterior = self.__tb.id_sentenca
                self.__tb.id_sentenca = id_sentenca
        
        def __recupera_sentenca_anterior(self, id_sentenca = None):
            '''
            Recupera id_sentenca da instancia da classe após impressão pelos métodos: ent, pos, graph, ent_timebank utilizando o parametro opcional id_sentenca.
            '''
            if id_sentenca is not None:
                if self.__sentenca_anterior:
                    self.__tb.sentenca_texto = self.__sentenca_anterior
                self.__tb.id_sentenca = self.__id_sentenca_anterior
                
        def __trata_lista_doc(self):
            '''
            Retorna uma lista de Doc, se receber um Doc apenas, coloque-o em uma lista com apenas um elemento.
            '''
            lista_doc = []
            if type(self.__tb.doc) == Doc:
                lista_doc.append(self.__tb.doc)
            else:
                lista_doc.extend(self.__tb.doc)
            return lista_doc
        
        
        #---------------------------------------------
        #DEFINIÇÃO DOS CAMPOS
        def __campos_morph(self, token: Token):
            '''
            Retorna dicionário contendo os campos do spaCy que serão impressos na tela
            '''
            return {
                'i'        :token.i,
                'Token'    :token.orth_,
                'ENT'      :token.ent_type_,
                'POS'      :token.pos_,
                'Desc POS' :self.__tb.explicar_spacy(token.pos_), 
                'PAI'      :token.head, 
                'POS PAI'  :token.head.pos_,
                'DEP'      :token.dep_, 
                'Desc Dep' :self.__tb.explicar_spacy(token.dep_), 
                'Morph'    :token.morph
            }
        
        def __campos_pais(self, token: Token):
            '''
            Retorna dicionário contendo os campos do spaCy que serão impressos na tela
            '''
            return {
                'i'         :token.i,
                'Token'     :token.orth_,
                'ENT'       :token.ent_type_,
                'POS'       :token.pos_,
                'Desc POS'  :self.__tb.explicar_spacy(token.pos_), 
                'PAI'       :token.head, 
                'POS PAI'   :token.head.pos_,
                'DEP'       :token.dep_, 
                'Desc Dep'  :self.__tb.explicar_spacy(token.dep_), 
                'Ancestors'    :([ ('{0}:{1}'.format(a, a.dep_) ) for a in token.ancestors])
            }
        
        def __campos_filhos(self, token: Token):
            '''
            Retorna dicionário contendo os campos do spaCy que serão impressos na tela
            '''
            return {
                'i'         :token.i,
                'Token'     :token.orth_,
                'ENT'       :token.ent_type_,
                'POS'       :token.pos_,
                'Desc POS'  :self.__tb.explicar_spacy(token.pos_), 
                'PAI'       :token.head, 
                'POS PAI'   :token.head.pos_,
                'DEP'       :token.dep_, 
                'Desc Dep'  :self.__tb.explicar_spacy(token.dep_), 
                'Filhos Esq:dep'    :[ ('{0}:{1}'.format(child, child.dep_) ) for child in token.lefts],
                'Filhos Dir:dep'    :[ ('{0}:{1}'.format(child, child.dep_) ) for child in token.rights]
            }
        
        def __campos_timebank(self, token: Token):
            '''
            Retorna dicionário contendo os campos do spaCy que serão impressos na tela
            '''
            return {
                'i'           :token.i,
                'Token'       :token.orth_,
                'ENT'         :token.ent_type_,
                'POS'         :token.pos_,
                'Desc POS'    :self.__tb.explicar_spacy(token.pos_), 
                #'PAI'         :token.head, 
                #'POS PAI'     :token.head.pos_,
                #'DEP'         :token.dep_, 
                'id_sentenca' :token._.id_sentenca,  
                'id_tag'      :token._.id_tag,
                'classe'      :token._.classe, 
                'aspecto'     :token._.aspecto, 
                'tempo'       :token._.tense,
                'pos'         :token._.pos,
                'tipo'        :token._.tipo,
                'valor'       :token._.value,
                'relType'     :tuple(filter(lambda x: x[1] in ['A', 'B'], [list(token._.relType[x].values()) for x in token._.relType]))
            }
        
        def __campos_tokens(self, token: Token):
            '''
            Retorna dicionário contendo os campos do spaCy que serão impressos na tela
            '''
            return {
                'i'         :token.i,
                'Token'     :token.orth_,
                'Lemma'     :token.lemma_,
                'ENT'       :token.ent_type_,
                'POS'       :token.pos_,
                'Desc POS'  :self.__tb.explicar_spacy(token.pos_), 
            }
        #FIM DEFINIÇÃO DOS CAMPOS
        #--------------------------------------------------
        
        def imprimir_campos(self, func_campos, id_sentenca = None):
            '''
            Função genérica que recebe dados de tokens (em func_campos) para imprimi-los. 
            Os dados contem classes gramaticais (Part Of Speech), análise morfológica, análise de dependência e tags do corpus TimebankPT.
            
            Args:
                func_campos: Função que retorna dicionário contendo dados dos tokens que serão impressos. Os dados estão em funções iniciadas por '__campos_', ex: __campos_timebank, __campos_morph ...
            
            '''
            self.__salva_sentenca_anterior(id_sentenca)
            
            if not self.__valida_doc():
                return
            
            lista_doc = self.__trata_lista_doc()

            i = 0
            for doc in lista_doc:
                head = func_campos(doc[0]).keys()
                analise = [func_campos(token).values() for token in doc]
                
                print(self.__cabecalho_sentenca(i))
                print(tabulate(analise, head))
                i += 1
            
            self.__recupera_sentenca_anterior(id_sentenca)
            
        
        #---------------------------------------------------
        #FUNÇÕES QUE IMPRIME OS CAMPOS
        def timebank(self, id_sentenca = None):
            '''
            Imprime tags do timebank.
            
            '''
            if not self.__valida_id_sentenca():
                return
            
            if not self.__valida_pipe():
                return
            
            self.imprimir_campos(self.__campos_timebank, id_sentenca)
            
        def morph(self, id_sentenca = None):
            '''
            Imprime Classes gramaticais (Part Of Speech) e análise morfológica.
            
            '''
            self.imprimir_campos(self.__campos_morph, id_sentenca)
            
        def pais(self, id_sentenca = None):
            '''
            Imprime também os ancestrais de cada token.
            
            '''
            self.imprimir_campos(self.__campos_pais, id_sentenca)
            
        def filhos(self, id_sentenca = None):
            '''
            Imprime também os dependentes (filhos) de cada token.
            
            '''
            self.imprimir_campos(self.__campos_filhos, id_sentenca)
        
        def tokens(self, id_sentenca = None):
            '''
            Imprime tags POS.
            
            '''
            self.imprimir_campos(self.__campos_tokens, id_sentenca)
            
            
        #FIM FUNÇÕES QUE IMPRIME OS CAMPOS
        #---------------------------------------------------
        
            
        def ent(self, id_sentenca = None):
            '''
            Imprime Entidades Nomeadas, inclusive as tags EVENT e TIMEX3 do TimebankPT se o seu pipeline estiver adicionado ao spaCy.
            
            '''
            self.__salva_sentenca_anterior(id_sentenca)
            
            if not self.__valida_doc():
                return
            
            lista_doc = self.__trata_lista_doc()
            
            i = 0    
            for doc in lista_doc:
                print(self.__cabecalho_sentenca(i, False))
                spacy.displacy.render(doc, style="ent", jupyter=True, options={'colors':{'TIMEX3':'#6ecf42', 'EVENT': '#f1db42', 'MISC': '#d8e2dc'}})
                i += 1
            
            self.__recupera_sentenca_anterior(id_sentenca)
            
        
        def graph(self, id_sentenca = None, size = 'm', compact = True, punct = False):
            '''
            Imprime gráfico de análise de dependência.
            
            Args:
                id_sentenca: se fornecida lista de id_sentenca
                size: 'p', 'm', 'g' representa a distancia entre os tokens
                punct: se True, mostra as pontuações no grafo.
                
            '''
            distance = 140
            if size == 'g':
                distance = 180
            elif size == 'p':
                distance = 120
            
            self.__salva_sentenca_anterior(id_sentenca)
            
            if not self.__valida_doc():
                return
            
            lista_doc = self.__trata_lista_doc()

            i = 0
            for doc in lista_doc:
                print(self.__cabecalho_sentenca(i))
                spacy.displacy.render(doc, style='dep', jupyter=True, options={'distance':distance, 'compact':compact, 'word_spacing':30, 'fine_grained': True, 'collapse_punct': not punct, 'arrow_spacing': 20})
                i += 1
            
            self.__recupera_sentenca_anterior(id_sentenca)
            
        def graph_tlink(self, id_sentenca = None):
            '''
            Imprime gráfico das relações temporais anotadas no corpus entre eventos e expressões temporais (Task A).
            
            '''
            def graph_tlink_helper(doc):
                '''
                '''
                graph_tlinks = self.__tb.MyTlink(self.__tb)
                graph_tlinks.clear()
                
                for ent in doc.ents:
                    if ent.label_ in ['EVENT']:
                        for token in ent:
                            eventID = token._.id_tag
                            nome_documento = token.doc._.nome
                            isentenca = self.__tb.get_id_sentenca_do_doc(eventID, nome_documento)
                            rule = 2000

                            relTypes = token._.relType
                            for rel in relTypes:
                                task = relTypes[rel]['task']
                                if task == 'A':
                                    lid = rel
                                    relatedTo = relTypes[rel]['relatedTo']
                                    relType = relTypes[rel]['relType']
                                    graph_tlinks.add(relType, eventID, relatedTo, task, isentenca, nome_documento, rule, lid)

                graph_tlinks.graph_rt()
                
            
            if id_sentenca is None:
                id_sents = self.__tb.id_sentenca
            else:
                self.__salva_sentenca_anterior(id_sentenca)
            
            if not self.__valida_doc():
                return
            
            if id_sentenca is None:
                for_id_sents = id_sents
            else:
                for_id_sents = id_sentenca
            
            i = 0
            for id_sent in for_id_sents:
                self.__tb.id_sentenca = id_sent
                print(self.__cabecalho_sentenca(i))
                graph_tlink_helper(self.__tb.doc_unico)
            
            if id_sentenca is None:
                self.__tb.id_sentenca = id_sents
            else:
                self.__recupera_sentenca_anterior(id_sentenca)
            
            
        def graph_dfs(self, id_sentenca = None, mostrar_mais = True):
            '''
            Imprime árvore sintática utilizando a Busca por Profundidade
            '''
            visited = set()
            level_max = 7
            def dfs(node, level = 0):
                if node not in visited:
                    if mostrar_mais:
                        mais = '\t' * (level_max - level + 1) + '| {1}\t{2}\t{3}\t{4}\t{5}'.format(str(level), node.pos_, node.ent_type_, node._.classe, node._.tipo, str(node.morph))
                    else:
                        mais = ''
                    
                    print('{0:>3}\t'.format('|') * level + '{0:>4} {1:<20} {2}'.format(str(node.i) + '. ', node.text + ':' + node.dep_, mais))
                    
                    visited.add(node)
                    for neighbour in node.children:
                        dfs(neighbour, level+1)
            
            self.__salva_sentenca_anterior(id_sentenca)
            
            if not self.__valida_doc():
                return
            
            lista_doc = self.__trata_lista_doc()
            
            i = 0
            for doc in lista_doc:
                print(self.__cabecalho_sentenca(i))
                roots = self.__tb.get_doc_root(doc)
                for root in roots:
                    dfs(root)
                i += 1
            
            self.__recupera_sentenca_anterior(id_sentenca)
            
        
        def graph_treelib(self, id_sentenca = None):
            '''
            Imprime arvore sintática utilizando a Busca por Profundidade com a biblioteca treelib
            from treelib import Node, Tree
            '''
            visited = set()
            def dfs(node, level = 0):
                if node not in visited:
                    visited.add(node)
                    for neighbour in node.children:
                        tree.create_node(neighbour.text + ': ' + neighbour.dep_ , neighbour.i, parent=neighbour.head.i)
                        dfs(neighbour, level+1)
            
            self.__salva_sentenca_anterior(id_sentenca)
            
            if not self.__valida_doc():
                return
            
            lista_doc = self.__trata_lista_doc()
            
            i = 0
            for doc in lista_doc:
                print(self.__cabecalho_sentenca(i))
                
                roots = self.__tb.get_doc_root(doc)
                for root in roots:
                    tree = Tree()
                    tree.create_node(root, root.i)
                    dfs(root)
                    tree.show(idhidden=True, line_type = 'ascii-exr')   #line_type = 'ascii' 'ascii-ex' 'ascii-exr' 'ascii-em' 'ascii-emv' 'ascii-emh'
                i += 1
            
            self.__recupera_sentenca_anterior(id_sentenca)


    #---------------------------------------------------------------------
    # FIM  class Print
    #--------------------------------------------------------------------


    
    #=========================================================================================================================
    # '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
    #  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
    #  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
    #  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
    #  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
    #  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
    #  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
    # ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #
    #=========================================================================================================================
    
    
    #----------------------------------------------------------------------
    #-----  CLASSE MYTLINK  -  SALVA MEUS TLINK EXTRAIDOS  ----------------
    #----------------------------------------------------------------------
    
    class MyTlink:
        '''
        Estrutura de dados para as Relações Temporais previstas pelo método aqui proposto.
        Fornece impressão gráfica das relações.

        Args:
            tb: Instancia da classe TimebankPT.

        '''
        def __init__(self, tb: 'TimebankPT'):
            
            #Índice da lista MyTlink
            self.LID       = 0
            self.RELTYPE   = 1
            self.EVENTID   = 2
            self.TOTIME    = 3
            self.TOEVENT   = 4
            self.TASK      = 5
            self.ISENTENCA = 6
            self.DOC       = 7
            self.RULE      = 8
            
            self.relType_validos = ['BEFORE', 'AFTER', 'OVERLAP', 'BEFORE-OR-OVERLAP', 'OVERLAP-OR-AFTER', 'VAGUE']
            self.__tlink_list = []
            
            self.__struct = {}
            self.__struct['words'] = []
            self.__struct['arcs'] = []
            
            if not tb:
                return
            self.__tb = tb
        
        
        #---------------------------------
        # ------- PUBLIC  MYTLINK---------
        #---------------------------------
        
        def add(self, relType, eventID, relatedTo, task, isentenca, doc, rule, lid = None):
            '''
            Adiciona tags Tlink descoberta pelo método proposto à estrutura de dados armazenada em to_list.
            
            Args:
                relType: Tipo da relação temporal predita
                eventID: ID do EVENT 
                relatedTo: Pode ser relatedToTime ou relatedToEvent, é inferida através de task
                task: Tipo da tarefa do TempEval. 
                    A. EVENT-TIMEX3 (maioria intra-sentença)
                    B. EVENT-DCT  
                    C. EVENT-EVENT (inter-sentença)
                isentenca: id_sentenca
                doc: nome do arquivo do corpus, representa um documento
                rule: código da regra que previu o tipo de relação
                lid: ID do TLINK. Se não for fornecido, é calculado automaticamente, último + 1.

            '''
            #Valida eventID
            if not self.__is_event_doc(eventID):
                print("ERROR: O eventID '{0}' da relação: '{4}', da task: '{2}' e relatedTo: '{3}' não válido para o documento {1}.".format(eventID, self.__tb.nome_doc_unico, task, relatedTo, relType))
                return

            #Valida task
            task = task.upper()
            if task in ['A', 'B']:
                relatedToEvent = None
                relatedToTime = relatedTo
                #Seleciona relatedTo conforme task
                if task == 'A':
                    if not self.__is_timex3_doc(relatedToTime):
                        print("ERROR: O relatedToTime '{0}' da relação: '{4}' e eventID: '{3}', para a task: '{2}' deve ter tag igual a TIMEX3 válida no documento {1}.".format(relatedToTime, self.__tb.nome_doc_unico, task, eventID, relType))
                        return
                if task == 'B':
                    if not self.__is_dct_doc(relatedToTime):
                        print("ERROR: O relatedToTime '{0}' da relação: '{4}' e eventID: '{3}', para a task: '{2}' deve ter tag igual a DCT válido no documento {1}.".format(relatedToTime, self.__tb.nome_doc_unico, task, eventID, relType))
                        return
            elif task in ['C']:
                relatedToTime = None
                relatedToEvent = relatedTo
                #Valida relatedToEvent
                if not self.__is_event_doc(relatedToEvent):
                    print("ERROR: O relatedToEvent '{0}' da relação: '{4}', eventID: '{3}' e task: '{2}' não é válido para o documento {1}.".format(relatedToEvent, self.__tb.nome_doc_unico, task, eventID, relType))
                    return
            else: 
                print("ERROR: A task '{2}' informada para o eventID: '{0}' e relatedTo '{1}' não é válida. Task válidas: A, B, e C.".format(eventID, relatedTo, task))
                return

            #valida relType
            relType = relType.upper()
            
            relType_validos = self.relType_validos
            if relType not in relType_validos: #permitir adicionar as relações disjuntivas
                print("ERROR: A relação: '{0}' de eventID: '{2}', relatedTo: '{3}' e Task: '{4}' não é um tipo de relação válida. \n       Valores válidos: {1}".format(relType, str(relType_validos), eventID, relatedTo, task) )
                return

            #se lid não for informado, pega o último + 1
            if not lid:
                lid = self.__next_lid

            #Verifica se já existe
            #Impede inserir eventID e relatedTo repetidos
            text = [lid, relType, eventID, relatedToTime, relatedToEvent, task, isentenca, doc, rule]
            esta_na_lista, text_encontrado = self.__has_list(text)
            if not esta_na_lista:
                self.to_list.append(text)
            else:
                print("WARNING: Está tentando adicionar: id_sentenca: {3}, eventID: '{0}', relatedTo: '{1}', task: '{2}', rule {4}. POREM JÁ EXISTE COMO: relType: '{5}', lid: '{6}', rule: '{7}'.".format(eventID, relatedTo, task, isentenca, rule, text_encontrado[self.RELTYPE], text_encontrado[self.LID], text_encontrado[self.RULE]))

        
        def remove(self, relType, eventID, relatedTo, task, isentenca, doc, rule, lid = None):
            '''
            Remove TLink da estrutura de dados.
            Busca par eventID e relatedTo e apaga pelo lid encontrado.
            
            '''
            if task in ['A', 'B']:
                relatedToEvent = None
                relatedToTime = relatedTo
            elif task in ['C']:
                relatedToTime = None
                relatedToEvent = relatedTo
            else: 
                print("ERROR: A task '{2}' informada para o eventID: '{0}' e relatedTo '{1}' não é válida. Task válidas: A, B, e C.".format(eventID, relatedTo, task))
                return

            text = [lid, relType, eventID, relatedToTime, relatedToEvent, task, isentenca, doc, rule]
            esta_na_lista, text_encontrado = self.__has_list(text)
            if esta_na_lista:
                if self.__remove_list(text_encontrado[self.LID]):
                    print("Dados encontrado com o lid: '{0}'. Foi removido com sucesso.".format(text_encontrado[self.LID]))
            else:
                print("INFO: eventID: '{0}', relatedTo: '{1}' e task: '{2}' não pode ser removido pois não se encontra na lista.".format(eventID, relatedTo, task))


        def save_to_file(self, file_tlink, sobrescrever = False):
            '''
            Salva as tags TLINK em arquivo.

            Args:
                file_tlink: Nome do arquivo tml que conterá tags TLINK.

                sobrescrever:   Se True, sobrescreve o arquivo file_tlink se ele existir, se não existir, cria-o.
                                Se False, se o arquivo existir, não sobrescreve, não faz nada. Se o arquivo não existir, cria-o.

            '''
            if not sobrescrever:
                if os.path.isfile(file_tlink):
                    print("Arquivo já existente. \nPara sobrescrever, use o parametro 'sobrescrever = True'")
                    return

            try:
                with open(file_tlink, 'w') as arq:
                    arq.write('<?xml version="1.0" encoding="UTF-8" ?>\n\n')
                    arq.write('<TempEval>\n\n')

                    for linha in self.to_txt:
                        arq.write(linha + '\n')

                    arq.write('\n</TempEval>\n')
            except Exception as e:
                print(f'Ocorreu erro ao salvar o arquivo {file_tlink}. ERRO: {e}')
            else:
                print('Dados salvo com sucesso em ' + file_tlink)

        def load_from_file(self, file_tlink, modo = 'w'):
            '''
            ######### PROVAVELMENTE ESTE MÉTODO SERÁ EXCLUIDO: ANALISAR ISSO DEPOIS
            
            Carrega dados do arquivo salvo dados pelo método save_to_file()

            Args:
                file_tlink: Arquivo tml contendo tags TLINK criado pelo método save_to_file()

                modo:   se 'w' (write), limpa as carga anterior de self.to_list, sobrescreve conteúdo já carregado.
                        se 'a' (append), adiciona a carga atual no final da carga existente.

            '''
            if not os.path.isfile(file_tlink):
                print('Arquivo não existe.')
                return

            if modo == 'w':
                self.clear()

            try:
                # campos que não tem no arquivo
                isentenca = ''
                doc = ''
                rule = ''  
                for linha in self.__parse_arq_tlink(file_tlink):
                    if modo == 'w':
                        lid = linha['lid']
                    else:
                        lid = None

                    if (linha['task'] in ['A', 'B']):
                        self.add(linha['relType'], linha['eventID'], linha['relatedToTime'], linha['task'], isentenca, doc, rule, lid)
                    elif (linha['task'] == 'C'):
                        self.add(linha['relType'], linha['eventID'], linha['relatedToEvent'], linha['task'], isentenca, doc, rule, lid)
            except Exception as e:
                print(f'Ocorreu erro ao carregar o arquivo {file_tlink}. ERRO {e}.')
            else:
                print('Dados do arquivo ' + file_tlink + ' foram carregados com sucesso.')
    
        
        def clear(self):
            '''
            Limpa todas as tags TLink adicionadas.
            '''
            self.to_list.clear()

        @property
        def to_list(self):
            '''
            Estrutura de dados utilizada para armazenar as tags TLink.
            Utilizada para criar DataFrames e alimentar impressão gráfica das relações temporais.
            
            '''
            return self.__tlink_list

        @property
        def to_txt(self):
            '''
            Lista tags TLink em formato padrão das tags TLINK dos arquivos do corpus.
            Utilizada para salvar em arquivo.
            
            '''
            tlink_txt = []
            for index, row in self.to_df.iterrows():
                if (row['task'] == 'A') or (row['task'] == 'B'):
                    tlink_txt.append('<TLINK lid="' + row['lid'] + '" relType="' + row['relType'] + '" eventID="' + row['eventID'] + '" relatedToTime="' + row['relatedToTime'] + '" task="' + row['task'] + '"/>')
                elif (row['task'] == 'C'):
                    tlink_txt.append('<TLINK lid="' + row['lid'] + '" relType="' + row['relType'] + '" eventID="' + row['eventID'] + '" relatedToEvent="' + row['relatedToEvent'] + '" task="' + row['task'] + '"/>')
            return tlink_txt

        @property
        def to_df(self):
            '''
            Lista tags TLink extraídas em formato DataFrame
            '''
            df_tlink = pd.DataFrame(self.to_list, columns=['lid', 'relType', 'eventID', 'relatedToTime', 'relatedToEvent', 'task', 'isentenca', 'doc', 'rule'])
            return df_tlink

        @property
        def to_df_join(self): 
            '''
            Retorna DataFrame de MyTlink contendo os dados principais das chaves estrangeiras.

            '''
            if not self.__tb:
                print('Para utilizar este método, é necessário informar o caminho dos arquivos do corpus TimebankPT. \nReferencie a classe com o parâmetro path_tml.')
                return

            if not self.__tb.nome_doc_unico:
                print('Para utilizar este método, é necessário informar id_sentenca à classe TimebankPT.')
                return

            #df principais filtrados
            df_tlink  = self.__tb.my_tlink.to_df
            df_event  = self.__tb.df.event_completo
            df_timex3 = self.__tb.df.timex3_completo

            #df com PK renomeado para join
            df_event_eventID = df_event.rename(columns={'eid': 'eventID'})
            df_timex3_relatedToTime = df_timex3.rename(columns={'tid': 'relatedToTime'})
            df_event_relatedToEvent = df_event.rename(columns={'eid': 'relatedToEvent'})
            df_timex3_anchorTimeID = df_timex3.drop(columns=['anchorTimeID']).rename(columns={'tid': 'anchorTimeID'}) #apaga coluna anchorTimeID original antes de renomear as PK com o mesmo nome

            df_event = None
            df_timex3 = None

            #JOIN eventID
            df_join_eventID = df_tlink.merge(df_event_eventID, on=['eventID', 'doc'], how='left', suffixes=('_TLINK', '_EVENTID'))
            df_tlink = None

            #JOIN relatedToTime
            df_join_event_relatedToTime = df_join_eventID.merge(df_timex3_relatedToTime, on=['relatedToTime', 'doc'], how='left', suffixes=('_EVENTID', '_RTOTIME'))
            df_join_eventID = None

            #JOIN relatedToEvent
            df_join_event_relatedToTime_relatedToEvent = df_join_event_relatedToTime.merge(df_event_relatedToEvent, on=['relatedToEvent', 'doc'], how='left', suffixes=('_RTOTIME', '_RTOEVENT'))
            df_join_event_relatedToTime = None

            #JOIN anchorTimeID
            col_join = ['lid', 'rule', 'relType', 'task', 'doc',
                    'eventID', 'isentenca_EVENTID', 'text_EVENTID', 'class_RTOTIME', 'tense_RTOTIME', 'pos_RTOTIME',
                    'relatedToTime', 'isentenca_RTOTIME', 'tag_RTOTIME', 'type_RTOEVENT', 'value_RTOEVENT', 'text_RTOTIME', 
                    'relatedToEvent', 'isentenca_RTOEVENT', 'text_RTOEVENT', 'class_RTOEVENT', 'tense_RTOEVENT', 'pos_RTOEVENT',
                    'anchorTimeID', 'isentenca', 'tag_ANCHOR', 'type_ANCHOR', 'value_ANCHOR', 'text_ANCHOR']
            df_join_event_relatedToTime_relatedToEvent_anchorTimeID = df_join_event_relatedToTime_relatedToEvent.merge(df_timex3_anchorTimeID, on=['anchorTimeID', 'doc'], how='left', suffixes=('_RTOEVENT', '_ANCHOR'))[col_join]
            df_join_event_relatedToTime_relatedToEvent = None

            #Renomeia cnome das colunas para facilitar a visualização
            col_rename = {'isentenca_EVENTID': 'isentenca', 'text_EVENTID': 'text', 'class_RTOTIME': 'class', 'tense_RTOTIME': 'tense', 'pos_RTOTIME': 'pos', 
                    'isentenca_RTOTIME': 'isentenca_rt', 'tag_RTOTIME': 'tag_rt', 'type_RTOEVENT': 'type_rt', 'value_RTOEVENT': 'value_rt', 'text_RTOTIME': 'text_rt',
                    'isentenca_RTOEVENT': 'isentenca_re', 'text_RTOEVENT': 'text_re', 'class_RTOEVENT': 'class_re', 'tense_RTOEVENT': 'tense_re', 'pos_RTOEVENT': 'pos_re', 
                    'isentenca': 'isentenca_at', 'tag_ANCHOR': 'tag_at', 'type_ANCHOR': 'type_at', 'value_ANCHOR': 'value_at', 'text_ANCHOR': 'text_at'}
            df_join_event_relatedToTime_relatedToEvent_anchorTimeID.rename(columns=col_rename, inplace=True)

            return df_join_event_relatedToTime_relatedToEvent_anchorTimeID.sort_values(['isentenca','task', 'eventID', 'relatedToTime', 'relatedToEvent'])

        
        #--------------------------
        # PUBLIC GRAPH MYTLINK
        #--------------------------
        
        def tabela_id_timebank(self):
            '''
            Retorna tabela contendo todas entidades EVENT e TIMEX3 da sentença e seus respectivos IDs.

            '''
            print(tabulate( sorted(self.__struct_id_timebank, key=lambda item : item.get('ent_type')), headers="keys", tablefmt='presto' ))


        def graph_rt(self, compact = True, punct = False):
            '''
            Exibe as relações temporais em forma gráfica.
            
            '''
            self.__clear_struct()
            self.__load_words()
            self.__load_arcs()
            
            spacy.displacy.render(self.__struct_graph, style="dep", jupyter=True, manual=True, options={'distance':120, 'compact':compact, 'word_spacing':30, 'collapse_punct': not punct, 'arrow_spacing': 20, 'arrow_width': 6})
        
        
        def lista_id_timebank(self, task):
            '''
            Retorna dicionário contendo pares conforme task.
            
            Args:
                task:   A. EVENT-TIMEX3 (intra-sentença)
                        B. EVENT-DCT 
                        C. EVENT-EVENT (inter-sentença consecutivas) 
            '''
            def p_cartesiano(lista1, lista2 = None):
                '''
                Produto cartesiana entre as duas listas
                '''
                tuplas_ids = []
                
                pares = product(lista1, lista2)
                #pares = combinations(lista1, 2)

                for elem in pares:
                    if elem[0] != elem[1]:
                        tuplas_ids.append(elem)
                return tuplas_ids


            list_eventID = []
            list_relatedToTime = []
            list_dct = []
            list_eventID_next = []
            
            for item in self.__struct_id_timebank:
                if item['ent_type'] == 'EVENT':
                    list_eventID.append(item['id_tag'])
                elif item['ent_type'] == 'TIMEX3':
                    list_relatedToTime.append(item['id_tag'])
                elif item['ent_type'] == 'DCT':
                    list_dct.append(item['id_tag'])
            
            if not list_eventID:
                print('ERROR: list_eventID está vazia para id_sentenca: ' + str(self.__tb.id_sentenca_unica[0]))
            
            #task A
            if task == 'A':
                if not list_relatedToTime:
                    print('ERROR: list_relatedToTime está vazia para id_sentenca: ' + str(self.__tb.id_sentenca_unica[0]))
                
                pares = p_cartesiano(list_eventID, list_relatedToTime)
            
            #task B
            elif task == 'B':
                if not list_dct:
                    print('ERROR: list_dct está vazia.')
                
                pares = p_cartesiano(list_eventID, list_dct)
                
            #task C
            elif task == 'C':
                #task C = EVENT x EVENT entre sentença consecutivas
                
                list_id_sentencas_doc = self.__tb.id_sentencas_doc
                id_sentenca_next = self.__tb.id_sentenca_unica[0] + 1
                
                if id_sentenca_next in list_id_sentencas_doc:
                    self.__tb.id_sentenca = id_sentenca_next
                    for item in self.__struct_id_timebank:
                        if item['ent_type'] == 'EVENT':
                            list_eventID_next.append(item['id_tag'])
                    self.__tb.id_sentenca = id_sentenca_next - 1
                    
                    if not list_eventID_next:
                        print('ERROR: list_relatedToEvent está vazia.')
                else:
                    print('Última sentença do documento, não tem pares para task C.')
                    
                pares = p_cartesiano(list_eventID, list_eventID_next)
                
            else:
                print("'{0}' é uma task inválida.".format(task) )
                return
                
            return {'list_eventID': list_eventID, 'list_relatedToTime': list_relatedToTime, 'list_dct':list_dct, 'list_relatedToEvent': list_eventID_next, 'pares':pares}
        
        
        def __idtag_to_token_helper(self, id_tag: str, proxima_sentenca: bool) -> Token:
            '''
            Converte id_tag em token do Doc atual
            '''
            if not self.__tb.id_sentenca:
                print('ERROR: É necessário atribuir id_sentenca à instancia da classe TimebankPT.')
                return 
            
            token_encontrado = None
            id_sentenca = self.__tb.id_sentenca_unica[0]
            
            #Vai para a próxima sentença
            if proxima_sentenca:
                self.__tb.id_sentenca = id_sentenca + 1
            
            for token in self.__tb.doc_unico.ents:
                token = token[0]  #  ents é span, neste caso, cada span tem apenas um token
                if token._.id_tag == id_tag:
                    token_encontrado = token
                    break
            
            #Retorna à sentença original
            if proxima_sentenca:
                self.__tb.id_sentenca = id_sentenca
                
            return token_encontrado

        
        def idtag_to_token(self, id_tag: str) -> Token:
            return self.__idtag_to_token_helper(id_tag, proxima_sentenca = False)
        
        def idtag_to_token_next(self, id_tag: str) -> Token:
            return self.__idtag_to_token_helper(id_tag, proxima_sentenca = True)


        
        #----------------------------------
        # ------- PRIVADAS MYTLINK --------
        #----------------------------------
        
        def __is_event_doc(self, eid: str):
            '''
            Verifica se eid é um EVENT válido do documento atual.
            '''
            return not self.__tb.df.event_doc.query("eid == '" + eid + "'").empty

        def __is_timex3_doc(self, tid: str):
            '''
            Verifica se tid é um TIMEX3 válido do documento atual. 
            '''
            return not self.__tb.df.timex3_doc.query("tid == '" + tid + "' and tag == '" + 'TIMEX3' + "'").empty

        def __is_dct_doc(self, tid: str):
            '''
            Verifica se tid é um DCT válido do documento atual. 
            '''
            return not self.__tb.df.timex3_doc.query("tid == '" + tid + "' and tag == '" + 'DCT' + "'").empty

        
        def __has_list(self, text):
            '''
            Verifica se text está presente na lista de TLINKs (to_list).
            Verifica apenas os atributos que formam chave primária: (eventID e (relatedToTime ou relatedToEvent) )
            
            Args:
                text: É uma lista com os atributos [lid, relType, eventID, relatedToTime, relatedToEvent, task, isentenca, doc, rule]
                
            Return:
                encontrou: True se já existir na lista de TLINKs e
                Lista contendo valores dos atributos de 'text' encontrados.
            
            '''
            #text = [lid, relType, eventID, relatedToTime, relatedToEvent, task, isentenca, doc, rule]

            relatedTo = self.TOTIME
            if text[self.TASK] in ['A', 'B']:
                relatedTo = self.TOTIME
            elif text[self.TASK] in ['C']:
                relatedTo = self.TOEVENT

            for l in self.to_list:
                if (l[self.EVENTID] == text[self.EVENTID]) and (l[relatedTo] == text[relatedTo]) and (l[self.DOC] == text[self.DOC]):
                    return True, [ l[self.LID], l[self.RELTYPE], l[self.EVENTID], l[self.TOTIME], l[self.TOEVENT], l[self.TASK], l[self.ISENTENCA], l[self.DOC], l[self.RULE] ]
                    
            return False, []

        
        def __remove_list(self, lid):
            '''
            Remove o TLink com id = lid da estrutura de dados.
            
            Args:
                lid: id de TLINK
                
            '''
            r = False
            for l in self.to_list:
                if l[self.LID] == lid:
                    r = True
                    self.to_list.remove(l)
                    break
            return r

        def __parse_arq_tlink(self, file_tlink):
            '''
            Retorna dicionário das tags Tlink do arquivo TimeML

            Args:
                file_tlink: arquivo tml contendo apenas tags TLINK gerada pelo método save_to_file()

            '''
            if not os.path.isfile(file_tlink):
                return

            tlink = []

            xtree = et.parse(file_tlink)
            root = xtree.getroot()

            for node in list(root):
                if node.tag == 'TLINK':
                    tlink.append(node.attrib)

            return tlink

        @property
        def __next_lid(self):
            '''
            Retorna próximo lid de TLINK
            '''
            lista = self.to_df['lid'].tolist()
            id_max = 0
            if lista:
                id_max = max(map(lambda l: int(l[1:]), lista))
            lid = 'l' + str(id_max + 1)
            return lid

        
        #----------------------------
        # PRIVATE GRAPH MYTLINK
        #----------------------------
        
        @property
        def __struct_words(self):
            '''
            Retorna estrutura de dados contendo as "words" utilizadas para imprimir gráficos das relações temporais.
            
            '''
            return self.__struct['words']

        @property
        def __struct_arcs(self):
            '''
            Retorna estrutura de dados contendo os "arcs" utilizadas para imprimir gráficos das relações temporais.
            '''
            return self.__struct['arcs']

        @property
        def __struct_graph(self):
            '''
            Retorna estrutura de dados no padrão do displaCy dep para composição manual de árvore de dependência customizada.
            Neste caso, a árvore será composta de Relações Temporais.
            
            estrutura_padrão = 
            {
                "words": [
                    {"text": "This", "tag": "DT"},
                    {"text": "is", "tag": "VBZ"},
                    {"text": "a", "tag": "DT"},
                    {"text": "sentence", "tag": "NN"}
                ],
                "arcs": [
                    {"start": 0, "end": 1, "label": "nsubj", "dir": "left"},
                    {"start": 2, "end": 3, "label": "det", "dir": "left"},
                    {"start": 1, "end": 3, "label": "attr", "dir": "right"}
                ]
            }
            
            '''
            return self.__struct

        def __clear_struct(self):
            '''
            Limpa dados da estrutura de dados que armazena informações para compor o gráfico de relações.
            
            '''
            self.__struct_words.clear()
            self.__struct_arcs.clear()
            
        
        def __idtag_to_i(self, id_tag):
            '''
            Converte as id_tag do timebankpt em i que representa o índice do token na sentença.

            Args: 
                id_tag: código dos EVENT, TIMEX3 e DCT no corpus. Ex: ('e12', 't25', 't40')

            '''
            for tag in self.__struct_id_timebank:
                if tag['id_tag'] == id_tag:
                    return int(tag['i'])
                

        def __has_list_arcs(self, text):
            '''
            Verifica se text está presente na estrutura de dados que armazena dados para o compor o gráfico de relações temporais (__struct_arcs).
            Verifica apenas os atributos que formam chave primária: (start e end).
            
            Args:
                text: É um dicionário que representa uma linha da estrutura contendo os atributos: {"start": start, "end": end, "label": label, "dir": direcao}
                
            Return:
                encontrou (bool)
                
            '''
            encontrou = False
            for l in self.__struct_arcs:
                if (l['start'] == text['start']) and (l['end'] == text['end']):
                    encontrou = True
                    break
            return encontrou

        def __load_words(self):
            '''
            Carrega a estrutura de dados __struct_words com todas as palavras (words) da sentença.
            
            '''
            if self.__tb.doc_unico:
                for token in self.__tb.doc_unico:
                    self.__add_words(token.text, token.pos_)

        def __load_arcs(self):
            '''
            Carrega a estrutura de dados __struct_words com todos os arcos (Relações Temporais) extraídas.
            
            '''
            # l = ['l1', 'BEFORE', 'e13', 't53', None, 'A']
            LID = 0
            RELTYPE = 1
            EVENTID = 2
            TOTIME = 3
            TOEVENT = 4
            TASK = 5

            for l in self.to_list:
                relatedTo = TOTIME
                if l[TASK] in ['A', 'B']:
                    relatedTo = TOTIME
                elif l[TASK] in ['C']:
                    relatedTo = TOEVENT

                start = l[EVENTID]
                end = l[relatedTo]
                label = l[RELTYPE]

                self.__add_arcs(start, end, label)


        def __add_words(self, text, tag):
            '''
            Adiciona uma 'word' que é composta por text e tag à estrutura de dados __struct_words.
            
            '''
            text = {'text': text, 'tag':tag}
            self.__struct_words.append(text)

        def __add_arcs(self, start, end, label):
            '''
            Adiciona um 'arc' que é composta por start, end e label à estrutura de dados __struct_arcs.
            Um arcs representa uma relação temporal.
            
            '''
            #valida label
            label = label.upper()
            label_validos = self.relType_validos
            if label not in label_validos:
                print('ERROR: {0} de start: {2} e end: {3} não é um tipo de relação válida. \n       Valores válidos: {1}'.format(label, str(label_validos), start, end) )
                return

            #Valida dir
            if label in ['AFTER', 'OVERLAP-OR-AFTER']:
                direcao = 'left'
            elif label in ['BEFORE', 'BEFORE-OR-OVERLAP']:
                direcao = 'right'
            else:  #['OVERLAP', 'VAGUE']
                direcao = 'right'

            #valida start e end
            start_ori = start
            if type(start) == str:
                start = self.__idtag_to_i(start)
                if start is None:
                    print("ERROR start: id_tag '{0}' não encontrada".format(start_ori))
                    return

            end_ori = end
            if type(end) == str:
                end = self.__idtag_to_i(end)
                if end is None:
                    print("ERROR end: id_tag '{0}' não encontrada".format(end_ori))
                    return

            if start == end:
                print('ERROR: start: {1} não pode ser igual a end: {2}. Label {0}.'.format(label, start_ori, end_ori) )
                return
            if start > end:
                start, end = end, start

            #Impedir repetidos verificando apenas start e end
            text = {"start": start, "end": end, "label": label, "dir": direcao}
            if not self.__has_list_arcs(text):
                self.__struct_arcs.append(text)
            else:
                print('WARNING: start: {0} e end: {1} já foi adicionado'.format(start_ori, end_ori))

        @property
        def __struct_id_timebank(self):
            '''
            Retorna estrutura de dados contendo todas entidades EVENT e TIMEX3 da sentença com informações sobre IDs.
            
            '''
            struct = []
            
            if not self.__tb.id_sentenca:
                print('ERROR: É necessário atribuir id_sentenca à instância da classe TimebankPT.')
                return []
            
            #add EVENT e TIMEX3
            for ent in self.__tb.doc_unico.ents:
                if ent.label_ in ('EVENT', 'TIMEX3'):
                    for token in ent:
                        struct.append({'i':token.i, 'id_tag':token._.id_tag, 'token':token.text, 'ent_type':token.ent_type_})
            
            #add DCT
            id_tag = self.__tb.df.timex3_doc.query("tag == 'DCT'")['tid'].tolist()[0]
            valor_dct = self.__tb.df.timex3_doc.query("tag == 'DCT'")['value'].tolist()[0]
            struct.append({'i':None, 'id_tag':id_tag, 'token':valor_dct, 'ent_type':'DCT'})
            
            return struct


    #---------------------------------------------
    #-----  FIM  class MyTlink  ------------------
    #---------------------------------------------

#---------------------------------------------------------------------
# FIM  class Timebank
#--------------------------------------------------------------------



# '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
#  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
#  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
#  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
#  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
#  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
#  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
# ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #

#---------------------------------------------------------------------
#     CLASSE FEATUREStoDATASET
#--------------------------------------------------------------------

class FeaturesToDataset():
    '''
    Retorna todos os pares event-time, anotados e não anotados, e todas as features utilizadas para gerar regras

    Args:
        tb: instancia da class TimebankPT
    '''
    def __init__(self, tb: TimebankPT):
        self.__tb = tb
        
        #self.__id_sentenca = [3, 29, 30, 643, 661, 669, 1822, 1825, 1826, 1827, 1828, 1829, 1830, 1835, 1836, 1837]
        self.__id_sentenca = self.__tb.id_sentencas_task('A')
        
        self.__df_features = pd.DataFrame()
        self.__filtra_sentenca_sem_predicao = False
        
    @property
    def filtra_sentenca_sem_predicao(self):
        '''
        Se True, filtrar dataset exibindo apenas as sentenças que não houve predição.
        Se False, exibe dataset completo.
        '''
        return self.__filtra_sentenca_sem_predicao
    
    @filtra_sentenca_sem_predicao.setter
    def filtra_sentenca_sem_predicao(self, filtrar: bool):
        if type(filtrar) != bool:
            raise ValueError(f'ERRO: Atribuição inválida para a propriedade filtra_sentenca_sem_predicao: {filtrar}. \nDeve receber valor booleano.')
        if filtrar == True:
            print('Filtro Ativado:', self.__tb.tr.status_resumido())
        #else:
        #    print('Filtro Desativado.')
        self.__filtra_sentenca_sem_predicao = filtrar
    

    def dataset(self):
        '''
        Retorna dataframe com todas as features. Anotadas e não anotadas; train, train_test e test.
        Não processa novamente.
        '''
        if self.__df_features.empty:
            raise ValueError('As features ainda não foram processadas. \nÉ necessário chamar o método "create_dataset() ou carregar dataset salvo com load_dataset()"')
        
        #Essa substituição faz os algoritmos de aprendizado de regras trabalho melhor
        #Deixando o valor 'NONE', atributos com esse valor não são incluídos nas regras geradas
        self.__df_features.replace('NONE', None, inplace=True)

        if self.filtra_sentenca_sem_predicao: #[ ] monitorar filtrar
            id_sentencas_sem_predicao = self.__tb.tr.id_sentencas_sem_predicao
            return self.__df_features.query("isentenca in @id_sentencas_sem_predicao")
        else:
            return self.__df_features
    
    
    def __dataset_colunas(self, so_anotados: bool = True):
        '''
        Trata as colunas de self.dataset() para os dados de treino e de testes
        '''
        colunas = self.dataset().columns.to_list()
        colunas_excluir = ['id', 'relType', 'train_test', 'task', 'eventID', 'e_text', 'relatedTo', 't_text', 'isentenca', 'doc']
        for excluir in colunas_excluir:
            colunas.remove(excluir)
        colunas.append('relType')

        if so_anotados:
            colunas.remove('anotado')
            
        return colunas
        
    def dataset_treino(self, so_anotados: bool = True) -> DataFrame:
        '''
        Retorna dataframe com os dados de treino. Inclui a classe.
        
        Args:
            so_anotados: Se True, retorna apenas os dados em que o tipo da relação temporal é anotado.
                        Se False, retorna os anotados e os não anotados
        '''
        if so_anotados:
            dataset = self.dataset().query('anotado == True')
        else:
            dataset = self.dataset()
        colunas = self.__dataset_colunas(so_anotados)
        
        if self.__tb.dev:
            query_train_test = "train_test == 'train'"
        else:
            query_train_test = "train_test.str.startswith('train')"

        result = dataset.query(query_train_test, engine='python')[colunas]
        return result
        
    
    def dataset_teste(self, so_anotados: bool = True) -> DataFrame:
        '''
        Retorna dataframe com os dados de teste. Inclui a classe.
        
        Args:
            so_anotados: Se True, retorna apenas os dados em que o tipo da relação temporal é anotado.
                        Se False, retorna os anotados e os não anotados.
        '''
        if so_anotados:
            dataset = self.dataset().query('anotado == True')
        else:
            dataset = self.dataset()
        colunas = self.__dataset_colunas(so_anotados)

        if self.__tb.dev:
            query_train_test = "train_test == 'train_test'"
        else:
            query_train_test = "train_test == 'test'"

        result = dataset.query(query_train_test)[colunas]
        return result
    

    @property
    def X_train(self):
        return self.dataset_treino().drop(['relType'], axis=1)
    
    @property
    def y_train(self):
        return self.dataset_treino()['relType']
        
    @property
    def X_test(self):
        return self.dataset_teste().drop(['relType'], axis=1)
        
    @property
    def y_test(self):
        return self.dataset_teste()['relType']
        
    @property
    def X_train_encoder(self):
        return self.df_encoder(self.X_train)
    

    def __select_best_features_chi2(self, k: int = None) -> pd.core.indexes.base.Index:
        '''
        Retorna as k melhores features dos dados de treino, segundo cálculo do chi2 (qui-quadrado).

        Args:
            k:  Quantidades de features retornadas.
                Se não informada, retorna todas as features em ordem de importância.
        '''
        if not k:
            k = len(self.X_train_encoder.columns)

        model = SelectKBest(score_func=chi2, k=k)
        fit = model.fit(self.X_train_encoder, self.y_train)
        cols = fit.get_support(indices=True)
        #features = fit.transform(self.X_train_encoder)
        return self.X_train_encoder.iloc[:, cols].columns.to_list()
    
    def __select_best_features_rfe(self, k: int = None, cv: int = 5) -> pd.core.indexes.base.Index:
        '''
        Retorna as melhores features dos dados de treino, utilizado a técnica de Eliminação Recursiva de Features com Validação Cruzada.

        Args:
            k:  Quantidades de features retornadas.
                Se não informada, retorna as features selecionadas pelo algoritmo.
            cv: Quantidade de folds (default = 5).
        '''
        estimator = DecisionTreeClassifier(random_state=42)
        rfecv = RFECV(estimator=estimator, cv=cv)
        X_new = rfecv.fit_transform(self.X_train_encoder, self.y_train)

        if not k:
            k = np.count_nonzero(rfecv.support_ == True)

        # Obtenha as features selecionadas
        #selected_features = X_train_encoder.columns[rfecv.support_]

        # Escolha um número de features com base na importância relativa
        feature_ranks = pd.Series(rfecv.ranking_, index=self.X_train_encoder.columns)
        selected_features = feature_ranks.nsmallest(k).index
        return selected_features.to_list()

    def select_best_features(self, k: int = None, metodo: str = 'rfe', cv: int = 5) -> pd.core.indexes.base.Index:
        '''
        Retorna as melhores features calculadas sobre os dados de treino. 
        Se método 'chi2' as melhores segundo cálculo do chi2 (qui-quadrado)
        Se 'rfe' segundo técnica de Eliminação Recursiva de Features com Validação Cruzada (RFECV)
        
        Args:
            k:  Quantidades de features retornadas.
                Se k não informado: 
                    . retorna todas as features em ordem de importância, se metodo for 'chi2'.
                    . retorna as features selecionadas pelo algoritmo, se método for 'rfe'.
            metodo: 'chi2' - Estatística qui-quadrado
                    'rfe'  - Eliminação Recursiva de Features com Validação Cruzada (RFECV) (default)
            cv: Quantidade de folds (default = 5).
        '''
        metodos = ['rfe', 'chi2']
        if metodo not in metodos:
            raise ValueError(f"{metodo} é método inválido. Válidos: {str(metodos)}")

        if metodo == 'chi2':
            return self.__select_best_features_chi2(k)
        elif metodo == 'rfe':
            return self.__select_best_features_rfe(k=k, cv=cv)


    def df_encoder(self, df: DataFrame) -> DataFrame:
        '''Aplica LabelEncoder em df'''
        df_categoric = df.select_dtypes(include=['object', 'category'])
        df_encoded = df_categoric.apply(LabelEncoder().fit_transform)
        return pd.concat([df.select_dtypes(exclude=['object', 'category']), df_encoded], axis=1)
    

    def to_csv(self, nome_arquivo: str, so_anotados: bool = True):
        '''
        Dataset contendo features em formato csv.
        Salva dois arquivos. Um de treino e outro de testes conforme divisão pre-existente do corpus.
        
        Args:
            nome_arquivo: nome do arquivo. Salva nome_arquivo_train.csv e nome_arquivo_test.csv
            so_anotados: Se True, salva apenas dados onde o tipo da relação temporal (relType) está anotado.
                        Se False, salva todos.
        '''
        if not nome_arquivo:
            print("É necessário informar o nome do arquivo.")
            return 
        
        nome_arquivo, ext = os.path.splitext(nome_arquivo.strip())
        arq_treino =  nome_arquivo.strip() + '_treino.csv'
        arq_teste = nome_arquivo.strip() + '_teste.csv'
        
        self.dataset_treino(so_anotados).to_csv(arq_treino)
        self.dataset_teste(so_anotados).to_csv(arq_teste)
        
    
    def create_dataset(self):
        '''
        Processa todos os pares event-time e alimenta 'dataset' com as informações linguísticas (features).
        '''
        tipos_dados =  {'relType': 'category', 
                        'event_class': 'category', 
                        'event_closest_to_event_class': 'category', 
                        'event_closest_to_event_equal_class': bool, 
                        'event_pos': 'category', 
                        'event_closest_to_event_pos': 'category', 
                        'event_closest_to_event_equal_pos': bool, 
                        'event_closest_to_event_tense': 'category', 
                        'event_closest_to_event_equal_tense': bool, 
                        'event_closest_to_timex3_pos': 'category', 
                        'event_closest_to_timex3_equal_pos': bool, 
                        'event_conjunction_closest_follow': 'category', 
                        'event_conjunction_closest_precede': 'category', 
                        'event_polarity': 'category', 
                        'event_dep': 'category', 
                        'timex3_dep': 'category', 
                        'timex3_type': 'category', 
                        'timex3_pos': 'category', 
                        'timex3_temporalfunction': bool, 
                        'event_root': bool, 
                        'event_pos_token_1_precede': 'category', 
                        'event_pos_token_1_follow': 'category', 
                        'event_pos_token_2_precede': 'category', 
                        'event_pos_token_2_follow': 'category', 
                        'event_pos_token_3_precede': 'category', 
                        'event_pos_token_3_follow': 'category', 
                        'timex3_pos_token_1_precede': 'category', 
                        'timex3_pos_token_1_follow': 'category', 
                        'timex3_pos_token_2_precede': 'category', 
                        'timex3_pos_token_2_follow': 'category', 
                        'timex3_pos_token_3_precede': 'category', 
                        'timex3_pos_token_3_follow': 'category', 
                        'event_preposition_precede': 'category', 
                        'timex3_preposition_precede': 'category', 
                        'event_timex3_distance': 'category', 
                        'event_first_order': bool, 
                        'event_between_order': bool, 
                        'timex3_between_order': bool, 
                        'event_timex3_no_between_order': bool, 
                        'event_closest_to_event_temporal_direction': 'category', 
                        'event_temporal_direction': 'category', 
                        'timex3_relevant_lemmas': 'category', 
                        'event_gov_verb_aspect': 'category', 
                        'event_gov_verb_tense': 'category', 
                        'timex3_gov_verb_tense': 'category', 
                        'event_head_pos': 'category', 
                        'timex3_head_pos': 'category', 
                        'event_intervening_following_tense': 'category', 
                        'event_intervening_preceding_class': 'category', 
                        'event_head_is_root': bool, 
                        'event_is_ancestor_timex3': bool, 
                        'timex3_head_is_root': bool, 
                        'timex3_is_ancestor_event': bool, 
                        'event_preposition_gov': 'category', 
                        'timex3_preposition_gov': 'category', 
                        'reichenbach_direct_modification': bool, 
                        'reichenbach_temporal_mod_function': bool, 
                        'event_timex3_dep': 'category', 
                        'signal_precede_event_text': 'category', 
                        'signal_precede_timex3_text': 'category', 
                        'signal_precede_event_pos': 'category', 
                        'signal_precede_timex3_pos': 'category', 
                        'signal_precede_event_distance_event': 'category', 
                        'signal_precede_timex3_distance_timex3': 'category', 
                        'signal_precede_event_child_event': 'category', 
                        'signal_precede_timex3_child_timex3': 'category', 
                        'signal_precede_event_dep_if_child_event': 'category', 
                        'signal_precede_timex3_dep_if_child_timex3': 'category', 
                        'reichenbach_tense': 'category', 
                        'event_modal_verb': 'category', 
                        'event_has_modal_verb_precede': bool, 
                        }
        
        try:
            lista_pares = self.__process_pares()
            self.__df_features = pd.DataFrame(lista_pares)

            #essas são bool com str, precisa primeiro tornar str, depois, abaixo, category
            self.__df_features['signal_precede_event_child_event'] = self.__df_features['signal_precede_event_child_event'].astype(str)
            self.__df_features['signal_precede_timex3_child_timex3'] = self.__df_features['signal_precede_timex3_child_timex3'].astype(str)

            #Atribui tipos de dados
            self.__df_features = self.__df_features.astype(tipos_dados)
            #Cria ID
            self.__df_features.reset_index(inplace = True)
            self.__df_features.rename(columns = {'index': 'id'}, inplace = True)
        except Exception as e:
            print(f'ERROR: problemas na geração do dataset. ERRO: {e}')
        else:
            print("Os método 'dataset_treino()' e 'dataset_teste() da classe features já podem ser acessados. \nBem como as propriedades: X_train, y_train, X_test, y_test")
    
    
    def save_dataset(self, nome_arquivo: str):
        '''
        Salva arquivo contendo dataset de features completo processado pelo método 'create_dataset()'.
        Utiliza ext .parquet, mas não é necessário informá-la em nome_arquivo
        '''
        arquivo = self.__tb.check_filename(filename=nome_arquivo, extensao='parquet')
        
        try:
            self.dataset().to_parquet(arquivo)
        except Exception as e:
            print(f'Erro ao salvar arquivo {arquivo}. ERRO: {e}')
        else:
            print(f"Dataset salvo em {arquivo}")
        
    
    def load_dataset(self, nome_arquivo: str):
        '''
        Carrega arquivo contendo dataset de features salvo pelo método 'save_dataset()'.
        Utiliza ext .parquet, mas não é necessário informá-la em nome_arquivo
        '''
        arquivo = self.__tb.check_filename(filename=nome_arquivo, extensao='parquet', check_if_exist = True)
        
        try:
            self.__df_features = pd.read_parquet(arquivo)
        except Exception as e:
            print(f'Erro ao carregar dataset {arquivo}. ERRO: {e}')
        else:
            print(f'Dataset {arquivo} carregado com sucesso.')
    
    def generate_params_functions(self) -> dict:
        ''' 
        Gera dicionário com o nome das funções que implementam as features e seus parâmetros. 
        É necessário gerar o dataset com as features ('TimebankPT.features.create_dataset()') ou 
            carregar dataset salvo ('TimebankPT.features.load_dataset(nome_arquivo)').
        '''
        params_functions = {}
        
        if self.dataset().empty:
            raise ValueError('As features ainda não foram processadas. \nÉ necessário chamar o método "create_dataset() ou carregar dataset com load_dataset()"')
            
        for col in self.X_train.columns:
            params_functions[col] = self.__param_function(col)
        return params_functions

    def __param_function(self, function: str) -> list:
        ''' Retorna os parâmetros da função que implementa uma feature '''
        
        tr = self.__tb.tr  #eval não reconhece variáveis privada, trazemos self.__tb.tr para o escopo da função
        class_features = 'tr.f'
        
        try:
            list_param = list(inspect.signature(eval(class_features + '.' + function)).parameters.keys())
        except AttributeError as e:
            print('ERROR:', e)
            return None
        else:
            return list_param
        
        
    def __process_pares(self) -> list:
        '''
        Retorna lista com todos os pares event-time
        '''
        #Processa predição para cada sentenca
        lista_pares = []
        for id_sentenca in self.__id_sentenca:
            #Atribui cada sentenca recebida como argumento para a classe TimebankPT, 
            #isso faz as funções abaixo responderem conforme a sentença selecionada
            self.__tb.id_sentenca = id_sentenca
            
            #Recebe os pares Event x Timex3 da sentença e 
            #Processa predição para cada par de entidade da sentenca
            for eventID, relatedTo in self.__tb.my_tlink.lista_id_timebank('A')['pares']:
                #converte ids de EVENT e TIMEX3 para Token
                token_eventID  = self.__tb.my_tlink.idtag_to_token(eventID)
                token_relatedTo = self.__tb.my_tlink.idtag_to_token(relatedTo) 
                
                #Alimenta self.__df_features com atributos sintáticos e POS tagger
                lista_pares.append(self.__process_um_par(token_eventID, token_relatedTo))
        return lista_pares
    

    def __process_um_par(self, tokenE: Token, tokenT: Token) -> dict:
        '''
        Retorna dicionário de um par event-time com todas as features
        '''
        
        #features de identificação
        eventID = tokenE._.id_tag
        relatedTo = tokenT._.id_tag
        id_sentenca = self.__tb.id_sentenca[0]
        nome_doc = self.__tb.nome_doc_unico
        train_test = self.__tb.get_train_test(nome_doc)
        e_text  = tokenE.text
        t_text  = tokenT.text
        
        #Tabela com os Tlinks event-time anotados 
        df = self.__tb.df.tlink_completo[['eventID', 'relatedToTime', 'doc', 'task', 'relType']]
        df = df[(df['eventID'] == eventID) & (df['relatedToTime'] == relatedTo) & (df['doc'] == nome_doc) & (df['task'] == 'A')]
        if not df.empty:
            anotado = True
            relType = df.iloc[0].relType
        else:
            anotado = False
            relType = ''
        
        event_class = self.__tb.tr.f.event_class(tokenE)
        event_tense = self.__tb.tr.f.event_tense(tokenE)
        event_pos = self.__tb.tr.f.event_pos(tokenE)
        event_closest_to_event_class = self.__tb.tr.f.event_closest_to_event_class(tokenE)
        event_closest_to_event_pos = self.__tb.tr.f.event_closest_to_event_pos(tokenE)
        event_closest_to_event_tense = self.__tb.tr.f.event_closest_to_event_tense(tokenE)
        event_closest_to_event_equal_lemma = self.__tb.tr.f.event_closest_to_event_equal_lemma(tokenE)
        event_closest_to_event_equal_class = self.__tb.tr.f.event_closest_to_event_equal_class(tokenE)
        event_closest_to_event_equal_pos = self.__tb.tr.f.event_closest_to_event_equal_pos(tokenE)
        event_closest_to_event_equal_tense = self.__tb.tr.f.event_closest_to_event_equal_tense(tokenE)
        event_closest_to_timex3_pos = self.__tb.tr.f.event_closest_to_timex3_pos(tokenT)
        event_closest_to_timex3_equal_pos = self.__tb.tr.f.event_closest_to_timex3_equal_pos(tokenE, tokenT)
        event_conjunction_closest_follow = self.__tb.tr.f.event_conjunction_closest_follow(tokenE)
        event_conjunction_closest_precede = self.__tb.tr.f.event_conjunction_closest_precede(tokenE)
        event_polarity = self.__tb.tr.f.event_polarity(tokenE)
        event_aspect = self.__tb.tr.f.event_aspect(tokenE)
        event_dep = self.__tb.tr.f.event_dep(tokenE)
        timex3_dep = self.__tb.tr.f.timex3_dep(tokenT)
        timex3_type = self.__tb.tr.f.timex3_type(tokenT)
        timex3_pos = self.__tb.tr.f.timex3_pos(tokenT)
        timex3_temporalfunction = self.__tb.tr.f.timex3_temporalfunction(tokenT)
        event_root = self.__tb.tr.f.event_root(tokenE)
        timex3_root = self.__tb.tr.f.timex3_root(tokenT)
        event_pos_token_1_precede = self.__tb.tr.f.event_pos_token_1_precede(tokenE)
        event_pos_token_1_follow = self.__tb.tr.f.event_pos_token_1_follow(tokenE)
        event_pos_token_2_precede = self.__tb.tr.f.event_pos_token_2_precede(tokenE)
        event_pos_token_2_follow = self.__tb.tr.f.event_pos_token_2_follow(tokenE)
        event_pos_token_3_precede = self.__tb.tr.f.event_pos_token_3_precede(tokenE)
        event_pos_token_3_follow = self.__tb.tr.f.event_pos_token_3_follow(tokenE)
        timex3_pos_token_1_precede = self.__tb.tr.f.timex3_pos_token_1_precede(tokenT)
        timex3_pos_token_1_follow = self.__tb.tr.f.timex3_pos_token_1_follow(tokenT)
        timex3_pos_token_2_precede = self.__tb.tr.f.timex3_pos_token_2_precede(tokenT)
        timex3_pos_token_2_follow = self.__tb.tr.f.timex3_pos_token_2_follow(tokenT)
        timex3_pos_token_3_precede = self.__tb.tr.f.timex3_pos_token_3_precede(tokenT)
        timex3_pos_token_3_follow = self.__tb.tr.f.timex3_pos_token_3_follow(tokenT)
        event_preposition_precede = self.__tb.tr.f.event_preposition_precede(tokenE)
        timex3_preposition_precede = self.__tb.tr.f.timex3_preposition_precede(tokenT)
        event_timex3_distance = self.__tb.tr.f.event_timex3_distance(tokenE, tokenT)
        event_first_order = self.__tb.tr.f.event_first_order(tokenE, tokenT)
        event_between_order = self.__tb.tr.f.event_between_order(tokenE, tokenT)
        timex3_between_order = self.__tb.tr.f.timex3_between_order(tokenE, tokenT)
        event_timex3_no_between_order = self.__tb.tr.f.event_timex3_no_between_order(tokenE, tokenT)
        timex3_between_quant = self.__tb.tr.f.timex3_between_quant(tokenE, tokenT)
        event_closest_to_event_temporal_direction = self.__tb.tr.f.event_closest_to_event_temporal_direction(tokenE)
        event_temporal_direction = self.__tb.tr.f.event_temporal_direction(tokenE)
        timex3_relevant_lemmas = self.__tb.tr.f.timex3_relevant_lemmas(tokenT)
        event_gov_verb_aspect = self.__tb.tr.f.event_gov_verb_aspect(tokenE)
        event_gov_verb_tense = self.__tb.tr.f.event_gov_verb_tense(tokenE)
        timex3_gov_verb_tense = self.__tb.tr.f.timex3_gov_verb_tense(tokenT)
        event_head_pos = self.__tb.tr.f.event_head_pos(tokenE)
        timex3_head_pos = self.__tb.tr.f.timex3_head_pos(tokenT)
        event_intervening_following_tense = self.__tb.tr.f.event_intervening_following_tense(tokenE, tokenT)
        event_intervening_preceding_class = self.__tb.tr.f.event_intervening_preceding_class(tokenE, tokenT)
        event_gov_verb = self.__tb.tr.f.event_gov_verb(tokenE)
        timex3_gov_verb = self.__tb.tr.f.timex3_gov_verb(tokenT)
        event_head_is_root = self.__tb.tr.f.event_head_is_root(tokenE)
        event_is_ancestor_timex3 = self.__tb.tr.f.event_is_ancestor_timex3(tokenE, tokenT)
        event_is_child_timex3 = self.__tb.tr.f.event_is_child_timex3(tokenE, tokenT)
        timex3_head_is_root = self.__tb.tr.f.timex3_head_is_root(tokenT)
        timex3_is_ancestor_event = self.__tb.tr.f.timex3_is_ancestor_event(tokenE, tokenT)
        timex3_is_child_event = self.__tb.tr.f.timex3_is_child_event(tokenE, tokenT)
        event_preposition_gov = self.__tb.tr.f.event_preposition_gov(tokenE)
        timex3_preposition_gov = self.__tb.tr.f.timex3_preposition_gov(tokenT)
        reichenbach_direct_modification = self.__tb.tr.f.reichenbach_direct_modification(tokenE, tokenT)
        reichenbach_temporal_mod_function = self.__tb.tr.f.reichenbach_temporal_mod_function(tokenE, tokenT)
        event_timex3_dep = self.__tb.tr.f.event_timex3_dep(tokenE, tokenT)
        signal_follow_event_ancestor_event = self.__tb.tr.f.signal_follow_event_ancestor_event(tokenE)
        signal_follow_timex3_ancestor_event = self.__tb.tr.f.signal_follow_timex3_ancestor_event(tokenE, tokenT)
        signal_precede_event_ancestor_event = self.__tb.tr.f.signal_precede_event_ancestor_event(tokenE)
        signal_precede_timex3_ancestor_event = self.__tb.tr.f.signal_precede_timex3_ancestor_event(tokenE, tokenT)
        signal_follow_event_ancestor_timex3 = self.__tb.tr.f.signal_follow_event_ancestor_timex3(tokenE, tokenT)
        signal_follow_timex3_ancestor_timex3 = self.__tb.tr.f.signal_follow_timex3_ancestor_timex3(tokenT)
        signal_precede_event_ancestor_timex3 = self.__tb.tr.f.signal_precede_event_ancestor_timex3(tokenE, tokenT)
        signal_precede_timex3_ancestor_timex3 = self.__tb.tr.f.signal_precede_timex3_ancestor_timex3(tokenT)
        signal_follow_event_text = self.__tb.tr.f.signal_follow_event_text(tokenE)
        signal_precede_event_text = self.__tb.tr.f.signal_precede_event_text(tokenE)
        signal_precede_timex3_text = self.__tb.tr.f.signal_precede_timex3_text(tokenT)
        signal_follow_timex3_text = self.__tb.tr.f.signal_follow_timex3_text(tokenT)
        signal_precede_event_pos = self.__tb.tr.f.signal_precede_event_pos(tokenE)
        signal_follow_event_pos = self.__tb.tr.f.signal_follow_event_pos(tokenE)
        signal_precede_timex3_pos = self.__tb.tr.f.signal_precede_timex3_pos(tokenT)
        signal_follow_timex3_pos = self.__tb.tr.f.signal_follow_timex3_pos(tokenT)
        signal_precede_event_distance_event = self.__tb.tr.f.signal_precede_event_distance_event(tokenE)
        signal_follow_event_distance_event = self.__tb.tr.f.signal_follow_event_distance_event(tokenE)
        signal_precede_timex3_distance_event = self.__tb.tr.f.signal_precede_timex3_distance_event(tokenE, tokenT)
        signal_follow_timex3_distance_event = self.__tb.tr.f.signal_follow_timex3_distance_event(tokenE, tokenT)
        signal_precede_event_distance_timex3 = self.__tb.tr.f.signal_precede_event_distance_timex3(tokenE, tokenT)
        signal_follow_event_distance_timex3 = self.__tb.tr.f.signal_follow_event_distance_timex3(tokenE, tokenT)
        signal_precede_timex3_distance_timex3 = self.__tb.tr.f.signal_precede_timex3_distance_timex3(tokenT)
        signal_follow_timex3_distance_timex3 = self.__tb.tr.f.signal_follow_timex3_distance_timex3(tokenT)
        signal_precede_event_comma_between_event = self.__tb.tr.f.signal_precede_event_comma_between_event(tokenE)
        signal_follow_event_comma_between_event = self.__tb.tr.f.signal_follow_event_comma_between_event(tokenE)
        signal_precede_timex3_comma_between_event = self.__tb.tr.f.signal_precede_timex3_comma_between_event(tokenE, tokenT)
        signal_follow_timex3_comma_between_event = self.__tb.tr.f.signal_follow_timex3_comma_between_event(tokenE, tokenT)
        signal_precede_event_comma_between_timex3 = self.__tb.tr.f.signal_precede_event_comma_between_timex3(tokenE, tokenT)
        signal_follow_event_comma_between_timex3 = self.__tb.tr.f.signal_follow_event_comma_between_timex3(tokenE, tokenT)
        signal_precede_timex3_comma_between_timex3 = self.__tb.tr.f.signal_precede_timex3_comma_between_timex3(tokenT)
        signal_follow_timex3_comma_between_timex3 = self.__tb.tr.f.signal_follow_timex3_comma_between_timex3(tokenT)
        signal_precede_event_child_event = self.__tb.tr.f.signal_precede_event_child_event(tokenE)
        signal_follow_event_child_event = self.__tb.tr.f.signal_follow_event_child_event(tokenE)
        signal_precede_timex3_child_event = self.__tb.tr.f.signal_precede_timex3_child_event(tokenE, tokenT)
        signal_follow_timex3_child_event = self.__tb.tr.f.signal_follow_timex3_child_event(tokenE, tokenT)
        signal_precede_event_child_timex3 = self.__tb.tr.f.signal_precede_event_child_timex3(tokenE, tokenT)
        signal_follow_event_child_timex3 = self.__tb.tr.f.signal_follow_event_child_timex3(tokenE, tokenT)
        signal_precede_timex3_child_timex3 = self.__tb.tr.f.signal_precede_timex3_child_timex3(tokenT)
        signal_follow_timex3_child_timex3 = self.__tb.tr.f.signal_follow_timex3_child_timex3(tokenT)
        signal_precede_event_is_event_head = self.__tb.tr.f.signal_precede_event_is_event_head(tokenE)
        signal_follow_event_is_event_head = self.__tb.tr.f.signal_follow_event_is_event_head(tokenE)
        signal_precede_timex3_is_event_head = self.__tb.tr.f.signal_precede_timex3_is_event_head(tokenE, tokenT)
        signal_follow_timex3_is_event_head = self.__tb.tr.f.signal_follow_timex3_is_event_head(tokenE, tokenT)
        signal_precede_event_is_timex3_head = self.__tb.tr.f.signal_precede_event_is_timex3_head(tokenE, tokenT)
        signal_follow_event_is_timex3_head = self.__tb.tr.f.signal_follow_event_is_timex3_head(tokenE, tokenT)
        signal_precede_timex3_is_timex3_head = self.__tb.tr.f.signal_precede_timex3_is_timex3_head(tokenT)
        signal_follow_timex3_is_timex3_head = self.__tb.tr.f.signal_follow_timex3_is_timex3_head(tokenT)
        signal_precede_event_dep_advmod_advcl_event = self.__tb.tr.f.signal_precede_event_dep_advmod_advcl_event(tokenE)
        signal_follow_event_dep_advmod_advcl_event = self.__tb.tr.f.signal_follow_event_dep_advmod_advcl_event(tokenE)
        signal_precede_timex3_dep_advmod_advcl_event = self.__tb.tr.f.signal_precede_timex3_dep_advmod_advcl_event(tokenE, tokenT)
        signal_follow_timex3_dep_advmod_advcl_event = self.__tb.tr.f.signal_follow_timex3_dep_advmod_advcl_event(tokenE, tokenT)
        signal_precede_event_dep_advmod_advcl_timex3 = self.__tb.tr.f.signal_precede_event_dep_advmod_advcl_timex3(tokenE, tokenT)
        signal_follow_event_dep_advmod_advcl_timex3 = self.__tb.tr.f.signal_follow_event_dep_advmod_advcl_timex3(tokenE, tokenT)
        signal_precede_timex3_dep_advmod_advcl_timex3 = self.__tb.tr.f.signal_precede_timex3_dep_advmod_advcl_timex3(tokenT)
        signal_follow_timex3_dep_advmod_advcl_timex3 = self.__tb.tr.f.signal_follow_timex3_dep_advmod_advcl_timex3(tokenT)
        signal_precede_event_head_is_event = self.__tb.tr.f.signal_precede_event_head_is_event(tokenE)
        signal_follow_event_head_is_event = self.__tb.tr.f.signal_follow_event_head_is_event(tokenE)
        signal_precede_timex3_head_is_event = self.__tb.tr.f.signal_precede_timex3_head_is_event(tokenE, tokenT)
        signal_follow_timex3_head_is_event = self.__tb.tr.f.signal_follow_timex3_head_is_event(tokenE, tokenT)
        signal_precede_event_head_is_timex3 = self.__tb.tr.f.signal_precede_event_head_is_timex3(tokenE, tokenT)
        signal_follow_event_head_is_timex3 = self.__tb.tr.f.signal_follow_event_head_is_timex3(tokenE, tokenT)
        signal_precede_timex3_head_is_timex3 = self.__tb.tr.f.signal_precede_timex3_head_is_timex3(tokenT)
        signal_follow_timex3_head_is_timex3 = self.__tb.tr.f.signal_follow_timex3_head_is_timex3(tokenT)
        signal_precede_event_dep_if_child_event = self.__tb.tr.f.signal_precede_event_dep_if_child_event(tokenE)
        signal_follow_event_dep_if_child_event = self.__tb.tr.f.signal_follow_event_dep_if_child_event(tokenE)
        signal_precede_timex3_dep_if_child_event = self.__tb.tr.f.signal_precede_timex3_dep_if_child_event(tokenE, tokenT)
        signal_follow_timex3_dep_if_child_event = self.__tb.tr.f.signal_follow_timex3_dep_if_child_event(tokenE, tokenT)
        signal_precede_event_dep_if_child_timex3 = self.__tb.tr.f.signal_precede_event_dep_if_child_timex3(tokenE, tokenT)
        signal_follow_event_dep_if_child_timex3 = self.__tb.tr.f.signal_follow_event_dep_if_child_timex3(tokenE, tokenT)
        signal_precede_timex3_dep_if_child_timex3 = self.__tb.tr.f.signal_precede_timex3_dep_if_child_timex3(tokenT)
        signal_follow_timex3_dep_if_child_timex3 = self.__tb.tr.f.signal_follow_timex3_dep_if_child_timex3(tokenT)
        reichenbach_tense = self.__tb.tr.f.reichenbach_tense(tokenE, tokenT)
        event_modal_verb = self.__tb.tr.f.event_modal_verb(tokenE)
        event_has_modal_verb_precede = self.__tb.tr.f.event_has_modal_verb_precede(tokenE)
    
        par = { 'relType': relType, 'train_test': train_test, 'anotado': anotado, 'task': 'A', 'eventID': eventID, 'e_text': e_text, 
                'relatedTo': relatedTo, 't_text': t_text, 'isentenca': id_sentenca, 'doc': nome_doc, 
                'event_class': event_class, 
                'event_closest_to_event_class': event_closest_to_event_class, 
                'event_closest_to_event_equal_class':event_closest_to_event_equal_class, 
                #'event_closest_to_event_equal_lemma':event_closest_to_event_equal_lemma,
                'event_pos': event_pos, 
                'event_closest_to_event_pos': event_closest_to_event_pos, 
                'event_closest_to_event_equal_pos': event_closest_to_event_equal_pos, 
                #'event_tense': event_tense,  
                'event_closest_to_event_tense':event_closest_to_event_tense,
                'event_closest_to_event_equal_tense': event_closest_to_event_equal_tense,
                'event_closest_to_timex3_pos': event_closest_to_timex3_pos,
                'event_closest_to_timex3_equal_pos': event_closest_to_timex3_equal_pos,
                'event_conjunction_closest_follow': event_conjunction_closest_follow,
                'event_conjunction_closest_precede': event_conjunction_closest_precede,
                'event_polarity': event_polarity,
                #'event_aspect': event_aspect,
                'event_dep': event_dep,
                'timex3_dep': timex3_dep,
                'timex3_type': timex3_type,
                'timex3_pos': timex3_pos,
                'timex3_temporalfunction': timex3_temporalfunction,
                #'timex3_root': timex3_root,
                'event_root': event_root,
                'event_pos_token_1_precede': event_pos_token_1_precede,
                'event_pos_token_1_follow': event_pos_token_1_follow,
                'event_pos_token_2_precede': event_pos_token_2_precede,
                'event_pos_token_2_follow': event_pos_token_2_follow,
                'event_pos_token_3_precede': event_pos_token_3_precede,
                'event_pos_token_3_follow': event_pos_token_3_follow,
                'timex3_pos_token_1_precede': timex3_pos_token_1_precede,
                'timex3_pos_token_1_follow': timex3_pos_token_1_follow,
                'timex3_pos_token_2_precede': timex3_pos_token_2_precede,
                'timex3_pos_token_2_follow': timex3_pos_token_2_follow,
                'timex3_pos_token_3_precede': timex3_pos_token_3_precede,
                'timex3_pos_token_3_follow': timex3_pos_token_3_follow,
                'event_preposition_precede': event_preposition_precede,
                'timex3_preposition_precede': timex3_preposition_precede,
                'event_timex3_distance': event_timex3_distance,
                'event_first_order': event_first_order,
                'event_between_order': event_between_order,
                'timex3_between_order': timex3_between_order,
                'event_timex3_no_between_order': event_timex3_no_between_order,
                #'timex3_between_quant': timex3_between_quant,
                'event_closest_to_event_temporal_direction': event_closest_to_event_temporal_direction,
                'event_temporal_direction': event_temporal_direction,
                'timex3_relevant_lemmas': timex3_relevant_lemmas,
                'event_gov_verb_aspect': event_gov_verb_aspect,
                'event_gov_verb_tense': event_gov_verb_tense,
                'timex3_gov_verb_tense': timex3_gov_verb_tense,
                'event_head_pos': event_head_pos,
                'timex3_head_pos': timex3_head_pos,
                'event_intervening_following_tense': event_intervening_following_tense,
                'event_intervening_preceding_class': event_intervening_preceding_class,
                #'event_gov_verb': event_gov_verb,
                #'timex3_gov_verb': timex3_gov_verb,
                'event_head_is_root': event_head_is_root,
                'event_is_ancestor_timex3': event_is_ancestor_timex3,
                #'event_is_child_timex3': event_is_child_timex3,
                'timex3_head_is_root': timex3_head_is_root,
                'timex3_is_ancestor_event': timex3_is_ancestor_event,
                #'timex3_is_child_event': timex3_is_child_event,
                'event_preposition_gov': event_preposition_gov,
                'timex3_preposition_gov': timex3_preposition_gov,
                'reichenbach_direct_modification': reichenbach_direct_modification,
                'reichenbach_temporal_mod_function': reichenbach_temporal_mod_function,
                'event_timex3_dep': event_timex3_dep,
                #'signal_follow_event_ancestor_event': signal_follow_event_ancestor_event,
                #'signal_follow_timex3_ancestor_event': signal_follow_timex3_ancestor_event,
                #'signal_precede_event_ancestor_event': signal_precede_event_ancestor_event,
                #'signal_precede_timex3_ancestor_event': signal_precede_timex3_ancestor_event,
                #'signal_follow_event_ancestor_timex3': signal_follow_event_ancestor_timex3,
                #'signal_follow_timex3_ancestor_timex3': signal_follow_timex3_ancestor_timex3,
                #'signal_precede_event_ancestor_timex3': signal_precede_event_ancestor_timex3,
                #'signal_precede_timex3_ancestor_timex3': signal_precede_timex3_ancestor_timex3,
                #'signal_follow_event_text': signal_follow_event_text,
                'signal_precede_event_text': signal_precede_event_text,
                'signal_precede_timex3_text': signal_precede_timex3_text,
                #'signal_follow_timex3_text': signal_follow_timex3_text,
                'signal_precede_event_pos': signal_precede_event_pos,
                #'signal_follow_event_pos': signal_follow_event_pos,
                'signal_precede_timex3_pos': signal_precede_timex3_pos,
                #'signal_follow_timex3_pos': signal_follow_timex3_pos,
                'signal_precede_event_distance_event': signal_precede_event_distance_event,
                #'signal_follow_event_distance_event': signal_follow_event_distance_event,
                #'signal_precede_timex3_distance_event': signal_precede_timex3_distance_event,
                #'signal_follow_timex3_distance_event': signal_follow_timex3_distance_event,
                #'signal_precede_event_distance_timex3': signal_precede_event_distance_timex3,
                #'signal_follow_event_distance_timex3': signal_follow_event_distance_timex3,
                'signal_precede_timex3_distance_timex3': signal_precede_timex3_distance_timex3,
                #'signal_follow_timex3_distance_timex3': signal_follow_timex3_distance_timex3,
                #'signal_precede_event_comma_between_event': signal_precede_event_comma_between_event,
                #'signal_follow_event_comma_between_event': signal_follow_event_comma_between_event,
                #'signal_precede_timex3_comma_between_event': signal_precede_timex3_comma_between_event,
                #'signal_follow_timex3_comma_between_event': signal_follow_timex3_comma_between_event,
                #'signal_precede_event_comma_between_timex3': signal_precede_event_comma_between_timex3,
                #'signal_follow_event_comma_between_timex3': signal_follow_event_comma_between_timex3,
                #'signal_precede_timex3_comma_between_timex3': signal_precede_timex3_comma_between_timex3,
                #'signal_follow_timex3_comma_between_timex3': signal_follow_timex3_comma_between_timex3,
                'signal_precede_event_child_event': signal_precede_event_child_event,
                #'signal_follow_event_child_event': signal_follow_event_child_event,
                #'signal_precede_timex3_child_event': signal_precede_timex3_child_event,
                #'signal_follow_timex3_child_event': signal_follow_timex3_child_event,
                #'signal_precede_event_child_timex3': signal_precede_event_child_timex3,
                #'signal_follow_event_child_timex3': signal_follow_event_child_timex3,
                'signal_precede_timex3_child_timex3': signal_precede_timex3_child_timex3,
                #'signal_follow_timex3_child_timex3': signal_follow_timex3_child_timex3,
                #'signal_precede_event_is_event_head': signal_precede_event_is_event_head,
                #'signal_follow_event_is_event_head': signal_follow_event_is_event_head,
                #'signal_precede_timex3_is_event_head': signal_precede_timex3_is_event_head,
                #'signal_follow_timex3_is_event_head': signal_follow_timex3_is_event_head,
                #'signal_precede_event_is_timex3_head': signal_precede_event_is_timex3_head,
                #'signal_follow_event_is_timex3_head': signal_follow_event_is_timex3_head,
                #'signal_precede_timex3_is_timex3_head': signal_precede_timex3_is_timex3_head,
                #'signal_follow_timex3_is_timex3_head': signal_follow_timex3_is_timex3_head,
                #'signal_precede_event_dep_advmod_advcl_event': signal_precede_event_dep_advmod_advcl_event,
                #'signal_follow_event_dep_advmod_advcl_event': signal_follow_event_dep_advmod_advcl_event,
                #'signal_precede_timex3_dep_advmod_advcl_event': signal_precede_timex3_dep_advmod_advcl_event,
                #'signal_follow_timex3_dep_advmod_advcl_event': signal_follow_timex3_dep_advmod_advcl_event,
                #'signal_precede_event_dep_advmod_advcl_timex3': signal_precede_event_dep_advmod_advcl_timex3,
                #'signal_follow_event_dep_advmod_advcl_timex3': signal_follow_event_dep_advmod_advcl_timex3,
                #'signal_precede_timex3_dep_advmod_advcl_timex3': signal_precede_timex3_dep_advmod_advcl_timex3,
                #'signal_follow_timex3_dep_advmod_advcl_timex3': signal_follow_timex3_dep_advmod_advcl_timex3,
                #'signal_precede_event_head_is_event': signal_precede_event_head_is_event,
                #'signal_follow_event_head_is_event': signal_follow_event_head_is_event,
                #'signal_precede_timex3_head_is_event': signal_precede_timex3_head_is_event,
                #'signal_follow_timex3_head_is_event': signal_follow_timex3_head_is_event,
                #'signal_precede_event_head_is_timex3': signal_precede_event_head_is_timex3,
                #'signal_follow_event_head_is_timex3': signal_follow_event_head_is_timex3,
                #'signal_precede_timex3_head_is_timex3': signal_precede_timex3_head_is_timex3,
                #'signal_follow_timex3_head_is_timex3': signal_follow_timex3_head_is_timex3,
                'signal_precede_event_dep_if_child_event': signal_precede_event_dep_if_child_event,
                #'signal_follow_event_dep_if_child_event': signal_follow_event_dep_if_child_event,
                #'signal_precede_timex3_dep_if_child_event': signal_precede_timex3_dep_if_child_event,
                #'signal_follow_timex3_dep_if_child_event': signal_follow_timex3_dep_if_child_event,
                #'signal_precede_event_dep_if_child_timex3': signal_precede_event_dep_if_child_timex3,
                #'signal_follow_event_dep_if_child_timex3': signal_follow_event_dep_if_child_timex3,
                'signal_precede_timex3_dep_if_child_timex3': signal_precede_timex3_dep_if_child_timex3,
                #'signal_follow_timex3_dep_if_child_timex3': signal_follow_timex3_dep_if_child_timex3,
                'reichenbach_tense': reichenbach_tense,
                'event_modal_verb': event_modal_verb,
                'event_has_modal_verb_precede': event_has_modal_verb_precede,
                }
        return par



#=========================================================================================================================
#=========================================================================================================================
# '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
#  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
#  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
#  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
#  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
#  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
#  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
# ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #
#=========================================================================================================================
#=========================================================================================================================

#---------------------------------------------------------------------
#     CLASSE SETRULES
#--------------------------------------------------------------------
class SetRulesEmpty:
    '''
    Classe para informar sobre a necessidade de inicializar setRules com o método TemporalRelation.setRules_start().
    '''
    def __getattr__(self, name):
        if not name.startswith("_"):
            print(f"Para utilizar os métodos de setRules que manipulam conjuntos de regras, \ninicialize-o utilizando o método TemporalRelation.setRules_start(). Consulte documentação.")

class ErrorRulesFiltered(Exception):
    def __init__(self, mensagem):
        self.mensagem = mensagem

    def __str__(self):
        return repr(self.mensagem)

class SetRules():
    '''
    Implementa estrutura de dados de Conjunto de Regras.
    Classe independente que processa apenas regras originadas do dataset de features (TimebankPT.FeaturesToDataset).
    Para que as regras da classe sejam processadas, é necessários atribuí-las a TemporalRelation.rules.
    Por exemplo: TemporalRelation.rules = temporalRelation.SetRules.rules
    
    args:
        class_features: String que representa a classe onde estão as funções que implementam as features.
                As funções estão na classe TemporalRelation.f, geralmente instanciada como tr.f ou self.f
            
        params_functions: Dicionário de funções e seus parâmetros. Deve ser passado ao instanciar a classe.
                Ex: {'event_class': ['E'], 'event_timex3_dep': ['E', 'T']}
    '''
    params_functions = {}
    _class_features = 'self.f'
    
    @property
    def class_features(self):
        return self.__class__._class_features
    
    @class_features.setter
    def class_features(self, valor):
        self.__class__._class_features = valor
        
        
    def __init__(self, params_functions: dict = None):
        self.__rule = None
        self.__rules = []
        self.__rules_all = []
        self.__is_filter = False
        self.__is_sorted = False
        
        self.fields = ['cod_regra', 'relType', 'ordem', 'origem', 'acuracia', 'acertos', 'acionamentos']

        if params_functions:
            SetRules.params_functions = params_functions
    
    def __str__(self):
        return '[' + ',\n'.join(str(r) for r in self.rules) + ']'
        
    def __repr__(self):
        return self.__str__()

    def copy(self):
        return copy.copy(self)
    
    
    #======================================================================
    #----CLASSE RULE  ----------------------------------------------
    
    class Rule():
        '''
        Estrutura de dados de uma Regra.
        '''
        def __init__(self, cod_regra: float, relType: str, ordem: float, origem: str, acuracia: float, acertos: int, acionamentos: int):
            self.cod_regra = cod_regra
            self.relType = relType
            self.ordem = ordem
            self.predicate = None
            self.origem = origem
            self.acuracia = acuracia
            self.acertos = acertos
            self.acionamentos = acionamentos
            self.predicates = []

            self.fields_rule = ['cod_regra', 'relType', 'ordem', 'predicates', 'origem', 'acuracia', 'acertos', 'acionamentos']
            self.attrib = ''

        def __str__(self):
            return f"[{str(self.cod_regra)}, '{self.relType}', {str(self.ordem)}, \"{' and '.join(str(p) for p in self.predicates)}\", '{self.origem}', {self.acuracia}, {self.acertos}, {self.acionamentos}]"

        def __repr__(self):
            return self.__str__()

        def __iter__(self):
            self.__current = 0
            return self

        def __next__(self):
            if self.__current >= len(self.fields_rule):
                raise StopIteration
            self.attrib = self.fields_rule[self.__current]
            self.__current += 1
            return eval('self.' + self.attrib)


        #======================================================================
        #----CLASSE PREDICATE  ----------------------------------------------

        class Predicate():
            '''
            Estrutura de dados de um predicado (condições) da regras.
            '''
            __TOKENE = 'tokenE'
            __TOKENT = 'tokenT'

            def __init__(self, feature: str, operator: str, value: str):

                self.feature = feature
                self.operator = operator
                self.value = value

                self.fields_predicate = ['feature', 'operator', 'value']
                self.attrib = ''
                
            def __str__(self):
                return f"{self.feature} {self.operator} {self.value}"

            def __repr__(self):
                return self.__str__()

            def __iter__(self):
                self.__current = 0
                return self

            def __next__(self):
                if self.__current >= len(self.fields_predicate):
                    raise StopIteration
                self.attrib = self.fields_predicate[self.__current]
                self.__current += 1
                return eval('self.' + self.attrib)

            def __lt__(self, other):
                '''
                Ordena predicado por features e value
                '''
                return (self.feature + str(self.value)) < (other.feature + str(other.value))


            #======================================================================
            #----FUNÇÕES DE PREDICATE -------------------------------------------------

            @property
            def is_empty(self):
                if (self.feature == '') or (self.value == ''):
                    return True
                return False
        
            def __params_function(self, function: str) -> list:
                ''' 
                Retorna lista com os parâmetros da função que implementa uma feature.

                Args:
                    function: nome da função que implementa a feature correspondente.
                '''
                params_functions = SetRules.params_functions
                if not params_functions:
                    raise ValueError('Não foi carregado as funções e seus parâmetros (params_functions).')
                    
                try:
                    list_param = params_functions[function]
                except KeyError as e:
                    print('Valor inválido: ', e)
                    raise
                else:
                    return list_param


            @property
            def feature(self):
                return self._feature

            @feature.setter
            def feature(self, feature: str):
                self._feature = ''
                if not feature:
                    return
                if feature in ['TRUE', 'True', 'true']:
                    self._feature = 'True'
                    return
                feature = feature.lower()
                self._feature = self.__trata_feature(feature)

                
            @property
            def operator(self: str):
                return self._operator

            @operator.setter
            def operator(self, operator: str):
                self._operator = ''
                if not operator:
                    return
                if operator == '=':
                    operator = '=='
                if operator not in  ['==', '!=', 'in', 'not in']:
                    print(f'Operador inválido: {operator}')
                    self._operator = '=='
                    return
                self._operator = operator


            @property
            def value(self: str):
                return self._value

            @value.setter
            def value(self, value):
                self._value = ''
                if not value:
                    return
                if value not in ['True', 'False']:
                    if not value.startswith("'"):
                        value = "'" + value + "'"
                if value.startswith('[') or value.startswith("'["):
                    value = value.replace('[', '').replace(']', '')
                    value = value.replace("'", '')
                    value = value.split(sep=',')
                    value = str(list(map(str.strip, value)))
                self._value = value


            def __trata_param(self, param: str):
                if param == 'E':
                    return self.__TOKENE
                if param == 'T':
                    return self.__TOKENT

            def __trata_feature(self, feature) -> str:
                if not feature:
                    return
                list_params = self.__params_function(feature)
                paramentros = '('
                for param in list_params:
                    param = self.__trata_param(param)
                    paramentros += param + ', '
                paramentros = paramentros.strip()[:-1] + ')'

                return SetRules._class_features + '.' + feature + paramentros

        #----FIM PREDICATE --------------------------------------------------
        #======================================================================


        #======================================================================
        #----FUNÇÕES DE RULE -------------------------------------------------
        
        def add_predicate(self, feature: str = '', operator: str = '', value: str = ''):
            self.predicate = self.Predicate(feature, operator, value)
            self.predicates.append(self.predicate)
            self.predicates.sort()
        
            
        @property
        def is_empty(self):
            if (self.relType == '') or (self.cod_regra == None):
                return True
            return False

        @property
        def cod_regra(self):
            return self._cod_regra

        @cod_regra.setter
        def cod_regra(self, cod_regra: float):
            self._cod_regra = cod_regra


        @property
        def relType(self):
            return self._relType

        @relType.setter
        def relType(self, relType: str):
            relTypes = ['BEFORE', 'AFTER', 'OVERLAP', 'VAGUE', 'BEFORE-OR-OVERLAP', 'OVERLAP-OR-AFTER']
            
            self._relType = ''
            if not relType:
                return
            if relType not in relTypes:
                raise ValueError(f'relType Inválido: {relType}')
            self._relType = relType


        @property
        def ordem(self):
            return self._ordem

        @ordem.setter
        def ordem(self, ordem: float):
            self._ordem = ordem


        @property
        def predicate(self):
            return self._predicate

        @predicate.setter
        def predicate(self, predicate: Predicate):
            self._predicate = predicate


        @property
        def predicates(self):
            return self._predicates

        @predicates.setter
        def predicates(self, predicates: list):
            self._predicates = predicates


        @property
        def origem(self):
            return self._origem

        @origem.setter
        def origem(self, origem: str):
            self._origem = origem


        @property
        def acuracia(self):
            return self._acuracia

        @acuracia.setter
        def acuracia(self, acuracia: float):
            if acuracia:
                if 1 < acuracia <= 100:
                    acuracia /= 100
                if not (0 <= acuracia <= 1):
                    raise ValueError(f'Acurácia inválida: {acuracia}')
                
            self._acuracia = acuracia

        @property
        def acertos(self):
            return self._acertos

        @acertos.setter
        def acertos(self, acertos: int):
            if acertos:
                if not acertos >= 0:
                    raise ValueError(f'Quantidade de acertos inválido: {acertos}')
                
            self._acertos = acertos

        @property
        def acionamentos(self):
            return self._acionamentos

        @acionamentos.setter
        def acionamentos(self, acionamentos: int):
            if acionamentos:
                if not acionamentos >= 0:
                    raise ValueError(f'Quantidade de acionamentos inválido: {acionamentos}')
                
            self._acionamentos = acionamentos
    
    #----FIM RULE --------------------------------------------------
    #======================================================================
    
    
    
    #======================================================================
    #----FUNÇÕES DE SETRULES ----------------------------------------------
    
    def clear(self):
        '''
        Apaga todas as regras do conjunto de regras para iniciar um novo conjunto de regras.
        '''
        self.rules = []
        self.__rules_all = []
        self.__is_filter = False
        self.__is_sorted = False
        
    
    def add_rule(self, relType: str, cod_regra: float = None, ordem: int = None, origem: str = '', acuracia: float = 0, acertos: int = 0, acionamentos: int = 0):
        '''
        Adiciona novas regras ao conjunto de regras
        '''
        if self.is_filter:
            raise ErrorRulesFiltered('Não pode adicionar novas regras estando o conjunto de regras filtrado. Desative o filtro com o método filter_clear()')
            
        if not cod_regra:
            cod_regra = self.get_max_cod_rule() + 1
        if not ordem:
            ordem = self.get_max_order_rule(origem) + 1

        if not self.get_rule(cod_regra).is_empty:
            raise ValueError(f'Código de regra {cod_regra} já existe.')
            
        self.rule = self.Rule(cod_regra, relType, ordem, origem, acuracia, acertos, acionamentos)
        self.rules.append(self.rule)
        
    
    def get_rule(self, cod_regra: list) -> list:
        '''
        Retorna a regra ou lista de regras conforme 'cod_regra' especificado.
        
        Args:
            cod_regra: Pode ser numérico ou lista de números.
                Se for informado um número retorna uma regra do tipo Rule, ou Rule vazio se 'cod_regra' não existir
                Se for informado uma lista de números, retorna uma lista de regras.
        '''
        if type(cod_regra) != list:
            cod_regra = [cod_regra]
            
        result = list(filter(lambda rule: rule.cod_regra in cod_regra, self.rules))
        if len(cod_regra) == 1:
            #Rule(cod_regra: float, relType: str, ordem: float, origem: str, acuracia: float, acertos: int, acionamentos: int)
            return result[0] if result else self.Rule(None, '', None, '', None, None, None,)
        else:
            return result if result else []
    
    
    def remove_rule(self, field: str, value, verbosy=True) -> bool:
        '''
        Remove regras do conjunto de regras conforme campo e valor especificado.

        Args:
            field: Especifica o campo que deseja filtrar o valor.
                Campos válidos: ['cod_regra', 'relType', 'ordem', 'origem', 'acuracia', 'acertos', 'acionamentos']
            valor: valor do campo 'field' que deseja remover.
                Pode ser lista ou valor único. 
                Se field for acuracia, acertos, acionamentos, remove os valores menores que 'value'.
                Se field for ordem, remove 'ordem' maiores que 'value'.
        '''
        if self.is_filter:
            raise ErrorRulesFiltered('Não pode remover regras se o conjunto de regras estiver filtrado. Desative o filtro com o método filter_clear()')
            
        cod_excluidos = []
        if field not in self.fields:
            print(f"Field '{field}' inválido. \nValor válidos: {str(self.fields)}.")
            return

        rules_all = self.rules.copy()
        self.filter_rules(field, value, verbosy)
        cod_filtrado = [rule.cod_regra for rule in self.rules]
        
        try:
            sinal = '='
            if field == 'ordem' or field == 'acuracia' or field == 'acertos' or field == 'acionamentos':
                if field == 'acuracia':
                    sinal = '<='
                    if (type(value) not in [int, float] or not (0 < value < 1)):
                        raise ValueError(f'{value} é acurácia inválida. Valores válidos entre 0 e 1.')
                
                if field == 'acertos':
                    sinal = '<='
                    if (type(value) not in [int] or not (value >= 0)):
                        raise ValueError(f'{value} é quantidade de acertos inválido. Valores válidos entre maior ou igual a 0.')
                    
                if field == 'acionamentos':
                    sinal = '<='
                    if (type(value) not in [int] or not (value >= 0)):
                        raise ValueError(f'{value} é quantidade de acionamentos inválido. Valores válidos entre maior ou igual a 0.')
                
                if field == 'ordem': 
                    sinal = '>='
                    if (type(value) not in [int, float] or not (value >= 1)):
                        raise ValueError(f"{value} é inválido. Order deve ser numérico maior ou igual a 1.")
                
                rules_que_fica = [rule for rule in rules_all if rule.cod_regra in cod_filtrado]
                cod_excluidos = list(set([rule.cod_regra for rule in rules_all]) - set(cod_filtrado))
            else:
                rules_que_fica = [rule for rule in rules_all if rule.cod_regra not in cod_filtrado]
                cod_excluidos = cod_filtrado
        except Exception as e:
            self.filter_clear()
            print(f'Erro ao tentar remover regras. ERRO: {e}')
        else:
            self.rules[:] = rules_que_fica
            self.__is_filter = False
            
        if len(rules_que_fica) != len(rules_all):
            if verbosy:
                print(f'{len(cod_excluidos)} regras removidas: {cod_excluidos}.')
            return True
        else:
            if verbosy:
                print(f'Nenhuma regra foi removida. Não foi encontrada regra que atenda ao critério: {field} {sinal} {value}.')
            return False
    
    
    def get_max_cod_rule(self) -> int:
        if len(self.rules) == 0:
            return 0
        return max(self.rules, key=lambda x: x.cod_regra).cod_regra
    
    def get_max_order_rule(self, origem: str) -> int:
        ''' Retorna dicionário com a ordem máxima de cada origem '''
        if len(self.rules) == 0:
            return 0
        #Agrupando as regras por origem e obtendo o máximo da ordem para cada grupo
        origem_max = {origem: max((r.ordem for r in rules), default=0) for origem, rules in groupby(sorted(self.rules, key=lambda x: x.origem), key=lambda x: x.origem)}
        if origem in origem_max.keys():
            return origem_max[origem]
        else:
            return 0
    
        #Se for ordem única, independente da origem
        #return max(self.rules, key=lambda x: x.ordem).ordem
    
    
    def __list_upper(self, lista: list) -> list:
        ''' Converte as strings da lista em maiúsculas  '''
        return [valor.upper() if isinstance(valor, str) else valor for valor in lista]
    
    def filter_rules(self, field: str, value: list, verbosy=True) -> List[list]:
        '''
        Filtra regras conforme o campo e valor especificado
        
        Args:
            field = campo da regra que deseja filtrar. 
                    Valores válidos: ['cod_regra', 'relType', 'ordem', 'origem', 'acuracia', 'acertos', 'acionamentos']
            value = pode ser lista ou valor único. 
                    Se field for acuracia, acertos ou acionamentos, exibe as regras cuja valor é maior que 'value'.
                    Se field for ordem, exibe as regras cuja ordem é menor que 'value'.
        '''
        if not field in self.fields:
            raise ValueError(f"Campo do filtro inválido.\n Válidos: {self.fields}")
        
        if field == 'acuracia':
            if type(value) not in [int, float] or not(0 < value < 1):
                raise ValueError(f"{value} é inválido. A acurácia deve ser numérico e entre 0 e 1.")
        
        if field == 'acertos':
            if type(value) not in [int] or not(value >= 0):
                raise ValueError(f"{value} é inválido. A quantidade de acertos deve ser inteiro maior ou igual a 0.")
            
        if field == 'acionamentos':
            if type(value) not in [int] or not(value >= 0):
                raise ValueError(f"{value} é inválido. A quantidade de acionamentos deve ser inteiro maior ou igual a 0.")

        if field == 'ordem':
            if type(value) not in [int, float] or not (value >= 1):
                raise ValueError(f"{value} é inválido. Order deve ser numérico maior ou igual a 1.")
                
        rules_filtered = []    
        
        #Salva todas as regras antes de filtrar
        if not self.is_filter:
            self.__rules_all = self.rules.copy()
        
        try:
            sinal = '='
            if field == 'acuracia' or field == 'acertos' or field == 'acionamentos':
                #rules_filtered = [lista for lista in self.rules if lista.acuracia >= value]
                rules_filtered = eval('[lista for lista in self.rules if lista.' + field + ' >= ' + str(value) + ']')
                sinal = '>='
            elif field == 'ordem':
                rules_filtered = [lista for lista in self.rules if lista.ordem <= value]
                sinal = '<='
            else:
                if type(value) != list:
                    value = [value]
                value = self.__list_upper(value)
                upper = '' if field == 'cod_regra' else '.upper()'
                rules_filtered = eval('[lista for lista in self.rules if lista.' + field + upper + ' in ' + str(value) + ']')
        except Exception as e:
            self.filter_clear()
            print(f'Erro ao filtrar. ERROR: {e}')
        else:
            self.rules = rules_filtered
            self.__is_filter = True
            if verbosy:
                print(f'Regras filtradas conforme critério: {field} {sinal} {value}. Total de regras: {len(self.rules)}')

    
    def filter_clear(self):
        ''' Limpa todos os filtros aplicados à regras'''
        if self.__is_sorted:
            self.__is_sorted = False

        self.rules = self.__rules_all
        self.__is_filter = False
        print(f"Filtro de regras foi desativado. Total regras ativas: {len(self.rules)}")
    
    
    def sort_rules(self, fields: list, reverse = False):
        '''
        Ordena as regras conforme os campos especificados. 
        
        Args:
            field = pode ser um campo ou lista de campos da regra que deseja filtrar.
                    valores válidos: 'cod_regra', 'relType', 'ordem', 'origem', 'acuracia', 'acertos', 'acionamentos'
        '''
        if not isinstance(fields, list):
            fields = [fields]
        fields = list(reversed(fields))

        if not self.rules:
            raise ValueError(f"Ainda não foi atribuido um conjunto de regras para SetRules.rules.")

        if not all(field in self.rule.fields_rule for field in fields):
            raise ValueError(f"Campo(s) de ordenação da regra inválido.\n Válidos: {self.rule.fields_rule}")
        
        if isinstance(reverse, bool):
            reverse = [reverse] * len(fields)
        elif len(reverse) != len(fields):
            raise ValueError("Os parâmetros fields e reverse devem ter o mesmo tamanho.")
        reverse = list(reversed(reverse))
        
        #Salva todas as regras antes de ordenar
        if not self.__is_sorted:
            self.__rules_all = self.rules.copy()
            self.__is_sorted = True

        #Ordena por mais de um campo e #PERMITE QUE REVERSE SEJA LISTA DE BOOL, CASANDO COM FIELDS
        for i, field in enumerate(fields):
            reverse_field = reverse[i]
            self.rules.sort(key=lambda rule: [list(rule)[self.rule.fields_rule.index(field)]], reverse=reverse_field)

        fields_str = [f'"{field}" {"em ordem decrescente" if rev else "em ordem crescente"}' for field, rev in zip(fields, reverse)]
        print(f"Regras ordenadas por: [{', '.join(fields_str)}]")
        #return self.rules

        #Ordena por mais de um campo  #NÃO PERMITE QUE REVERSE SEJA LISTA DE BOOL
        #self.rules.sort(key=lambda rule: [list(rule)[self.rule.fields_rule.index(field)] for field in fields], reverse=reverse)

        #Não consegui fazer funcionar assim. Em 'rule.field', field não é interpretado com variável mas como atributo de rule, eval não funcionou
        #self.rules.sort(key=lambda rule: [rule.field for field in str(fields)], reverse = reverse)
    
        
    def sort_clear(self):
        ''' Retorna à ordem inicial. Limpa ordenação'''
        if self.is_filter:
            self.__is_filter = False

        self.rules = self.__rules_all
        self.__is_sorted = False
        print('Regras em ordem original.')
        #return self.rules
    

    def remove_rules_duplicate(self): #FIXME Avisar se há regras duplicada com tipo relação diferentes
        '''
        Remove regras duplicadas. A primeira da lista de duplicados permanece.
        Considera apenas a lista de predicados para comparar a regra.
        Não leva em conta a ordem dos predicados.

        Return:
            lista de regras duplicadas excluídas
        '''
        duplicatas = []
        for i, rule in enumerate(self.rules):
            predicates = set([str(predicate) for predicate in rule.predicates]) 
            list_predicates = [set([str(predicate) for predicate in r.predicates]) for r in self.rules[:i]]
            if predicates in list_predicates:
                duplicatas.append(rule)
        for duplicata in duplicatas:
            self.rules.remove(duplicata)
        print(len(duplicatas), 'regras duplicadas excluídas.')
        return (duplicatas)
    


    def check_filename(self, filename: str, check_if_exist: bool = False) -> str:
        '''
        Retorna nome do arquivo com a extensão padrão .pickle se não for informada e pode verificar se o arquivo existe.
        '''
        if not filename:
            raise ValueError(f"É necessário informar o nome do arquivo.")
        filename, ext = os.path.splitext(filename.strip())
        
        if filename.count('.') >= 2:
            print(f'O arquivo {filename} possui mais de um ponto. Certifique-se que haja uma extensão.')

        if not ext:
            ext = '.pickle'
        filename =  filename.strip() + ext
        
        if check_if_exist:
            if not os.path.exists(filename):
                raise ValueError(f"O arquivo '{filename}' não existe.")
                
        return filename
    
    def save_object(self, name_object, nome_arquivo: str):
        '''
        Salva name_object em arquivo físico .pickle.

        Args:
            name_object: objeto serializável.
            nome_arquivo: arquivo .pickle
        '''
        if not name_object:
            raise ValueError(f"É necessário informar o nome do objeto.")

        nome_arquivo = self.check_filename(nome_arquivo)
        try:
            with open(nome_arquivo, 'wb') as arquivo:
                pickle.dump(name_object, arquivo)
        except Exception as e:
            print(f'Erro ao salvar conjunto de regras no arquivo {nome_arquivo}. ERRO: {e}')
        else:
            print(f'Conjunto de regras salvo com sucesso no arquivo {nome_arquivo}.')
            
    def save_rules(self, nome_arquivo: str):
        '''
        Salva objeto SetRules.rules, que representa o conjunto de regras atual, em arquivo físico .pickle.
        '''
        if not self.rules or len(self.rules) == 0:
            print('Conjunto de regra vazio. Regras não foram salvas.')
            return
        self.save_object(self.rules, nome_arquivo)

    def __load_rules(self, nome_arquivo: str) -> 'SetRules':
        '''
        Retorna objeto SetRules.rules de arquivo físico salvo pelo método 'SetRules.save_rules(nome_arquivo)'.

        '''
        nome_arquivo = self.check_filename(nome_arquivo, check_if_exist=True)
        with open(nome_arquivo, 'rb') as arquivo:
            return pickle.load(arquivo)

    def save_rules_to_txt(self, nome_arquivo: str):
        '''
        Salva conjunto de regras em formato de texto em arquivo.
        Para carregar, use o método add_setRules(nome_arquivo, 'my_rules'). 
            O parametro 'algoritmo' deve ser 'my_rules'.
        '''
        #atribui extensão .txt se nome_arquivo não tiver extensão
        filename, ext = os.path.splitext(nome_arquivo.strip())
        if not ext:
            ext = '.txt'
            nome_arquivo =  filename.strip() + ext

        nome_arquivo = self.check_filename(nome_arquivo)
        try:
            with open(nome_arquivo, 'w') as arquivo:
                arquivo.writelines([str(lista) + '\n' for lista in self.rules])
        except Exception as e:
            print(f'Erro ao salvar conjunto de regras em formato texto no arquivo {nome_arquivo}. ERRO: {e}')
        else:
            print(f'Conjunto de regras salvo em formato texto no arquivo {nome_arquivo}.')

        
    def add_setRules(self, nome_arquivo: str, algoritmo: str = '', reset_cod_regra: bool = False, reset_order: bool = False):
        '''
        Adiciona conjunto de regras à instancia atual.
        
        Args:
            nome_arquivo:   Nome do arquivo de origem das regras.
                            Pode ser arquivos .pickle salvos pelo método 'save_rules(nome_arquivo)'
                            Pode ser arquivos .txt contendo regras geradas por um dos algoritmos válido.
                            Pode ser arquivos .txt salvo pelo método 'save_rules_to_txt(nome_arquivo)'
                                neste caso, o algoritmo deve ser 'my_rules'.

            algoritmo:  Obrigatório se for arquivo .txt ou qualquer outro arquivo texto contendo regras.
                        Serve para identificar a origem da regras para utilizar o parser apropriado.
                        Valores válidos 'jrip_weka', 'cn2_orange', 'CN2' e 'my_rules'. Sendo 'cn2_orange' = 'CN2'.
                        Não é usado quando o arquivo é .pickle

            reset_cod_regra: Se algoritmo for 'my_rules', então reset_cod_regra determina se o código da regra que está sendo 
                        adicionada será levado em consideração. Se True, o código da regra do arquivo será desconsiderado.
            
            reset_order: Se algoritmo for 'my_rules', então reset_order determina se a ordem da regra será levado em consideração. 
                        Se True, a ordem da regra do arquivo será desconsiderado.
        '''
        nome_arquivo = self.check_filename(nome_arquivo, check_if_exist = True)
        _, ext = os.path.splitext(nome_arquivo)
        
        if ext == '.pickle':
            self.__add_setRules_pickle(nome_arquivo)
        else:
            if not algoritmo:
                print(f'Para arquivo de texto contendo regras é obrigatório informa o algoritmo gerador das regras.')
                return
            self.__add_setRules_file_txt(nome_arquivo, algoritmo, reset_cod_regra=reset_cod_regra, reset_order=reset_order)
    

    def __trata_feature(self, func: str):
        '''
        Retira os parêntesis, parâmetros e referência à classe se houver.
        Permanece apenas o nome da função que implementa a feature.
        '''
        prefix_class = SetRules._class_features

        if func.startswith(prefix_class):
            func = func.removeprefix(prefix_class + '.')

        pos_abre = func.find('(')
        if pos_abre > 0: # encontrou abre parêntesis
            func = func[:pos_abre]

        return func.strip()
    
    def __add_setRules(self, setRules: 'SetRules.rules'):
        '''
        Adiciona conjunto de regras do tipo SetRules.rules à instancia atual'
        '''
        if type(setRules) != list:
            raise ValueError("ERROR: setRules deve ser uma lista de Rule: 'SetRules.rules'")
        
        try:
            for rule in setRules:
                #serve para manter compatibilidade com setRules que não tem 'acertos' e 'acionamentos'
                acertos = 0
                acionamentos = 0
                if hasattr(rule, 'acertos'):  #verifica se acertos é um atributo de rule
                    acertos = rule.acertos
                if hasattr(rule, 'acionamentos'):
                    acionamentos = rule.acionamentos

                #print(rule.relType, rule.ordem, rule.origem, rule.acuracia, rule.acertos, rule.acionamentos)
                self.add_rule(relType=rule.relType, ordem=rule.ordem, origem=rule.origem, acuracia=rule.acuracia, acertos=acertos, acionamentos=acionamentos)
                for predicate in rule.predicates:
                    feature = self.__trata_feature(predicate.feature)
                    #print('\t', 'Trat:', feature, predicate.operator, predicate.value)
                    self.rule.add_predicate(feature=feature, operator=predicate.operator, value=predicate.value)

        except Exception as e:
            print(f"Erro ao adicionar regras ao conjunto de regras atual. ERRO em '__add_setRules': {e}")


    def __add_setRules_pickle(self, nome_arquivo: str):
        '''
        Adiciona conjunto de regras à instancia atual a partir de arquivos .pickle salvos pelo método 'save_rules(nome_arquivo)'
        '''
        nome_arquivo = self.check_filename(nome_arquivo, check_if_exist=True)
        setRules_salvo = self.__load_rules(nome_arquivo)

        try:
            self.__add_setRules(setRules_salvo)

        except ErrorRulesFiltered as e:
            print(f'ERROR: {e}')
        except Exception as e:
            print(f"Erro ao adicionar regras ao conjunto de regras atual. ERRO em '__add_setRules_pickle': {e}")
        else:
            print(f'As regras do arquivo {nome_arquivo} foram adicionadas com sucesso.')
            print(f'Total de regras: {len(self.rules)}\n')

            
    def __add_setRules_file_txt(self, nome_arquivo: str, algoritmo: str, reset_cod_regra: bool = False, reset_order: bool = False):
        '''
        Adiciona conjunto de regras à instancia atual a partir de arquivos txt contendo regras geradas pelo 'algoritmo'.
        '''
        nome_arquivo = self.check_filename(nome_arquivo, check_if_exist = True)
        
        algoritmos = ['my_rules', 'jrip_weka', 'cn2_orange', 'CN2']

        algoritmo = algoritmo.lower()
        if algoritmo not in algoritmos:
            raise ValueError(f'{algoritmo} é um algoritmo inválido. Válido: {str(algoritmos)}.')

        if algoritmo == 'my_rules':
            self.__add_setRules_myrules_txt(nome_arquivo, reset_cod_regra=reset_cod_regra, reset_order=reset_order)
            return
        if algoritmo == 'jrip_weka':
            reg_sep = '\d+\.\d+\/\d+\.\d+|relType=|[()\s]+'
        if algoritmo in ['cn2_orange', 'CN2']:
            reg_sep = 'IF|relType=|[()\s]+'
            
        features = list(self.params_functions.keys())
        operators = ['=', '==', '!=', 'in', 'not in']
        relTypes = ['BEFORE', 'AFTER', 'OVERLAP', 'BEFORE-OR-OVERLAP', 'OVERLAP-OR-AFTER', 'VAGUE']
        sep_predicates = ['and', 'AND']
        inutil = ['=>', 'THEN', 'then']
        data_types = {'feature': features, 'operator': operators, 'relType': relTypes, 'sep_predicates': sep_predicates, 'inutil': inutil}
        types = list(data_types.keys()) 

        def tipo_palavra(palavra):
            for tipo in types: 
                if palavra in data_types[tipo]:
                    return tipo
            return 'value'

        def relType_rule(palavras):
            for palavra in palavras:
                tipo = next((key for key, values in data_types.items() if palavra in values), None)
                if tipo == 'relType':
                    return palavra

        def line_to_words(line) -> list:
            line = line.replace('==', ' == ').replace('!=', ' != ')
            words = re.split(reg_sep, line)
            words = list(filter(bool, words)) #apaga itens vazios
            return words

        try:
            with open(nome_arquivo, 'r', encoding='utf-8') as f:
                for linha in f:
                    palavras = line_to_words(linha)
                    relType = relType_rule(palavras)
                    #print(f'\n==========={relType}==============\n', palavras, '\n\tL: ', linha)
                    self.add_rule(relType=relType, origem=algoritmo)
                    if len(palavras) <= 4: # geralmente a ultima regra atribuindo class padrão
                        self.rule.add_predicate('True', '=', 'True')
                        break
                    for palavra in palavras:
                        #print(palavra, ': ', tipo_palavra(palavra)) 
                        if tipo_palavra(palavra) == 'feature':
                            self.rule.add_predicate()
                            self.rule.predicate.feature = palavra
                        if tipo_palavra(palavra) == 'operator':
                            self.rule.predicate.operator = palavra
                        if tipo_palavra(palavra) == 'value':
                            if palavra:
                                self.rule.predicate.value = palavra
                
        except ErrorRulesFiltered as e:
            print(f'ERROR: {e}')
        except Exception as e:
            print(f"Erro ao adicionar regras ao conjunto de regras atual. ERRO em '__add_setRules_file_txt': {e}")
        else:
            print(f'As regras do arquivo {nome_arquivo} foram adicionadas com sucesso.')
            print(f'Total de regras: {len(self.rules)}\n')


    def __parse_predicates(self, predicates_str: str) -> list:
        '''
        Converte a string que representa os predicados em lista de predicados 
        '''
        # Dividindo a string em subexpressões separadas pelo operador "and"
        predicates = predicates_str.split(sep=' and ')
        list_predicates = []
        # Para cada predicates, dividir em triplas separadas pelos operadores "==", "!=", ou "in"
        for predicate_str in predicates:
            predicate = re.split(r'\s+(==|!=|in|not in)\s+', predicate_str.strip())
            list_predicates.append(predicate)
        return list_predicates

    def __add_myrule_str(self, myrule: Union[str, list], reset_cod_regra: bool = False, reset_order: bool = False):
        '''
        Adiciona uma regra 'my_rule' em formato texto à instancia atual.
        
        Args:
            myrule: uma regra 'my_rules' em formato texto: [cod_regra, 'relType', ordem, "predicados", 'origem', acuracia, acertos, acionamentos]
            reset_cod_regra = desconsidera cod_regra da origem
            reset_order = desconsidera ordem da origem
        '''
        icod_regra = 0 
        irelType = 1
        iordem = 2
        ipredicates = 3
        iorigem = 4
        iacuracia = 5
        iacertos = 6
        iacionamentos = 7
        
        debug = False

        try:
            if type(myrule) == list:
                list_myrule = myrule

            if type(myrule) == str:
                myrule = myrule.replace('[', '').replace(']', '')
                regex = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)') #seleciona apenas as virgulas que não estão entre aspas duplas
                #regex = re.compile(r'"(?:\\"|[^"])*"|\S+')  #seleciona cada elemento que fará parte da lista. Tudo que estiver entre aspas duplas é considerado um elemento mesmo contendo vírgula (separador)
                #lista_linha = regex.findall(linha)
                
                list_myrule = regex.split(myrule)
                list_myrule = list(map(str.strip, list_myrule))

            size_list_myrule = len(list_myrule)

            if debug: print('list_myrule: ', list_myrule)

            if reset_cod_regra:
                cod_regra = None 
            else:
                cod_regra = float(list_myrule[icod_regra]) 
            relType = list_myrule[irelType].replace("'", "")
            if reset_order:
                ordem = None
            else:
                ordem = float(list_myrule[iordem])
            predicates = list_myrule[ipredicates].replace('"', '')
            origem = list_myrule[iorigem].replace("'", "")
            acuracia = float(list_myrule[iacuracia])
            
            if iacertos > size_list_myrule-1: #para manter compatibilidade com setRules salvos sem 'acertos' e 'acionamentos'
                acertos = 0
            else:
                acertos = int(list_myrule[iacertos])
            if iacionamentos > size_list_myrule-1:
                acionamentos = 0
            else:
                acionamentos = int(list_myrule[iacionamentos])

            if debug: print(cod_regra, relType, ordem, predicates, origem, acuracia, acertos, acionamentos)

            self.add_rule(cod_regra=cod_regra, relType=relType, ordem=ordem, origem=origem, acuracia=acuracia, acertos=acertos, acionamentos=acionamentos)
            
            list_predicates = self.__parse_predicates(predicates) #Converte a string que representa os predicados em lista de predicados
            for predicate in list_predicates:
                atributo = self.__trata_feature(predicate[0])
                operador = predicate[1]
                valor = predicate[2]
                if debug: print('\tP:', 'A:', atributo, 'O:', operador, 'V:', valor)

                self.rule.add_predicate(atributo, operador, valor)
        except Exception as e:
            print(f"Erro ao adicionar uma regra 'my_rules' ao conjunto de regras atual. ERRO: {e} \n{myrule}")


    def __add_setRules_myrules_txt(self, nome_arquivo: str, reset_cod_regra: bool = False, reset_order: bool = False):
        '''
        Adiciona conjunto de regras à instancia atual a partir de arquivo txt contendo regras geradas por essa classe.
        Denominamos de 'my_rules' onde precisar ser identificado o algoritmo gerador das regras.
        '''
        nome_arquivo = self.check_filename(nome_arquivo, check_if_exist = True)
        try:
            with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
                for rule_str in arquivo:
                    self.__add_myrule_str(rule_str, reset_cod_regra=reset_cod_regra, reset_order=reset_order)
                        
        except ErrorRulesFiltered as e:
            print(f'ERROR: {e}')
        except Exception as e:
            print(f"Erro ao adicionar regras 'my_rules' ao conjunto de regras atual. ERRO: {e} \n{rule_str}")
        else:
            print(f'As regras do arquivo {nome_arquivo} foram adicionadas com sucesso.')
            print(f'Total de regras: {len(self.rules)}\n')


    def __add_setRules_myrules_obj(self, setRules_obj: List[list], reset_cod_regra: bool = False, reset_order: bool = False):
        '''
        Adiciona conjunto de regras à instancia atual a partir do objeto TemporalRelation.rules.

        Args:
            setRules_obj: objeto TemporalRelation.rules que é uma lista de lista
        '''
        if type(setRules_obj) != list or type(setRules_obj[0]) not in [list, str]:
            raise ValueError('ERROR: setRules_obj não é do tipo List[list]')

        try:
            for rule in setRules_obj:
                self.__add_myrule_str(rule, reset_cod_regra=reset_cod_regra, reset_order=reset_order)
                        
        except ErrorRulesFiltered as e:
            print(f'ERROR: {e}')
        except Exception as e:
            print(f"Erro ao adicionar regras 'my_rules' ao conjunto de regras atual. ERRO: {e} \n{str(rule)}")
        else:
            print(f'As regras do objeto foram adicionadas com sucesso.')
            print(f'Total de regras: {len(self.rules)}\n')

    def add_setRules_ojb(self, algoritmo: str, setRules_obj, relType: str = '', verbosy: bool = True):
        '''
        Adicionar regras no setRules atual conforme algoritmo gerador de regras.

        Args:
            algoritmo: são válidos 'RIPPER', 'CBA', 'IDS', 'MY_RULES', 'SELF'.

            setRules_obj: Objeto que representa o conjunto de regras gerado por 'algoritmo'.
                Se RIPPER: classificador.ruleset_.rules
                Se CBA: classificador.rules
                Se IDS: classificador.clf.rules
                Se MY_RULES: TemporalRelation.rules
                Se SELF: TemporalRelation.SetRules.rules

            relType: Se o algoritmo for RIPPLE, é obrigatório informar o tipo da relação temporal das regras. 
                    Este algoritmo é de classificação binária, só gera regras para uma classe de cada vez.
        '''
        algoritmos = ['RIPPER', 'CBA', 'IDS', 'MY_RULES', 'SELF']
        algoritmo = algoritmo.upper()

        if algoritmo not in algoritmos:
            print(f'Algoritmo {algoritmo} inválido. \nVálido: {str(algoritmos)}')
            return

        if not setRules_obj:
            print('setRules_obj não existe ou não foi processado o classificador.')
            return

        if type(setRules_obj) not in [list, dict]:
            print('setRules_obj deve ser do tipo list ou dict contendo as regras.')
            return

        try:
            if algoritmo == 'RIPPER':
                if not relType:
                    print(f'Se o algoritmo for RIPPLE, é obrigatório informar o tipo da relação das regras (relType). \nEste algoritmo é de classificação binária, só gera regras para uma classe de cada vez.')
                    return

                #setRules_obj = ripper_clf.ruleset_.rules
                for rule in setRules_obj:
                    self.add_rule(relType=relType, origem=algoritmo)
                    antecedente_obj = rule.conds
                    for ante in antecedente_obj:
                        self.rule.add_predicate(str(ante.feature), '=', str(ante.val))

            if algoritmo == 'CBA':
                #setRules_obj = classifier.rules
                for rule in setRules_obj:
                    relType = rule.consequent.value
                    self.add_rule(relType=relType, origem=algoritmo)
                    antecedente_obj = rule.antecedent.itemset
                    for ante in antecedente_obj:
                        self.rule.add_predicate(ante, '=', rule.antecedent.itemset[ante])

            if algoritmo == 'IDS':
                #setRules_obj = ids.clf.rules
                for rule in setRules_obj:
                    relType = rule.car.consequent.value
                    self.add_rule(relType=relType, origem=algoritmo)
                    antecedente_obj = rule.car.antecedent.itemset
                    for ante in antecedente_obj:
                        self.rule.add_predicate(ante, '=', rule.car.antecedent.itemset[ante])
            
            if algoritmo == 'MY_RULES':
                self.__add_setRules_myrules_obj(setRules_obj, reset_cod_regra=True, reset_order=False)

            if algoritmo == 'SELF':
                self.__add_setRules(setRules_obj) #sempre reseta cod_regra e mantem ordem

        except ErrorRulesFiltered as e:
            print(f'ERROR: {e}')
        except Exception as e:
            print(f"Erro ao adicionar regras ao conjunto de regras atual. ERRO em 'add_setRules_ojb': {e}")
        else:
            if verbosy:
                print(f"As regras '{algoritmo}' foram adicionadas com sucesso e disponível em TemporalRelation.SetRules.rules. Total de regras: {len(self.rules)}\n")

    
    def has_rule_class_default(self) -> str:
        '''
        Verifica se há regras com classe default.
        Se houver, retorna a regra de classe default.
        '''
        rule_default = list(filter(lambda x: 'True == True' in str(x.predicates), self.rules))
        if len(rule_default) >= 1:
            return rule_default[0] #retorna regra default
        return False # se não houver regra default

    def remove_rule_class_default(self):
        '''
        Remove todas as regras de classe default, se houver.
        '''
        if self.has_rule_class_default():
            while self.has_rule_class_default() in self.rules:
                print('Regra removida:', self.has_rule_class_default())
                self.rules.remove(self.has_rule_class_default())
        else:
            print('Não há regra de classe default.')

    def add_rule_class_default(self, class_default: str):
        '''
        Adicionar regra para classe default em TemporalRelation.SetRules.rules
        '''
        if self.has_rule_class_default():
            print('Já tem regra para classe default:', self.has_rule_class_default().relType)
        else:
            class_default = class_default.upper()
            self.add_rule(relType=class_default, ordem=2000, origem='DEFAULT')
            self.rule.add_predicate('True', '=', 'True')
            print('Regra default adicionada.')


    @property
    def rule(self):
        '''
        Retorna objeto Rule
        '''
        return self.__rule
    
    @rule.setter
    def rule(self, rule):
        self.__rule = rule
    
    
    @property
    def rules(self):
        '''
        Retorna conjunto de regras em formato de lista de objetos Rule
        '''
        return self.__rules
    
    @rules.setter
    def rules(self, rules):
        self.__rules = rules
        
    @property
    def rules_str(self):
        '''
        Retorna o conjunto de regras em formato de lista de listas
        '''
        return [eval(str(rule)) for rule in self.rules]
    
    @property
    def df_rules(self):
        '''
        Exibe as regras ativas em formato de tabela.
        '''
        colunas=['cod_regra', 'relType', 'ordem', 'predicados', 'origem', 'acuracia', 'acertos', 'acionamentos']
        return pd.DataFrame(self.rules_str, columns=colunas)
    
    @property
    def is_filter(self):
        return self.__is_filter
    
    
    def __features_counts(self) -> dict:
        '''
        Retorna dicionário contendo a quantidade de ocorrências de cada feature no conjunto de regras por tipo de relação temporal.
        Ex: {'feature1': {'AFTER': 3, 'BEFORE': 5}, 'feature2': {'AFTER': 4, 'OVERLAP': 6} }
        '''
        quant_features = defaultdict(lambda: defaultdict(int))
        for rule in self.rules:
            relType = rule.relType
            for predicate in rule.predicates:
                feature = self.__trata_feature(predicate.feature)
                quant_features[feature][relType] += 1
        return quant_features

    @property
    def df_features_counts(self) -> DataFrame:
        '''
        Retorna DataFrame contendo a quantidade de ocorrências de cada feature no conjunto de regras por tipo de relação temporal.
        Se o setRules conter mais de uma origem, será gravada apenas a primeira, que deverá ser ignorada.
        '''
        #if type(setRules) == SetRules:
        #    setRules = setRules.rules
        
        #if type(setRules) != list or type(setRules[0]) != SetRules.Rule:
        #    raise TypeError(f"ERROR: 'setRules' deve ser do tipo SetRules.")
        
        df = pd.DataFrame.from_dict(self.__features_counts(), orient='index').fillna(0)
        df['Total'] = df.sum(axis=1)
        df['origem'] = self.rules[0].origem

        df.reset_index(inplace=True)
        df.rename(columns={'index':'feature'}, inplace=True)
        df.set_index(['feature', 'origem'], inplace=True)
        
        return df.sort_values('Total', ascending=False)



#----- FIM SETRULES ----------------------------------------------------
#===============================================================================


#=========================================================================================================================
#=========================================================================================================================
# '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
#  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
#  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
#  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
#  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
#  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
#  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
# ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #
#=========================================================================================================================
#=========================================================================================================================


#---------------------------------------------------------------------
#     CLASSE RULESFUNCTIONS
#--------------------------------------------------------------------


class RulesFunctions():
    '''
    Funções que implementam as features que identificam Relações Temporais, 
    além de funções auxiliarem que podem compor regras manuais que identificam Relações Temporais.

    
    '''
    def __init__(self, tb: TimebankPT):
        
        self.__tb = tb
        self.signals = self.list_temporal_signal()
        
        #Criei outro nlp para melhorar performance, neste pipe só precisa do lemmatizer, os outros pipes foram excluídos
        #python 3.9
        #self.__nlp_lemma = spacy.load('pt_core_news_lg', exclude= ['tok2vec', 'morphologizer', 'parser', 'attribute_ruler', 'ner'])
        #python 3.11
        self.__nlp_lemma = spacy.load('pt_core_news_lg', exclude= ['morphologizer', 'parser', 'attribute_ruler', 'ner'])
        
    
    def search(self, palavras, frase, lemma: bool = False):
        '''
        Verifica se 'palavras' inteiras estão presentes em 'frase'.

        Args:
            palavras:   Pode ser list, str, Token ou expressão regular.
                        Pesquisa por palavras inteiras.
                        Regex gerado: (\W|^)(palavras_separadas_por_pipe)(\W|$)
            frase:  Texto onde as 'palavras' serão encontradas.
                    Pode ser list, str, Doc, Span e Token
            lemma:  Se True, lematiza palavras e frase.

        '''
        if lemma:
            palavras = self.__lemma(palavras)
            frase = self.__lemma(frase)

        palavras = self.list_to_str(palavras, '|')
        frase = self.list_to_str(frase, ' ')

        #verificar se \b é melhor que \W
        reg = "(\W|^)(" + palavras + ")(\W|$)"
        for match in re.finditer(reg, frase, flags=re.IGNORECASE):
            if match is not None:
                return True
        return False


    def list_to_str(self, lista, delimitador: str = ' ') -> str:
        '''
        Converte 'lista' em string minúsculas.

        Args:
            lista: Pode ser: str, list, Doc, Span, Token
            delimitador: separador de cada palavra na string quando 'lista' for do tipo list.

        Return:
            Se 'lista' for do tipo list, converte-a em string minúsculas separadas por 'delimitador'. 
            Se 'lista' for do tipo Doc, Span ou Token, converte em string minúsculas.
            Se não, retorna string minúsculas.

        '''
        if type(lista) in [Doc, Span, Token]:
            lista = lista.text

        if type(lista) in [list, tuple]:
            lista = delimitador.join(lista)

        return lista.lower().strip()

    def str_to_list(self, palavra: str) -> list:
        '''
        Se 'palavra' for string, converte em lista unitária de string maiúsculas.
        Se não, converte em maiúsculas.
        '''
        if type(palavra) == int:
            palavra = str(palavra)

        if type(palavra) == str:
            lista = []
            lista.append(palavra)
            palavra = lista

        palavra = list(map(str.upper, palavra))

        return palavra

    def __lemma(self, frase):
        '''
        Recebe frase e retorna as palavras lemmatizadas.

        Args:
            frase:  Texto que deseja lematizar. 
                    Pode ser Doc, Span, Token, list ou str.

        Return:
            Se frase for Doc, Span, Token ou str, retorna string lematizada.
            Se frase for list, retorna lista com elementos únicos lematizados.

        '''
        if type(frase) in [Span, Token, Doc]:
            if type(frase) == Doc:
                frase = frase[0:len(frase)]  # to Span
            return frase.lemma_   #  str

        elif type(frase) == str:
            doc = self.__nlp_lemma(frase.lower())
            span = doc[0:len(doc)]
            return span.lemma_    #  str

        elif type(frase) in [list, tuple]:
            lista_lemma = []
            for palavra in frase:
                doc = self.__nlp_lemma(palavra.lower())
                span = doc[0:len(doc)]
                lista_lemma.append(span.lemma_)
            return list(set(lista_lemma))  #  list única

        else:
            print('ERROR: função __lemma recebeu tipo desconhecido.')
            return

    def __doc_ids(self, ids) -> List[Doc]:
        '''
        Retorna Doc os Docs das sentença atual baseado no 'ids' do parâmetro.
        
        args:
            ids: representa o índice do Doc atual
                
        Return:
            se ids for dict no formato {'antes':[], 'depois':[]} -> retorna lista de Docs, o que vem antes e o que vem depois do token em questão
            se ids for list -> retorna lista com um Doc da sentença entre o id mínimo e o máximo.
        '''
        if type(ids) == dict:
            return [self.__tb.doc_unico[min(ids['antes']):max(ids['antes'])], self.__tb.doc_unico[min(ids['depois']):max(ids['depois'])]]
        else:
            if len(ids) == 0:
                return []
            return [self.__tb.doc_unico[min(ids):max(ids)]]    #Não adiciona mais 1 ao max porque já foi adicionado em __idsContexto.
        
        
    def __frase_ids(self, ids) -> str:
        '''
        Retorna texto da sentença atual baseado no 'ids' do parâmetro.
        
        args:
            ids: representa o índice do Doc atual
                se dict no formato {'antes':[], 'depois':[]} concatena texto que vem antes com o que vem depois
                se list retorna texto da sentença entre o id mínimo e o máximo.
                
        Return:
            String.
        '''
        docs_ids = self.__doc_ids(ids)
        frase = ''
        
        if type(docs_ids) == list:
            for doc in docs_ids:
                frase += ' ' + doc.text + ' '
        else:
            frase = doc.text 
            
        return frase.strip()
        

    def __idsContexto(self, token: Token, distancia = 5, contexto = None) -> list:
        '''
        Retorna lista de i dos tokens dentro da distância conforme contexto (antes ou depois do token).
        Se contexto 'antes', o 'token' é incluído. Se 'depois', é adicionado mais um token no final.

        Args:
            token: objeto Token do spaCy
            distancia:  Quantidade de tokens antes ou depois do token. 
                        Se 'max' (ou qualquer string), estende até o final ou o início da sentença, conforme contexto.
            contexto: 'antes' ou 'depois' do token

        '''
        if type(token) != Token:
            print("ERROR: 'token' deve ser do tipo Token (objeto do spaCy).")
            return []

        if contexto:
            contexto_valido = ['antes', 'depois']
            if type(contexto) == int:
                print('ERROR: Contexto obrigatório. Valores válidos: ' + ', '.join(contexto_valido))
                return []

            contexto = contexto.lower()
            if contexto not in contexto_valido:
                print('ERROR: Contexto válido: ' + ', '.join(contexto_valido))
                return []

        if type(distancia) == str:  # distância máxima. Final ou início da sentença.
            distancia = len(token.doc)-1

        #ids ANTES
        lim_inf = 0 if (token.i - distancia) < 0 else (token.i - distancia)
        ids_antes = list(range(lim_inf, token.i + 1))
        #ids DEPOIS
        quant_tokens = len(self.__tb.doc_unico)
        lim_sup = quant_tokens if (token.i + distancia) >= quant_tokens else (token.i + distancia) + 1
        ids_depois = list(range(token.i + 1, lim_sup + 1))
        
        ids = {'antes':ids_antes, 'depois':ids_depois}
        
        if contexto == 'antes':
            return ids['antes']
            
        elif contexto == 'depois':
            return ids['depois']

        return ids


    def __idsInBetween(self, token1: Token, token2: Token) -> list:
        '''
        Retorna lista de i dos tokens entre os tokens de início e de fim. 
        Exclusive o início e Inclusive o fim.

        '''
        if type(token1) != Token or type(token2) != Token:
            print("ERROR: 'token1' e 'token2' devem ser do tipo Token (objeto do spaCy).")
            return []

        i_ini = token1.i
        i_fim = token2.i

        if i_ini > i_fim:
            i_ini, i_fim = i_fim, i_ini

        i_ini += 1 #não entra o próximo token de inicio
        i_fim += 1 #em python o ultimo elemento da lista não é computado, e aqui é necessário, por isso +1

        return list(range(i_ini, i_fim))


    def lengthInBetween(self, token1: Token, token2: Token) -> int:
        '''
        Retorna distância entre os tokens.

        '''
        return len(self.__idsInBetween(token1, token2)) - 1


    def closelyFollowing(self, token1: Token, token2: Token, distancia = 10) -> bool:
        '''
        Retorna True se os token estiverem a uma distancia de no máximo 'distancia' tokens.

        '''
        return self.lengthInBetween(token1, token2) <= distancia

    def t1BeforeT2(self, t1: Token, t2: Token) -> bool:
        '''
        Retorna True se t1 estiver posicionado na frase antes de t2. Senão retorna False.
        
        '''
        if t1.i < t2.i:
            return True
        return False

    def t1AfterT2(self, t1: Token, t2: Token) -> bool:
        '''
        Retorna True se t1 estiver posicionado na frase depois de t2. Senão retorna False.
        
        '''
        return self.t1BeforeT2(t2, t1)
    
    
    def isEvent(self, token):
        '''
        Verifica se o token é um EVENT
        '''
        if type(token) != Token:
            print('ERROR: argumento não é do tipo Token.')
            return False
        return token.ent_type_ == 'EVENT'

    def isTimex3(self, token):
        '''
        Verifica se o token é um TIMEX3
        '''
        if type(token) != Token:
            print('ERROR: argumento não é do tipo Token.')
            return False
        return token.ent_type_ == 'TIMEX3'



    #--------------------------------------
    #WORD
    def __hasWord(self, palavras: list, ids: list, lemma: bool = False) -> bool:
        '''
        Verifica se existe uma das 'palavras' nos tokens representados por 'ids'.

        Args:
            ids: lista com intervalo de tokens.i
        '''
        
        frase = self.__frase_ids(ids)
        
        if self.search(palavras, frase, lemma):
            return True
        return False


    def hasWordInContext(self, token: Token, palavras: list, distancia = 5, contexto = None, lemma: bool = False) -> bool:
        '''
        Verifica se existe uma das 'palavras' a uma certa distância do token, observando o contexto (antes ou depois do token).

        '''
        ids = self.__idsContexto(token, distancia, contexto)
        if not ids:
            return False

        return self.__hasWord(palavras, ids, lemma)


    def hasWordInContextPrecede(self, token: Token, palavras: list, distancia = 5, lemma: bool = False) -> bool:
        '''
        Verifica se existe 'palavras' a uma distância de até 5 tokens antes do 'token'.

        Args:
            token: Objeto Token do spaCy
            palavras: Expressão que deseja encontrar. Pode ser string, lista de strings, Token ou expressão regular.
                    Pesquisa por palavras inteiras.
                    Regex gerado: (\W|^)(palavras_separadas_por_pipe)(\W|$)
            distancia: Quantidade de tokens antes ou depois do token. 
                    Se 'max' (ou qualquer string), estende até o final ou o início da sentença, dependendo do contexto.
            lemma: Se True, lematiza palavras e contexto.
        '''
        return self.hasWordInContext(token, palavras, distancia, 'antes', lemma)

    def hasWordInContextFollow(self, token: Token, palavras: list, distancia = 5, lemma: bool = False) -> bool:
        '''
        Verifica se existe 'palavra' a uma distância de até 5 tokens depois do 'token'.

        Args:
            token: Objeto Token do spaCy
            palavras: Expressão que deseja encontrar. Pode ser string, lista de strings, Token ou expressão regular.
                    Pesquisa por palavras inteiras.
                    Regex gerado: (\W|^)(palavras_separadas_por_pipe)(\W|$)
            distancia: Quantidade de tokens antes ou depois do token. 
                    Se 'max' (ou qualquer string), estende até o final ou o início da sentença, dependendo do contexto.
            lemma: Se True, lematiza palavras e contexto.

        '''
        return self.hasWordInContext(token, palavras, distancia, 'depois', lemma)


    def hasWordInBetween(self, token1: Token, token2: Token, palavras, lemma: bool = False) -> bool:
        '''
        Verifica se há 'palavras' entre a entidade 1 e entidade 2.

        Args:
            token1 e token2: objeto Token do spaCy.
            palavras: expressão que deseja encontrar. Pode ser string, lista de strings, Token ou expressão regular.
                    Pesquisa por palavras inteiras.
                    Regex gerado: (\W|^)(palavras_separadas_por_pipe)(\W|$)
            lemma: Se True, lematiza palavras e contexto.
        '''
        ids = self.__idsInBetween(token1, token2)
        if not ids:
            return False

        return self.__hasWord(palavras, ids, lemma)


    def __tratar_id_token_unico(self, token: Token) -> list:
        '''
        Trata list de ids para token único.
        Acrescenta mais um na lista.
        '''
        if not token:
            return
        ids = []
        ids.append(token.i)
        ids.append(token.i + 1)
        return ids


    #--------------------------------------
    #POS
    def pos(self, token: Token, pos: list) -> bool:
        '''
        Verifica se 'token' é uma das classes gramaticais da lista 'pos'.

        Args:
            token: Token
            pos: lista de classes gramaticais
        '''
        ids = self.__tratar_id_token_unico(token)
        return self.__hasPos(pos, ids)

    def __hasPos(self, pos: list, ids: list) -> bool:
        '''
        Verifica se há POS tag na lista de ids.

        Args:
            pos: POS tag
            ids: lista de token.i

        '''
        pos = self.str_to_list(pos)
        list_pos = self.__tb.siglas.get_list('pos')
        for p in pos:
            if p not in list_pos:
                print("Pos: '{0}' inválido.\nPOS válidos: {1} ".format(p, list_pos))
                return False

        docs_ids = self.__doc_ids(ids)
        
        for doc in docs_ids:
            for token_atual in doc:
                if token_atual.pos_.upper() in pos:
                    return True
        return False


    def hasPosInContext(self, token, pos: list, distancia = 5, contexto = None) -> bool:
        '''
        Verifica se existe a classe gramatical 'pos' a uma distância de até 5 tokens, observando o contexto (antes ou depois do token).
        '''
        ids = self.__idsContexto(token, distancia, contexto)
        if not ids:
            return False

        if not pos:
            return False

        return self.__hasPos(pos, ids)

    def hasPosInContextPrecede(self, token: Token, pos: list, distancia = 5) -> bool:
        '''
        Verifica se existe a classe gramatical 'pos' a uma distância de até 5 tokens antes do 'token'.

        Args:
            token: Objeto Token do spaCy.
            pos: Classe gramatical - POS Tag.
            distancia: Quantidade de tokens antes de 'token'. 
                    Se 'max' (ou qualquer string), estende até o início da sentença, conforme contexto.
        '''
        return self.hasPosInContext(token, pos, distancia, 'antes')

    def hasPosInContextFollow(self, token: Token, pos: list, distancia = 5) -> bool:
        '''
        Verifica se existe a classe gramatical 'pos' a uma distância de até 5 tokens depois do 'token'.

        Args:
            token: Objeto Token do spaCy.
            pos: Classe gramatical - POS Tag.
            distancia:  Quantidade de tokens depois de 'token'. 
                        Se 'max' (ou qualquer string), estende até o fim da sentença, conforme contexto.
        '''
        return self.hasPosInContext(token, pos, distancia, 'depois')

    def hasPosInBetween(self, token1: Token, token2: Token, pos: list) -> bool:
        '''
        Verifica se existe a classe gramatical 'pos' entre os dois tokens.
        '''
        ids = self.__idsInBetween(token1, token2)
        if not ids:
            return False

        if not pos:
            return False

        return self.__hasPos(pos, ids)

    
    #VERBS

    def verbform(self, token: Token) -> str:
        '''
        Retorna a forma verbal. Ex: Inf, Fin.
        '''
        verbform_token = list(map(str.upper, token.morph.get('VerbForm')))
        if verbform_token:
            verbform_token = verbform_token[0]

        return verbform_token

    def verbform_check(self, token: Token, verbform: list) -> bool:
        '''
        Verifica se 'token' possui modo verbal 'mood'.

        Args:
            token: Token
            mood: str ou list
                mood válidos: 'Fin', 'Ger', 'Inf', 'Part'

        '''
        verbform_valid = ['FIN', 'GER', 'INF', 'PART']
        verbform = self.str_to_list(verbform)

        for v in verbform:
            if v not in verbform_valid:
                print('ERROR: modo verbal inválido. Valores válido: ' + str(verbform_valid))
                return False

        verbform_atual = self.verbform(token)
        if not verbform_atual:
            return False

        return verbform_atual in verbform
    
    
    def mood(self, token: Token) -> str:
        '''
        Retorna o modo verbal do token
        '''
        mood_token = list(map(str.upper, token.morph.get('Mood')))
        if mood_token:
            mood_token = mood_token[0]

        return mood_token
        
    def mood_check(self, token: Token, mood: list) -> bool:
        '''
        Verifica se 'token' possui modo verbal 'mood'.

        Args:
            token: Token
            mood: str ou list
                mood válidos: 'Cnd', 'Imp', 'Ind', 'Sub'
        '''
        mood_valid = ['CND', 'IMP', 'IND', 'SUB']
        mood = self.str_to_list(mood)

        for m in mood:
            if m not in mood_valid:
                print('ERROR: modo verbal inválido. Valores válido: ' + str(mood_valid))
                return False

        mood_atual = self.mood(token)
        if not mood_atual:
            return False

        return mood_atual in mood
    
    
    def tense(self, token: Token) -> str:
        '''
        Retorna o tempo verbal de 'token'
        Function tenseVerb(token, list_tense) pode ser usada para comparações
        '''
        tense_token = list(map(str.upper, token.morph.get('Tense')))
        if len(tense_token) > 0:
            tense_token = tense_token[0]
        else:
            tense_token = ''

        return tense_token
    
    def tense_compound(self, token: Token) -> str:
        '''
        Retorna o tempo verbal composto do token. Por enquanto, apenas do modo indicativo.
        
        Returns:
            . PRETPC    = pretérito perfeito composto do indicativo
            . FPRESC    = futuro do presente composto do indicativo
            . PRETMQPC  = pretérito mais-que-perfeito composto do indicativo
            . FPRETC    = futuro do pretérito composto do indicativo
        '''
        t_ant1 = self.__tb.nbor(token, -1)
        t_ant2 = self.__tb.nbor(token, -2)
        
        if not(t_ant1 and t_ant2):
            return None
        
        #PRETPC: pretérito perfeito composto do indicativo
        if self.pos(t_ant1, ['AUX']) and self.tenseVerb(t_ant1, ['PRES']) and self.mood_check(t_ant1, ['Ind']) and self.verbform_check(token, ['Part']):
            return 'PRETPC'
        
        #FPRESC: futuro do presente composto
        if self.pos(t_ant1, ['AUX']) and self.tenseVerb(t_ant1, ['Fut']) and self.mood_check(t_ant1, ['Ind']) and self.verbform_check(token, ['Part']):
            return 'FPRESC'
        
        if self.pos(t_ant2, ['AUX']) and self.tenseVerb(t_ant2, ['Pres']) and self.mood_check(t_ant2, ['Ind']) and (t_ant2.lemma_.lower() == 'ir') and self.verbform_check(t_ant1, ['Inf']) and self.verbform_check(token, ['Part']): 
            return 'FPRESC'
        
        #PRETMQPC: pretérito mais-que-perfeito composto do indicativo
        if self.pos(t_ant1, ['AUX']) and self.tenseVerb(t_ant1, ['Imp']) and self.mood_check(t_ant1, ['Ind']) and self.verbform_check(token, ['Part']): 
            return 'PRETMQPC'
        
        #FPRETC: futuro do pretérito composto do indicativo
        if self.pos(t_ant1, ['AUX']) and self.mood_check(t_ant1, ['Cnd']) and self.verbform_check(token, ['Part']): 
            return 'FPRETC'
        
        return None
    
    def tense_compound_check(self, token: Token, tense_compound: list) -> bool:
        '''
        Verifica se 'token' possui tempo verbal composto 'tense_compound'.

        Args:
            token: Token
            tense_compound: str ou list
                tempos verbais compostos: 'PRETPC', 'FPRESC', 'PRETMQPC', 'FPRETC'

        '''
        tense_valid = ['PRETPC', 'FPRESC', 'PRETMQPC', 'FPRETC']
        tense_compound = self.str_to_list(tense_compound)

        for t in tense_compound:
            if t not in tense_valid:
                print('ERROR: tempo verbal composto inválido. Valores válido: ' + str(tense_valid))
                return False

        tense_atual = self.tense_compound(token)
        if not tense_atual:
            return False

        return tense_atual in tense_compound
    
    

    def hasNoVerbInContext(self, token: Token, distancia = 5, contexto = None) -> bool:
        '''
        Checks for verb within entity_context of 5 words
        Verifica se existe VERBO a uma distância de até 5 tokens, observando o contexto (antes ou depois do token).
        '''
        ids = self.__idsContexto(token, distancia, contexto)
        if not ids:
            return False

        return not self.__hasPos(['VERB', 'AUX'], ids)

    def hasNoVerbInContextPrecede(self, token: Token, distancia = 5) -> bool:
        '''
        Verifica se existe VERBO a uma distância de até 5 tokens antes do 'token'.

        Args:
            token: objeto Token
            distancia:  Quantidade de tokens antes de 'token'. 
                        Se 'max' (ou qualquer string), estende até o início da sentença, conforme contexto.
        '''
        return self.hasNoVerbInContext(token, distancia, 'antes')

    def hasNoVerbInContextFollow(self, token: Token, distancia = 5) -> bool:
        '''
        Verifica se existe VERBO a uma distância de até 5 tokens depois do 'token'.

        Args:
            token: objeto Token
            distancia:  Quantidade de tokens depois de 'token'. 
                        Se 'max' (ou qualquer string), estende até o fim da sentença, conforme contexto.
        '''
        return self.hasNoVerbInContext(token, distancia, 'depois')


    def hasNoVerbInBetween(self, token1: Token, token2: Token) -> bool:
        '''
        Verifica se não há verbos entre os tokens de início e de fim.
        '''
        ids = self.__idsInBetween(token1, token2)
        if not ids:
            return False

        return not self.__hasPos(['VERB', 'AUX'], ids)


    def hasTenseVerbInBetween(self, token1: Token, token2: Token, tense):
        '''
        Verifica se tem VERB com tempo 'tense' entre dos dois tokens.

        '''
        ids = self.__idsInBetween(token1, token2)
        if not ids:
            return False

        for token_atual in self.__tb.doc_unico[min(ids):max(ids)]:
            if self.tenseVerb(token_atual.upper(), tense.upper()):
                return True
        return False

    def sameTense(self, token1: Token, token2: Token) -> bool:
        '''
        Verificar se os tokens possuem o mesmo tempo verbal.
        '''
        tempo1 = self.tense(token1)
        tempo2 = self.tense(token2)

        if not (tempo1 or tempo2):
            return False
        return tempo1 == tempo2

    def tenseVerb(self, token: Token, tense: list) -> bool:
        '''
        Verifica se 'token' possui tempo verbal 'tense'.

        Args:
            token: Token
            tense:  str ou list
                    tempo verbal válidos: 'FUT', 'IMP', 'PAST', 'PQP', 'PRES'

        '''
        tense_valid = ['FUT', 'IMP', 'PAST', 'PQP', 'PRES']
        tense = self.str_to_list(tense)

        for t in tense:
            if t not in tense_valid:
                print('ERROR: Regras contendo tempo verbal inválido. Valores válido: ' + str(tense_valid))
                return False

        tense_atual = self.tense(token)
        if not tense_atual:
            return

        return tense_atual in tense

    def verbGerundio(self, token: Token) -> bool:
        '''
        Verifica se é gerúndio.
        '''
        return self.morph(token, ('VerbForm', 'Ger'))

    #--------------------------------------
    #DEPENCENCY
    def dep(self, token: Token, dep: list) -> bool:
        '''
        Verifica se 'token' possui um dos tipos de dependência da lista 'dep'.

        Args:
            token: Token
            dep: lista de tipo de dependências.
        '''
        ids = self.__tratar_id_token_unico(token)
        return self.__hasDep(dep, ids)

    def __validaDep(self, dep: list):
        '''
        Verifica se 'dep' é válido
        '''
        dep = self.str_to_list(dep)
        list_dep = self.__tb.siglas.get_list('dep')
        print_dep = list(self.__tb.siglas.print_dep().keys())
        deps_false = []
        for d in dep:
            if d not in list_dep:
                deps_false.append(d)
        
        if len(deps_false) > 0:
            for dep_false in deps_false:
                print("Dep: '{0}' inválido.".format(dep_false))
            print("\nDEP válidos:\n{0}".format(print_dep))
            return False
        else:
            return True
            
    def __hasDep(self, dep: list, ids: list):
        '''
        Verifica se existe a dependência 'dep' nos token.i representados por 'ids'.
        '''
        dep = self.str_to_list(dep)
        if not self.__validaDep(dep):
            return False
        
        docs_ids = self.__doc_ids(ids)
        
        for doc in docs_ids:
            for token_atual in doc:
                if (token_atual.dep_.upper() in dep):
                    return True
        return False

    def hasDepInContext(self, token, dep: list, distancia = 5, contexto = None) -> bool:
        '''
        Verifica se existe a dependência 'dep' a uma distância de até 5 tokens, observando o contexto (antes ou depois do token).
        '''
        ids = self.__idsContexto(token, distancia, contexto)
        if not ids:
            return False

        return self.__hasDep(dep, ids)

    def hasDepInContextPrecede(self, token: Token, dep: list, distancia = 5) -> bool:
        '''
        Verifica se existe a dependência 'dep' a uma distância de até 5 tokens antes do 'token'.

        Args:
            token: objeto Token
            distancia:  Quantidade de tokens antes de 'token'. 
                        Se 'max' (ou qualquer string), estende até o início da sentença, conforme contexto.
        '''
        return self.hasDepInContext(token, dep, distancia, 'antes')

    def hasDepInContextFollow(self, token: Token, dep: list, distancia = 5) -> bool:
        '''
        Verifica se existe a dependência 'dep' a uma distância de até 5 tokens depois do 'token'.

        Args:
            token: objeto Token
            distancia:  Quantidade de tokens depois de 'token'. 
                        Se 'max' (ou qualquer string), estende até o fim da sentença, conforme contexto.
        '''
        return self.hasDepInContext(token, dep, distancia, 'depois')

    def hasDepInBetween(self, token1: Token, token2: Token, dep: list):
        '''
        Verifica se existe a dependência 'dep' entre os dois tokens.
        '''
        ids = self.__idsInBetween(token1, token2)
        if not ids:
            return False

        return self.__hasDep(dep, ids)
    
    def is_dependencyType(self, tokenPai: Token, tokenFilho: Token, tipo_dep: str) -> bool:
        '''
        Checks for type(token1=governor, token2=dependent)
        Verifica se a relação de dependência entre 'tokenPai' e 'tokenFilho' é 'tipo_dep'.

        Args:
            tokenPai: governor
            tokenFilho: dependent
            tipo_dep: String que representa o tipo de dependência de tokenFilho para tokenPai.

        '''
        if not self.__validaDep(tipo_dep):
            return False
        
        if self.dependencyType(tokenPai, tokenFilho) == tipo_dep.upper():
            return True
        return False
        
    def dependencyType(self, tokenPai: Token, tokenFilho: Token) -> str:
        '''
        Retorna a relação de dependência entre 'tokenPai' e 'tokenFilho'.

        Args:
            tokenPai: governor
            tokenFilho: dependent
        
        Retorna
            String que representa o tipo de dependência de tokenFilho para tokenPai.
            
        '''
        if tokenPai.i == tokenFilho.head.i:
            return tokenFilho.dep_.upper()
        return ''

    def children(self, tPai: Token) -> list:
        '''
        Retorna lista contendo todos descendentes de tPai.
        Usa busca em profundidade para verificar todos os filhos e filhos dos filhos.
        
        Args:
            tPai: Token que deseja conhecer os descendentes
        '''
        def dfs(node, level = 0):
            if node not in visited:
                visited.add(node)
                for neighbour in node.children:
                    lista_filhos.append(neighbour)
                    dfs(neighbour, level+1)

        visited = set()
        lista_filhos = []
        dfs(tPai)

        return lista_filhos

    def is_child(self, tPai: Token, tFilho: Token) -> bool:
        '''
        Retorna True se tFilho for descendente de tPai
        
        Args:
            tPai: Token pai
            tFilho: Token filho
        '''
        #return tFilho in self.children(tPai)
        for child in self.children(tPai):
            if child.i == tFilho.i:
                return True
        return False

    #--------------------------------------
    #MORPH

    def morph(self, token: Token, morph: tuple) -> bool:
        '''
        Verifica se 'token' possui o elemento morph representado pela tupla (key, value) da análise morfológica.

        Args:
            token: Token
            morph: tupla (key, value), ex: ('Tense', 'Fut')
        '''
        ids = self.__tratar_id_token_unico(token)
        return self.__hasMorph(morph, ids)

    def __hasMorph(self, keyvalue: tuple, ids: list) -> bool:
        '''
        Verifica se existe na análise morfológica o 'key' de valor 'value' nos token.i representados por ids.

        '''
        if type(keyvalue) not in [tuple, list]:
            print("ERROR: 'keyvalue' deve ser um tupla (key, value), ex: ('Tense', 'Fut')")
            return False

        key = keyvalue[0].upper()
        value = keyvalue[1].upper()
        
        if type(key) != str or type(value) != str:
            print("ERROR:'key' or 'value' inválidos.")
            return False
        
        list_morph_key = self.__tb.siglas.get_list('morph_key')
        if key not in list_morph_key:
            print("Key Morph: '{0}' inválido.\nKeys de Morph válidos: {1} ".format(key, list_morph_key))
            return False

        #
        morph_upper = defaultdict(str)

        docs_ids = self.__doc_ids(ids)
        
        for doc in docs_ids:
            for token_atual in doc:
                #valores de morph em maiúsculas
                morph_upper.clear()
                morph_dict = token_atual.morph.to_dict()
                for k in morph_dict:
                    morph_upper[k.upper()] = morph_dict[k].upper()

                if morph_upper:
                    if value == morph_upper.get(key):
                        return True
                    #Para os casos em que desejo encontrar um key com value vazio, ex: ('Tense', '') ou ('VerbForm', '')
                    if morph_upper.get(key) == None and value == '':
                        return True
                    
        return False

    def hasMorphInContext(self, token: Token, keyvalue: tuple, distancia = 5, contexto = None):
        '''
        Verifica se existe na análise morfológica o 'key' de valor 'value' a uma distância de até 5 tokens, observando o contexto (antes ou depois do token).
        '''
        ids = self.__idsContexto(token, distancia, contexto)
        if not ids:
            return False

        return self.__hasMorph(keyvalue, ids)

    def hasMorphInContextPrecede(self, token: Token, keyvalue: tuple, distancia = 5):
        '''
        Verifica se existe na análise morfológica o 'key' de valor 'value' a uma distância de até 5 tokens antes de 'token'.

        Args:
            token: objeto Token do spaCy.
            keyvalue:   Tuple que representa o par da morfologia que deseja (key, value). 
                        Ex:  (Tense, Fut) ou (VerbForm, Inf).
            distancia:  Quantidade de tokens antes de 'token'. 
                        Se 'max' (ou qualquer string), estende até o início da sentença, conforme contexto.
        '''
        return self.hasMorphInContext(token, keyvalue, distancia, 'antes')

    def hasMorphInContextFollow(self, token: Token, keyvalue: tuple, distancia = 5):
        '''
        Verifica se existe na análise morfológica o 'key' de valor 'value' a uma distância de até 5 tokens depois de 'token'.

        Args:
            token: objeto Token do spaCy.
            keyvalue:   Tuple que representa o par da morfologia que deseja (key, value). 
                        Ex:  (Tense, Fut) ou (VerbForm, Inf).
            distancia:  Quantidade de tokens depois de 'token'. 
                        Se 'max' (ou qualquer string), estende até o fim da sentença, conforme contexto.

        '''
        return self.hasMorphInContext(token, keyvalue, distancia, 'depois')

    def hasMorphInBetween(self, token1: Token, token2: Token, keyvalue: tuple):
        '''
        Verifica se existe na análise morfológica o 'key' de valor 'value' entre os dois tokens.

        Args:
            token: objeto Token do spaCy.
            key: chave que representa a morfologia que deseja o valor, ex: Tense, VerbForm
            value: valor da key, ex: Fut, Inf
        '''
        ids = self.__idsInBetween(token1, token2)
        if not ids:
            return False

        return self.__hasMorph(keyvalue, ids)

    def hasPastParticipleInContext(self, token: Token, distancia = 5, contexto: Literal['antes', 'depois'] = None):
        '''
        Verifica se há particípio passado a uma distância de até 5 tokens, observando o contexto (antes ou depois do token).
        '''
        ids = self.__idsContexto(token, distancia, contexto)
        if not ids:
            return False
        return self.__hasMorph(('VerbForm', 'Part'), ids)

    def hasPastParticipleInContextPrecede(self, token: Token, distancia = 5):
        '''
        Verifica se há particípio passado a uma distância de até 5 tokens antes do 'token'.
        '''
        return self.hasPastParticipleInContext(token, distancia, 'antes')

    def hasPastParticipleInContextFollow(self, token: Token, distancia = 5):
        '''
        Verifica se há particípio passado a uma distância de até 5 tokens depois do 'token'.
        '''
        return self.hasPastParticipleInContext(token, distancia, 'depois')


    #-----CLASSE EVENT-----------
    def classe(self, token: Token, classe: list) -> bool:
        '''
        Verifica se 'token' é uma das classes de eventos da lista 'classe'.

        Args:
            token: Token
            classe: lista de classes do evento
        '''
        ids = self.__tratar_id_token_unico(token)
        return self.__hasClasse(classe, ids)
    

    def __hasClasse(self, classe: list, ids: list) -> bool:
        '''
        Verifica se há Classe de Evento na lista de ids.

        Args:
            classe: Classe do Evento
            ids: lista de token.i

        '''
        classe = self.str_to_list(classe)
        list_classe = self.__tb.siglas.get_list('classe')
        for c in classe:
            if c not in list_classe:
                print("Classe: '{0}' inválida.\nClasses válidas: {1} ".format(c, list_classe))
                return False

        docs_ids = self.__doc_ids(ids)
        
        for doc in docs_ids:
            for token_atual in doc:
                if token_atual._.classe.upper() in classe:
                    return True
        return False

    #-----TIPO TIMEX-----------
    def tipo(self, token: Token, tipo: list) -> bool:
        '''
        Verifica se 'token' é uma dos tipos de timex3 da lista 'tipo'.

        Args:
            token: Token
            tipo: lista de tipos de timex
        '''
        ids = self.__tratar_id_token_unico(token)
        return self.__hasTipo(tipo, ids)
    
    def __hasTipo(self, tipo: list, ids: list) -> bool:
        '''
        Verifica se há Tipo de Timex3 na lista de ids.

        Args:
            tipo: Tipos de Timex3 
            ids: lista de token.i
        '''
        tipo = self.str_to_list(tipo)
        list_tipo = self.__tb.siglas.get_list('tipo')
        for c in tipo:
            if c not in list_tipo:
                print("Tipo: '{0}' inválida.\nTipos válidos: {1} ".format(c, list_tipo))
                return False

        docs_ids = self.__doc_ids(ids)
        
        for doc in docs_ids:
            for token_atual in doc:
                if token_atual._.tipo.upper() in tipo:
                    return True
        return False
    
    
    #--------------------
    def contextBy(self, token: Token, tipo: Literal["str", "str_lemma", "token", "digito", "pos", "dep", "morph"], valor = None, distancia = 'max', contexto: Literal["antes", "depois"] = None) -> bool:
        '''
        Procura elementos na sentença conforme o tipo e a partir do 'token' na direção do contexto.
        '''

        #VALIDA TOKEN
        if type(token) != Token:
            print("ERROR: 'token' não é do tipo Token.")
            return False

        #VALIDA CONTEXTO
        if contexto:
            if contexto not in ['antes', 'depois']:
                print("ERROR: 'contexto' deve ser 'antes' ou 'depois'.")
                return False

        #VALIDA TIPO
        #Permite que o parametro 'tipo' seja omitido se for um Token (spaCy), neste caso, os argumentos são alterados para manter a ordem original
        if type(tipo) == Token:
            if type(valor) == int:
                distancia = valor
            valor = tipo
            tipo = 'token'

        if type(tipo) not in [str, list]:
            print("ERROR: Para as funções 'followedBy' e 'precededBy', é necessário informar o tipo correto do elemento que deseja pesquisar.")
            return False

        if type(tipo) == str: 
            tipo = tipo.lower()

        tipos_validos = ['str', 'str_lemma', 'token', 'digito', 'pos', 'dep', 'morph']
        if tipo not in tipos_validos:
            if type(tipo) in [str, list]:
                if type(valor) == int:
                    distancia = valor
                elif valor:
                    print("ERROR: Tipo '{0}' inválido.\nTipos Válidos: {1}".format(tipo, tipos_validos))
                    return False
                valor = tipo
                tipo = 'str'
            else:
                print("ERROR: Tipo '{0}' inválido.\nTipos Válidos: {1}".format(tipo, tipos_validos))
                return False

        #TIPO STR
        if ((tipo == 'str') or (tipo == 'string')) and valor:
            palavras = valor
            return self.hasWordInContext(token, palavras, distancia, contexto)

        #TIPO STR_LEMMA
        if tipo == 'str_lemma' and valor:
            palavras = valor
            return self.hasWordInContext(token, palavras, distancia, contexto, lemma=True)

        #TIPO TOKEN
        if tipo == 'token' and valor:
            if type(valor) != Token:
                print("ERROR: 'valor' não é do tipo Token.")
                return False

            span_list = self.spanContext(token, distancia, contexto)
            outro_token = valor
            
            #verifica se o índice de 'outro_token' está entre o inicio e fim de 'span'
            for span in span_list:
                if span.start <= outro_token.i < span.end:   # final é exclusivo '<'
                    return True

        #TIPO DIGITO
        if tipo == 'digito': 
            if type(valor) == int:
                distancia = valor
            valor = None

            ids = self.__idsContexto(token, distancia, contexto)
            docs_ids = self.__doc_ids(ids)
            
            for doc in docs_ids:
                for token_atual in doc:
                    if token_atual.is_digit or token_atual.pos_ == 'NUM':
                        return True
            return False

        #TIPO POS
        if tipo == 'pos' and valor:
            pos = valor
            return self.hasPosInContext(token, pos, distancia, contexto)

        #TIPO DEP
        if tipo == 'dep' and valor:
            dep = valor
            return self.hasDepInContext(token, dep, distancia, contexto)

        #TIPO MORPH
        if tipo == 'morph' and valor:
            keyvalue = valor
            return self.hasMorphInContext(token, keyvalue, distancia, contexto)

        if not valor:
            print("ERROR: Verificar argumento 'valor'.")
            
        return False

    def precededBy(self, token: Token, tipo: Literal["str", "str_lemma", "token", "digito", "pos", "dep", "morph"], valor = None, distancia = 'max') -> bool:
        '''
        Procura elementos na sentença antes do 'token', conforme o 'tipo'.

        Args:
            token: objeto Token do spaCy.
            tipo: string e pode ser:
                str     -> verifica se existe palavras especificada em valor 
                        -> valor: str ou list; PODE SER OMITIDO.
                str_lemma -> verifica se existe palavras lematizadas especificada em valor 
                        -> valor: str ou list;
                token   -> verifica se 'token' precede o outro token especificado em valor 
                        -> valor: Token; PODE SER OMITIDO.
                digito  -> verifica se há dígitos ou pos = 'NUM' -> Não tem valor;
                        -> Se valor for informado, ele será a distância.
                pos     -> verifica se há a classe gramatical especificada em valor 
                        -> valor: list, ex: ['VERB', 'NOUM'];
                dep     -> verifica se há na árvore de dependência o elemento especificado em valor 
                        -> valor: list, ex: ['nsubj', 'nmod'];
                morph   -> verifica se há na análise morfológica o elemento especificado em valor. 
                        -> valor: tuple (key, value), ex: ('Tense', 'Fut'). Só é permitido uma tupla.
            valor: valor do elemento que será procurado, conforme o tipo.
            distancia:  Se inteiro, é quantidade de tokens antes de 'token' onde a pesquisa será realizada.
                        Se string, a pesquisa será realizada em todos os tokens que precedem 'token'.
        '''
        return self.contextBy(token, tipo, valor, distancia, contexto = 'antes')

    def followedBy(self, token: Token, tipo: Literal["str", "str_lemma", "token", "digito", "pos", "dep", "morph"], valor = None, distancia = 'max') -> bool:
        '''
        Procura elementos na sentença depois do 'token', conforme o 'tipo'.

        Args:
            token: objeto Token do spaCy.
            tipo: string e pode ser:
                str     -> verifica se existe palavras especificada em valor 
                        -> valor: str ou list; PODE SER OMITIDO.
                str_lemma -> verifica se existe palavras lematizadas especificada em valor 
                        -> valor: str ou list;
                token   -> verifica se 'token' vem depois do outro token especificado em valor 
                        -> valor: Token; PODE SER OMITIDO.
                digito  -> verifica se há dígitos ou pos = 'NUM' -> Não tem valor;
                        -> Se valor for informado, ele será a distância.
                pos     -> verifica se há a classe gramatical especificada em valor 
                        -> valor: list, ex: ['VERB', 'NOUM'];
                dep     -> verifica se há na árvore de dependência o elemento especificado em valor 
                        -> valor: list, ex: ['nsubj', 'nmod'];
                morph   -> verifica se há na análise morfológica o elemento especificado em valor 
                        -> valor: tuple (key, value), ex: ('Tense', 'Fut'). Só é permitido uma tupla.
            valor: valor do elemento que será procurado, conforme o tipo.
            distancia:  Se inteiro, é quantidade de tokens depois de 'token' onde a pesquisa será realizada.
                        Se string, a pesquisa será realizada em todos os tokens que vem depois 'token'.
        '''
        return self.contextBy(token, tipo, valor, distancia, contexto = 'depois')

    def spanBetween(self, token1, token2) -> list:
        '''
        Retorna pedaço da sentença com a quantidade de token 'distancia', conforme contexto (antes ou depois de 'token')

        Return: Lista de Span
        '''
        ids = self.__idsInBetween(token1, token2)
        docs_ids = self.__doc_ids(ids)
        
        return docs_ids
    
    
    def spanContext(self, token, distancia = 5, contexto = None) -> list:
        '''
        Retorna pedaço da sentença com a quantidade de token 'distancia', conforme contexto (antes ou depois de 'token')

        Return: Lista de Span
        '''
        ids = self.__idsContexto(token, distancia, contexto)
        docs_ids = self.__doc_ids(ids)
        
        return docs_ids

    
    
    def spanPrecede(self, token, distancia = 5) -> Span:
        '''
        Retorna pedaço da sentença do tamanho da 'distancia' antes de 'token'.

        Args:
            token: Objeto Token.
            distancia:  Se inteiro, é quantidade de tokens antes de 'token'.
                        Se string, a pesquisa será realizada em todos os tokens que vem antes de 'token'.

        Return: Span
        '''
        lista_spans = self.spanContext(token, distancia, 'antes')
        
        if type(lista_spans) == list:
            if len(lista_spans) == 1:
                return lista_spans[0]
            
        return ''

    def spanFollow(self, token, distancia = 5) -> Span:
        '''
        Retorna pedaço da sentença do tamanho da 'distancia' depois de 'token'.

        Args:
            token: Objeto Token.
            distancia:  Se inteiro, é quantidade de tokens depois de 'token'.
                        Se string, a pesquisa será realizada em todos os tokens que vem depois de 'token'.

        Return: Span
        '''
        lista_spans = self.spanContext(token, distancia, 'depois')
        
        if type(lista_spans) == list:
            if len(lista_spans) == 1:
                return lista_spans[0]
            
        return ''

    def __intersection(self, ids1: list, ids2: list) -> bool:
        '''
        Verifica se há interseção entre ids1 e ids2.

        Args:
            ids: lista de ids representados por token.i
        '''
        for i in ids1:
            if i in ids2:
                return True
        return False

    def spanSomeMatch(self, span1: Span, span2: Span) -> bool:
        '''
        Verifica se há interseção entre dois pedaços da sentenças.

        Args:
            span1: parte 1 da sentença.
            span2: parte 2 da sentenca.

        '''
        i_min1 = span1.start
        i_max1 = span1.end
        i_min2 = span2.start
        i_max2 = span2.end
        ids1 = list(range(i_min1, i_max1 + 1))
        ids2 = list(range(i_min2, i_max2 + 1))

        return self.__intersection(ids1, ids2)

    def identicalHead(self, token1, token2) -> bool:
        '''
        Verifica se ambos os tokens possuem a mesma palavra principal (head).
        '''
        return token1.head == token2.head

    def governVerb(self, token) -> Token:
        '''
        Retorna o verbo pai mais próximo na hierarquia da árvore sintática.
        '''
        if not list(token.ancestors):
            return token

        for pai in token.ancestors:
            if pai.pos_ == 'VERB':
                return pai

            
    #----- CONHECIMENTO DE MUNDO ----------------------------
    def temporal_direction(self, word_pt: str) -> str:
        '''
        Retorna o tipo de relação temporal mais provável para o evento composto por palavras presente no arquivo temporal_direction.txt
        Retirado do Apêndice III - LX-TimeAnalyzer
        
        Args:
            word_pt: Texto do evento em português
        '''
        path_tml = self.__tb.path_tml
        path_temporal_direction = path_tml[0:path_tml.find('TimeBankPT')] + 'temporal_direction.txt'
        df_temporal_direction = pd.read_csv(path_temporal_direction, sep=';')

        word_pt_lemma = self.__lemma(word_pt)

        df_relType = df_temporal_direction.query('word_pt in [@word_pt, @word_pt_lemma]')
        
        if df_relType.empty:
            return ''

        return df_relType['relType'].tolist()[0].lower()

    #------- TEMPORAL SIGNAL ----------------------------------
    def list_temporal_signal(self, lemma = False) -> list:
        '''
        Retorna lista contendo os sinais temporais
        '''
        path_tml = self.__tb.path_tml
        path_temporal_signal = path_tml[0:path_tml.find('TimeBankPT')] + 'temporal_signal.txt'
        with open(path_temporal_signal, 'r', encoding='utf-8') as file:
            word_list = [line.strip() for line in file]
        
        if lemma:
            return self.__lemma(word_list)
            
        return word_list

    #-------- MODAL VERBS -------------------------------------
    def list_modal_verbs(self) -> list:
        '''
        Retorna lista contendo os principais verbos modais em português
        '''
        path_tml = self.__tb.path_tml
        path_temporal_signal = path_tml[0:path_tml.find('TimeBankPT')] + 'modal_verbs.txt'
        with open(path_temporal_signal, 'r', encoding='utf-8') as file:
            word_list = [line.strip() for line in file]
        
        return word_list
    
    
    #------ ASPECT ---------------------------------------
    def aspect_progressive(self, token: Token) -> bool:
        '''
        Verifica se o aspecto verbal de token é progressivo.
        Essa aspecto é formado por o verbo estar conjugado + o gerúndio do verbo principal (token)
        
        Args:
            token: token que representa o verbo ou o EVENT
            
        '''
        estar_ok = False
        i_estar = 100 # índice da posição de 'estar'
        if self.hasWordInContextPrecede(token, 'estar', distancia = 6, lemma = True): #verifica se há 'estar' e suas variações em até 6 tokens antes de 'token'
            for s in self.spanPrecede(token, 6):  #cada um dos 6 tokens que precedem 'token'
                if s.lemma_.lower() in ['estar', 'estamos', 'estão']: 
                    i_estar = s.i
                    if s.pos_ in ['VERB', 'AUX']:
                        estar_ok = True
                #se houver adjetivo ou verbo após o lemma 'estar' e ante de 'token', deixa de ser progressivo
                if s.i > i_estar:
                    if s.pos_ in ['ADJ', 'VERB']:
                        estar_ok = False

            #português brasil
            if self.verbGerundio(token):
                return estar_ok
            else:  #português Portugal
                return estar_ok and self.hasWordInContextPrecede(token, 'a', distancia = 2) and self.pos(token, 'VERB') and ( self.morph(token, ('VerbForm', 'Inf')) or self.morph(token, ('Voice', 'Pass')) )

        return False

    
    
    #===============================================================================
    # -----FUNÇÕES PARA AS FEATURES DO DATASET USADO PARA GERAR REGRAS -------------
    #-------------------------------------------------------------------------------
    
    #---------------------------------------------------
    # ------ AUXILIARES 
    
    def distance_tokens(self, token1: Token, token2: Token):
        i1 = np.inf
        i2 = np.inf
        
        if token1 != None:
            i1 = token1.i
        
        if token2 != None:
            i2 = token2.i
        
        if (i1 == np.inf) and (i2 == np.inf):
            return np.inf
        
        return abs(i1 - i2 - 1)

    def closest_to_token(self, tokenPrecede: Token, token: Token, tokenFollow: Token):
        '''
        Retorna o token mais mais próximo de 'token'.
        Ou tokenPrecede ou tokenFollow.
        '''
        distance_precede = self.distance_tokens(token, tokenPrecede)
        distance_follow = self.distance_tokens(tokenFollow, token)
    
        if (distance_precede == np.inf) and (distance_follow == np.inf):
            return None
        
        if distance_precede <= distance_follow:
            return tokenPrecede
        else:
            return tokenFollow
        
    def __resource_closest_to_token_helper(self, token: Token, context: Literal["antes", "depois"], type_resource: Literal["ENTITY", "POS"], value_resource, distancia = 'Max') -> Token:
        '''
        Retorna o token do evento mais próximo, conforme contexto, do evento da relação.
        '''
        context = context.lower()
        type_resource = type_resource.upper()
        
        if context == 'antes':
            span = reversed(self.spanPrecede(token, distancia))
        else:
            span = self.spanFollow(token, distancia)

        for t in span:
            if type_resource == 'ENTITY':
                if self.is_entity(t, value_resource):
                    return t
            if type_resource == 'POS':
                if self.pos(t, value_resource):
                    return t
                
    
    def __entidade_conjunction_closest(self, token: Token, contexto: Literal["antes", "depois"], distancia = 'Max'):
        '''
        Conjunção mais próxima conforme contexto do token.
        '''
        result = self.__resource_closest_to_token_helper(token, contexto, 'POS', ['CCONJ', 'SCONJ'], distancia)
        if not result:
            return 'NONE'
        
        return result.text.lower()
    
    def __entidade_preposition_closest(self, token: Token, contexto: Literal["antes", "depois"], distancia = 'Max'):
        '''
        Preposição mais próxima conforme contexto do token.
        '''
        result = self.__resource_closest_to_token_helper(token, contexto, 'POS', 'ADP', distancia)
        if not result:
            return 'NONE'
        
        return result.text.lower()
    
    def event_closest_to_token_precede(self, token: Token) -> Token:
        '''
        Retorna o token do evento mais próximo à esquerda do evento da relação.
        '''
        return self.__resource_closest_to_token_helper(token, 'antes', 'ENTITY', 'EVENT')
    
    
    def event_closest_to_token_follow(self, token: Token) -> Token:
        '''
        Retorna o token do evento mais próximo à direita do evento da relação.
        '''
        return self.__resource_closest_to_token_helper(token, 'depois', 'ENTITY', 'EVENT')
            
    
    def event_closest_to_token(self, token: Token) -> Token:
        '''
        Retorna o token do evento mais próximo (da esquerda ou da direita) do evento da relação.
        '''
        event_precede = self.event_closest_to_token_precede(token)
        event_follow = self.event_closest_to_token_follow(token) 
        
        return self.closest_to_token(event_precede, token, event_follow)
    
    def event_closest_to_token_resource(self, token: Token, resource: Literal["class", "pos", "tense", "temporal_direction"]):
        '''
        Recurso do EVENT mais próximo do EVENT do par da relação 

        '''
        resource = resource.lower()
        event_closest = self.event_closest_to_token(token)
        if not event_closest:
            return 'NONE'
        
        if resource == 'class':
            return self.event_class(event_closest)
        elif resource == 'pos':
            return self.event_pos(event_closest)
        elif resource == 'tense':
            return self.event_tense(event_closest)
        elif resource == 'temporal_direction':
            return self.event_temporal_direction(event_closest)
    
    def token_resource(self, token: Token, resource: Literal["class", "pos", "tense", "polarity", "aspect", "dep", "type", "temporalfunction"]):
        resource = resource.lower()
        
        if resource == 'class':
            return self.__trata_vazio(token._.classe)
        elif resource == 'pos':
            return self.__trata_vazio(token.pos_)
        elif resource == 'tense':
            tense_atual = list(map(str.upper, token.morph.get('Tense')))
            if tense_atual:
                tense_atual = tense_atual[0]
            else:
                tense_atual = 'NONE'
            return tense_atual
        elif resource == 'polarity':
            return self.__trata_vazio(token._.polarity)
        elif resource == 'aspect':
            #return self.__trata_vazio(token._.aspecto) # -> não vamos utiliza o aspecto do corpus, minha implementação está melhor
            if self.aspect_progressive(token):
                return 'PROGRESSIVE'
            else:
                return 'NONE'
        elif resource == 'dep':
            return self.__trata_vazio(token.dep_.upper())
        elif resource == 'type':
            return self.__trata_vazio(token._.tipo)
        elif resource == 'temporalfunction':
            return self.__trata_vazio(token._.temporalFunction)

    
    def __trata_vazio(self, obj):
        if obj == '':
            return 'NONE'
        else:
            return obj
        
    def is_equal(self, valor1, valor2) -> bool:
        '''
        Verifica se dois valores são iguais.
        Se ambos foram 'NONE', não são iguais
        '''
        valor1 = valor1.upper()
        valor2 = valor2.upper()
        
        if (valor1 == 'NONE') or (valor2 == 'NONE'):
            return False
        
        return valor1 == valor2
    
    def is_entity(self, token: Token, entidade: Literal["EVENT", "TIMEX"]) -> bool:
        '''
        Verifica de o token é Event ou Timex conforme valor de 'entidade'
        '''
        entidade = entidade.upper()
        
        if token.ent_type_ == entidade:
            return True
        else:
            return False
    
    def nbor(self, token: Token, n: int) -> Token:
        min = -token.i
        max = len(token.doc) - token.i - 1
        n = 0 if n < min else n
        n = 0 if n > max else n
        return None if n == 0 else token.nbor(n)

    def __entidade_pos_token_i(self, token, i):
        
        result = self.nbor(token, i)
        if not result:
            return 'NONE'
        
        return result.pos_
    
    def __distance_category(self, distance: int) -> str:
        '''
        Escala: “perto” até 4 tokens, “distancia_media”: 5 a 9, “longe”: 10 a 14 e "muito_longe": 14+ 
        '''
        if distance <= 4:
            return "perto"
        if distance <= 9:
            return "distancia_media"
        if distance <= 14:
            return "longe"
        else:
            return "muito_longe"
            
    
    def __is_expression(self, texto: str) -> bool:
        '''
        Retorna True se o texto for uma expressão (composto por mais de uma palavra).
        Retorna False se for apenas uma palavra
        '''
        #se tiver só uma palavra
        if len(texto.strip().split()) == 1:
            return False
        #se for expressão
        else:
            return True

    def lista_palavras(self, texto: str) -> list:
        '''
        Converte texto em uma lista de palavras
        '''
        return texto.strip().split()

    def list_combination(self, lista: list) -> list:
        '''
        Retorna lista com a combinação r = de 2 a len(lista) 
        '''
        combs = []
        for r in range(2, len(lista) + 1):
            combs.extend(combinations(lista, r))
        return [' '.join(c) for c in combs]

    def __grater_consecutive_intersection(self, phrase1: str, phrase2: str) -> str:
        '''
        Retorna maior intersecção consecutiva entre duas frases
        '''
        words1 = phrase1.split()
        words2 = phrase2.split()

        # Define a menor e maior frase em termos de número de palavras
        if len(words1) < len(words2):
            shorter_words, longer_words = words1, words2
        else:
            shorter_words, longer_words = words2, words1

        #lista de palavras iguais entre as duas frases
        equals_words = []
        for shorter_word in shorter_words:
            for longer_word in longer_words:
                if shorter_word == longer_word:
                    equals_words.append(shorter_word)

        #lista de sequencias com n palavras comuns nas duas frases
        list_exprs = self.list_combination(equals_words)

        #retorna a maior sequencia que for encontrada na frase
        longer_phrase = ' '.join(longer_words).strip()
        for expr in reversed(list_exprs):
            if expr in longer_phrase:
                return expr
            

    def temporal_signal_interssection_token(self, token: Token) -> str:
        '''
        Retorna a intersecção de palavras consecutivas do token que estão na lista de temporal signals
        '''
        #lemma do token
        T = token.lemma_.lower()
        #já deixamos a lista de sinais com os lemas
        signals = self.signals
        
        #se T casar com item de signals
        if T in signals:
            return T

        #se T for uma expressão
        if self.__is_expression(T):
            #cada palavraT
            for palavraT in self.lista_palavras(T):
                #cada signal de signals
                for signal in signals:
                    signal = signal.lower()
                    #se signal for expressão
                    if self.__is_expression(signal):
                        for palavraS in self.lista_palavras(signal):
                            #se encontrar alguma palavra de T que seja igual a alguma palavra de signal atual
                            if palavraT == palavraS:
                                #retorna a maior intersecção consecutiva entre T e signal atual
                                return self.__grater_consecutive_intersection(T, signal)
                    #se sinal for uma palavra
                    else:
                        #retorna o signal atual se for igual a uma das palavras de T
                        if palavraT == signal:
                            return signal
        return 'NONE'
    
    def __token_gov_verb(self, token: Token) -> Token:
        '''
        Se 'token' não for verbo, retorna o verbo governante (event.ancestor) com base em sua relação de dependência. 
        Se 'token' for verbo, retorna o próprio token
        '''
        if self.pos(token, 'VERB'):
            return token
        else:
            verb_gov = self.governVerb(token) 
            if not verb_gov:
                return 
            if verb_gov != token:
                return verb_gov
        return

    def __token_gov_verb_tense(self, token: Token):
        '''
        Tense de token: Se token não for verbo, o tempo verbal é estimado pelo seu verbo governante (event.ancestor) com base em sua relação de dependência. 
        Se token é verbo, o valor é do próprio token. 
        '''
        verb_gov = self.__token_gov_verb(token)
        if verb_gov:
            verb_tense = self.token_resource(verb_gov, 'tense')
            return verb_tense.upper()
        
        return 'NONE'
    
    
    def ancestors_between_filho_e_pai(self, Pai: Token, Filho: Token) -> List[Token]:
        '''
        Retorna lista de tokens contendo todos os ancestrais entre Filho e Pai. 
        O primeiro elemento da lista é o 'Filho' e o Último é o anterior ao 'Pai' (i.e. exclui o Pai).
        '''
        paisF = Filho.ancestors
        lista_paisF = []
        lista_paisF.append(Filho)
        for paiF in paisF:
            if paiF == Pai:
                return lista_paisF
            lista_paisF.append(paiF)
        return

    def has_dep_list_token(self, list_tokens: List[Token], dep: list) -> bool:
        '''
        Verifica de há a dependência sintática 'dep' na lista de tokens
        '''
        for token in list_tokens:
            if self.dep(token, dep):
                return True
        return False
    
    def event_is_pai_direto_timex3(self, E: Token, T: Token) -> bool:
        '''
        Verifica se EVENT é pai direto de TIMEX
        '''
        return E.i == T.head.i

    def timex3_is_pai_direto_event(self, E: Token, T: Token) -> bool:
        '''
        Verifica se TIMEX é pai direto de EVENT
        '''
        return E.head.i == T.i
    
    
    #----------------------------------------------
    # >>>>>>> AUXILIARES DE SIGNALS

    def signal_closest_span(self, span: Span) -> Token:
        '''
        Retorna o primeiro token que representa o sinal temporal, portanto, o mais próximo conforme contexto
        '''
        if not span:
            return

        signals = self.signals
        
        for token in span:
            if token.text in signals:
                return token
        return

    def signal_span_context(self, token: Token, context: Literal["antes", "depois"]) -> Span:
        '''
        Retorna Span entre 'token' e a próxima entity (Event ou Timex), conforme o contexto.
        '''
        context = context.lower()
        if context == 'antes':
            span = reversed(self.spanPrecede(token, 'Max'))
        else:
            span = self.spanFollow(token, 'Max')

        if not span:
            return 

        for s in span:
            if s.ent_type_ in ['EVENT', 'TIMEX3']:
                i = s.i
                if context == 'antes':
                    return token.doc[i+1:token.i]
                else:
                    return token.doc[token.i+1:i]

        if context == 'antes':
            return token.doc[:token.i]
        else:
            return token.doc[token.i+1:]

        
    def signal_context_token(self, token: Token, context: Literal["antes", "depois"]) -> Token:
        '''
        Retorna o sinal temporal (Token) mais próximo do contexto de 'token'.
        '''
        context = context.lower()
        span_context = self.signal_span_context(token, context)
        return self.signal_closest_span(span_context)

    
    def signal_has_comma_token(self, token: Token, signal: Token) -> bool:
        '''
        Verifica de há vírgula entre o 'token' e o sinal temporal
        '''
        
        if not signal:
            return
        
        span_context = self.spanBetween(token, signal)
        if not span_context:
            return False

        for token in span_context:
            if token.text == ",":
                return True
        return False
        
        
    def token1_token2_dep(self, t1: Token, t2: Token) -> str:
        '''
        DEP entre token1/token2 e token2/token1, se houver
        '''
        # E é filho direto de T
        if self.timex3_is_pai_direto_event(t1, t2):
            return self.token_resource(t1, 'dep')

        # T é filho direto de E
        if self.event_is_pai_direto_timex3(t1, t2):
            return self.token_resource(t2, 'dep')
        
        return 'NONE'
    
    
    def __signal_ancestor_token(self, token: Token, signal: Token) -> bool:
        '''
        Sinal domina sintaticamente o token?
        '''
        if not signal:
            return
        return signal.is_ancestor(token)

    
    
    def __reichenbach_table(self, tense: str) -> str:
        '''
        Retorna o tempo de Reichenbach de acordo com o tempo verbal em português
        '''
        table = {'FPRESC': 'Anterior', 'FPRETC': 'Anterior', 'PRETPC': 'Anterior', 'PRETMQPC': 'Anterior',
                 'FUT': 'Posterior', 'PRES': 'Simples', 'CND': 'Posterior', 'PAST': 'Simples', 'IMP':'Simples', 'PQP': 'Anterior'
                }
        
        tense = tense.upper()
        tense_validos = list(map(str.upper, table.keys()))
        if tense == '':
            return
        if tense not in tense_validos:
            print(f'ERROR: {tense} não é um tempo válido. Válidos {tense_validos}')
            return 
        
        return table[tense].lower()
    
    
    #----------------------------------------------------------------
    # ---- FUNÇÕES COM MESMOs NOMES DAS FEATURES
    #----------------------------------------------------------------

    def event_class(self, E: Token) -> str:
        '''
        Class de EVENT
        '''
        return self.token_resource(E, 'class')
        
    def event_pos(self, E: Token) -> str:
        '''
        POS de EVENT
        '''
        return self.token_resource(E, 'pos')
    
    def event_tense(self, E: Token) -> str:
        '''
        Tense de EVENT
        '''
        return self.token_resource(E, 'tense')
    
    def event_polarity(self, E: Token) -> str:
        '''
        Polaridade de EVENT
        '''
        return self.token_resource(E, 'polarity')
    
    def event_aspect(self, E: Token):
        '''
        Aspecto de EVENT
        '''
        return self.token_resource(E, 'aspect')
    
    def event_dep(self, E: Token):
        '''
        DEP de Event com seu pai
        '''
        return self.token_resource(E, 'dep')
    
    def timex3_dep(self, T: Token):
        '''
        Relação final: DEP de TIMEX com seu pai
        '''
        return self.token_resource(T, 'dep')
    
    def timex3_type(self, T: Token):
        '''
        Tipo de TIMEX
        '''
        return self.token_resource(T, 'type')
    
    def timex3_pos(self, T: Token):
        '''
        POS de TIMEX
        '''
        return self.token_resource(T, 'pos')
    
    def timex3_temporalfunction(self, T: Token):
        '''
        temporalFunction de TIMEX (bool)
        '''
        return self.token_resource(T, 'temporalfunction')
    
    def event_closest_to_event_class(self, E: Token):
        '''
        Class do EVENT mais próximo do EVENT do par da relação 

        '''
        return self.event_closest_to_token_resource(E, 'class')
    
    def event_closest_to_event_pos(self, E:Token):
        '''
        POS de EVENT mais próximo do EVENT da relação em consideração
        '''
        return self.event_closest_to_token_resource(E, 'pos')
    
    def event_closest_to_event_tense(self, E:Token):
        '''
        Tense de EVENT mais próximo do EVENT da relação em consideração
        '''
        return self.event_closest_to_token_resource(E, 'tense')

    def event_closest_to_event_equal_lemma(self, E: Token):
        '''
        LEMMA de EVENT da relação em consideração == LEMMA de EVENT mais próximo a ele
        '''
        event_closest = self.event_closest_to_token(E)
        if not event_closest:
            return False
        return E.lemma_ == event_closest.lemma_
    
    def event_closest_to_event_equal_class(self, E: Token):
        '''
        Class do EVENT da relação em consideração == CLASS do EVENT mais próximo a ele
        '''
        return self.is_equal(self.event_class(E), self.event_closest_to_event_class(E))
    
    def event_closest_to_event_equal_pos(self, E: Token):
        '''
        POS de EVENT da relação em consideração == POS de EVENT mais próximo a ele
        '''
        return self.is_equal(self.event_pos(E), self.event_closest_to_event_pos(E))
    
    def event_closest_to_event_equal_tense(self, E: Token):
        '''
        Tense do EVENT da relação em consideração == Tense do EVENT mais próximo a ele
        '''
        return self.is_equal(self.event_tense(E), self.event_closest_to_event_tense(E))
    
    def event_closest_to_timex3_pos(self, T: Token):
        '''
        POS do EVENT mais próximo do TIMEX da relação temporal em consideração
        '''
        return self.event_closest_to_token_resource(T, 'pos')
    
    def event_closest_to_timex3_equal_pos(self, E: Token, T: Token) -> bool:
        '''
        POS de EVENT da relação em consideração == POS de EVENT mais próximo do TIMEX da relação em consideração
        '''
        return self.is_equal(self.event_pos(E), self.event_closest_to_timex3_pos(T))

    def event_conjunction_closest_follow(self, E: Token) -> str:
        '''
        Conjunção mais próxima após o evento da relação processada.
        '''
        return self.__entidade_conjunction_closest(E, 'depois', 5)
    
    def event_conjunction_closest_precede(self, E: Token) -> str:
        '''
        Conjunção mais próxima antes do evento da relação processada.
        '''
        return self.__entidade_conjunction_closest(E, 'antes', 5)
    
    def event_root(self, E: Token) -> bool:
        '''
        EVENT é a raiz da sentença? (bool)
        '''
        return self.dep(E, 'ROOT')
        
    
    def timex3_root(self, T: Token) -> bool:
        '''
        TIMEX é a raiz da sentença? (bool)
        '''
        return self.dep(T, 'ROOT')

    def event_pos_token_1_follow(self, E: Token):
        '''POS do 1º token após o EVENT        '''
        return self.__entidade_pos_token_i(E, 1)
    
    def event_pos_token_1_precede(self, E: Token):
        '''POS do 1º token antes o EVENT        '''
        return self.__entidade_pos_token_i(E, -1)
    
    def event_pos_token_2_follow(self, E: Token):
        ''' POS do 2º token depois o EVENT        '''
        return self.__entidade_pos_token_i(E, 2)
    
    def event_pos_token_2_precede(self, E: Token):
        ''' POS do 2º token antes o EVENT        '''
        return self.__entidade_pos_token_i(E, -2)
    
    def event_pos_token_3_follow(self, E: Token):
        ''' POS do 3º token depois o EVENT         '''
        return self.__entidade_pos_token_i(E, 3)
    
    def event_pos_token_3_precede(self, E: Token):
        ''' POS do 3º token antes o EVENT        '''
        return self.__entidade_pos_token_i(E, -3)
    
    def timex3_pos_token_1_follow(self, T: Token):
        ''' POS do 1º token depois o TIMEX        '''
        return self.__entidade_pos_token_i(T, 1)
    
    def timex3_pos_token_1_precede(self, T: Token):
        ''' POS do 1º token antes o TIMEX        '''
        return self.__entidade_pos_token_i(T, -1)
    
    def timex3_pos_token_2_follow(self, T: Token):
        ''' POS do 2º token depois o TIMEX         '''
        return self.__entidade_pos_token_i(T, 2)
    
    def timex3_pos_token_2_precede(self, T: Token):
        ''' POS do 2º token antes o TIMEX         '''
        return self.__entidade_pos_token_i(T, -2)
    
    def timex3_pos_token_3_follow(self, T: Token):
        ''' POS do 3º token depois o TIMEX        '''
        return self.__entidade_pos_token_i(T, 3)
    
    def timex3_pos_token_3_precede(self, T: Token):
        ''' POS do 3º token antes o TIMEX        '''
        return self.__entidade_pos_token_i(T, -3)
    
    def event_preposition_precede(self, E: Token):
        '''Preposições que precedem um EVENT, ou NONE se essa palavra não for uma preposição.'''
        return self.__entidade_preposition_closest(E, 'antes', 5)
    
    def timex3_preposition_precede(self, T: Token):
        '''Preposição antes do TIMEX, ou NONE se essa palavra não for uma preposição.'''
        return self.__entidade_preposition_closest(T, 'antes', 5)
    
    
    def event_timex3_distance(self, E: Token, T: Token):
        '''A distância entre EVENT e TIMEX (número de tokens categorizado)
        Escala: “perto” até 4 tokens, “distancia_media”: 5 a 9, “longe”: 10 a 14 e "muito_longe": 14+ '''
        
        distancia = self.lengthInBetween(E, T)
        return self.__distance_category(distancia)
    
    def event_first_order(self, E: Token, T: Token) -> bool:
        '''Se EVENT precede textualmente TIMEX na relação em consideração'''
        return self.t1BeforeT2(E, T)
    
    def event_between_order(self, E: Token, T: Token) -> bool:
        '''Se há outro EVENT entre EVENT e TIMEX'''
        span = self.spanBetween(E, T)[0]
        for s in span:
            if s.ent_type_ == 'EVENT':
                return True
        return False

    def timex3_between_order(self, E: Token, T: Token) -> bool:
        '''Se há outro TIMEX entre EVENT e TIMEX'''
        span = self.spanBetween(E, T)[0]
        for s in span:
            if s.ent_type_ == 'TIMEX3':
                return True
        return False
    
    def event_timex3_no_between_order(self, E: Token, T: Token) -> bool:
        '''True se não houver EVENT ou TIMEX entre o par EVENT/TIMEX da relação que está sendo processada. 
        (é verdadeiro se e somente se timex3-between-order e event-between-order são falsos)
        '''
        if (self.event_between_order(E, T) == False) and (self.timex3_between_order(E, T) == False):
            return True
        return False

    def timex3_between_quant(self, E: Token, T: Token) -> int:
        '''A quantidade de TIMEX entre o par EVENT e TIMEX'''
        span = self.spanBetween(E, T)[0]
        quant = 0
        for s in span:
            if s.ent_type_ == 'TIMEX3':
                quant += 1
        return quant

    def event_temporal_direction(self, E: Token):
        '''
        Mapeamento manual entre EVENTs e a relação temporal esperada com seu complemento.
        '''
        return self.__trata_vazio(self.temporal_direction(E.lemma_))
    
    def event_closest_to_event_temporal_direction(self, E: Token):
        '''
        Direção temporal de EVENT mais próximo de EVENT da relação temporal em consideração
        '''
        return self.event_closest_to_token_resource(E, 'temporal_direction')


    def timex3_relevant_lemmas(self, T: Token):
        '''
        Se o lemma do TIMEX está contida em LISTA de palavras que têm algum conteúdo temporal. 
        '''
        result = self.temporal_signal_interssection_token(T)
        if not result:
            return 'NONE'
        return result


    def event_gov_verb_aspect(self, E: Token):
        '''
        Aspecto de EVENT não verbal: Se EVENT não for verbo, o aspecto é estimado pelo seu verbo governante (event.ancestor) com base em sua relação de dependência. 
        Se evento é verbo, o valor é do próprio evento
        '''
        verb_gov = self.__token_gov_verb(E)
        if verb_gov:
            #return verb_gov._.aspecto  # -> não vamos usar o aspecto anotado no corpus, minha implementação está melhor
            return self.token_resource(verb_gov, 'aspect')
        return 'NONE'

    def event_gov_verb_tense(self, E: Token):
        '''
        Tense de EVENT não verbal: Se EVENT não for verbo, o tempo verbal é estimado pelo seu verbo governante (event.ancestor) com base em sua relação de dependência. 
        Se evento é verbo, o valor é do próprio evento. (Tense do verbo que rege o EVENT)
        '''
        return self.__token_gov_verb_tense(E)

    def timex3_gov_verb_tense(self, T: Token):
        '''
        Tense do verbo que rege o TIMEX
        '''
        return self.__token_gov_verb_tense(T)
    
    def event_head_pos(self, E: Token):
        '''
        POS do pai de EVENT
        '''
        return E.head.pos_

    def timex3_head_pos(self, T: Token):
        '''
        POS do pai de TIMEX
        '''
        return T.head.pos_

    def event_intervening_following_tense(self, E: Token, T: Token):
        '''
        Tense de EVENT que está entre o EVENT e TIMEX, nesta ordem, da relação em consideração e está mais próximo do TIMEX. 
        Ex: EVENT -------- event.tense -- TIMEX
        '''
        if self.event_first_order(E, T):
            span = self.spanBetween(E, T)[0]
            for s in reversed(span):
                if s.ent_type_ == 'EVENT':
                    return self.token_resource(s, 'tense')
        return 'NONE'

    def event_intervening_preceding_class(self, E: Token, T: Token):
        '''
        Class do EVENT que está entre o TIMEX e EVENT, nesta ordem, da relação em consideração e está mais próximo do TIMEX. 
        Ex: TIMEX -- event.class -------- EVENT
        '''
        if not self.event_first_order(E, T):
            span = self.spanBetween(E, T)[0]
            for s in span:
                if s.ent_type_ == 'EVENT':
                    return self.token_resource(s, 'class')
        return 'NONE'

    
    
    def event_gov_verb(self, E: Token) -> str:
        '''
        Verbo que rege o EVENT (Para eventos que são verbos, essa feature é o próprio evento)
        NÃO USAR? BASEADO EM PALAVRAS
        '''
        verb_gov = self.__token_gov_verb(E)
        if verb_gov:
            verb_text = verb_gov.text
            return verb_text.lower()
        return 'NONE'
    
    def timex3_gov_verb(self, T: Token) -> Token:
        '''
        Verbo que rege o TIMEX
        NÃO USAR? BASEADO EM PALAVRAS
        '''
        verb_gov = self.__token_gov_verb(T)
        if verb_gov:
            verb_text = verb_gov.text
            return verb_text.lower()
        return 'NONE'
    
    
    def event_head_is_root(self, E: Token) -> bool:
        '''
        O Event modifica diretamente a raiz? (ex: Event é um filho direto da raiz?)
        '''
        roots = self.__tb.get_doc_root(E.doc)
        return E.head in roots
    
    def event_is_ancestor_timex3(self, E: Token, T: Token) -> bool:
        '''
        EVENT é a entidade regente na relação?
        '''
        return E.is_ancestor(T)
    
    def event_is_child_timex3(self, E: Token, T: Token) -> bool:
        '''
        EVENT é a entidade dependente na relação?
        '''
        return T.is_ancestor(E)
    
    
    def timex3_head_is_root(self, T: Token) -> bool:
        '''
        O Timex modifica diretamente a raiz? (ex: Timex é um filho direto da raiz?)
        '''
        roots = self.__tb.get_doc_root(T.doc)
        return T.head in roots

    def timex3_is_ancestor_event(self, E: Token, T: Token) -> bool:
        '''
        TIMEX é a entidade regente na relação?
        '''
        return T.is_ancestor(E)

    def timex3_is_child_event(self, E: Token, T: Token) -> bool:
        '''
        TIMEX é a entidade dependente na relação?
        '''
        return E.is_ancestor(T)
    
    
    
    def event_preposition_gov(self, E: Token) -> Token:
        '''
        preposição que rege sintaticamente o EVENT
        '''
        for child in E.children:
            if child.pos_ == 'ADP':
                return child.text.lower()
        return 'NONE'
    
    def timex3_preposition_gov(self, T: Token) -> Token:
        '''
        preposição que rege sintaticamente o TIMEX
        '''
        for child in T.children:
            if child.pos_ == 'ADP':
                return child.text.lower()
        return 'NONE'
    
    
    def reichenbach_direct_modification(self, E: Token, T: Token) -> bool:
        '''
        Modificação direta: O TIMEX modifica diretamente o EVENT? (TIMEX é filho de EVENT?)
        '''
        return self.event_is_pai_direto_timex3(E, T)
    
    
    def reichenbach_temporal_mod_function(self, E: Token, T: Token) -> bool:
        '''
        Função de modificação temporal: Existe uma relação tmod (usaremos nmod) no caminho de dependência do EVENT ao TIMEX? 
        (verificar se nmod ajuda, não há tmod em português)
        '''
        dep = ['nmod']
        if self.timex3_is_ancestor_event(E, T):
            lista_ancestors = self.ancestors_between_filho_e_pai(T, E)
            return self.has_dep_list_token(lista_ancestors, dep)

        if self.event_is_ancestor_timex3(E, T):
            lista_ancestors = self.ancestors_between_filho_e_pai(E, T)
            return self.has_dep_list_token(lista_ancestors, dep)
        
        return False
    
    
    
    def event_timex3_dep(self, E: Token, T: Token) -> str:
        '''
        DEP entre EVENT/TIMEX e TIMEX/EVENT, se houver
        '''
        return self.token1_token2_dep(E, T)
    

    # >>>>>>>>>>> SIGNAL
        
    
    def signal_follow_event_ancestor_event(self, E: Token) -> bool:
        '''
        Sinal que segue event domina sintaticamente o Event?
        '''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return self.__signal_ancestor_token(E, signal)

    def signal_follow_timex3_ancestor_event(self, E: Token, T: Token) -> bool:
        '''Sinal que segue timex domina sintaticamente o Event?'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return self.__signal_ancestor_token(E, signal)

    def signal_precede_event_ancestor_event(self, E: Token) -> bool:
        '''Sinal que precede event domina sintaticamente o Event?'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return self.__signal_ancestor_token(E, signal)
        
    def signal_precede_timex3_ancestor_event(self, E: Token, T: Token) -> bool:
        '''Sinal que precede timex domina sintaticamente o Event?'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return self.__signal_ancestor_token(E, signal)



    def signal_follow_event_ancestor_timex3(self, E: Token, T: Token) -> bool:
        '''
        Sinal que segue event domina sintaticamente o Timex?
        '''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return self.__signal_ancestor_token(T, signal)
    
    def signal_follow_timex3_ancestor_timex3(self, T: Token) -> bool:
        '''Sinal que segue timex domina sintaticamente o Timex?'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return self.__signal_ancestor_token(T, signal)
        
    def signal_precede_event_ancestor_timex3(self, E: Token, T: Token) -> bool:
        '''Sinal que precede event domina sintaticamente o Timex?'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return self.__signal_ancestor_token(T, signal)

    def signal_precede_timex3_ancestor_timex3(self, T: Token) -> bool:
        '''Sinal que precede timex domina sintaticamente o Timex?'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return self.__signal_ancestor_token(T, signal)
        
        

    def signal_follow_event_text(self, E: Token) -> str:
        '''Texto do Sinal que segue event.'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return signal.text.lower()
    
    def signal_precede_event_text(self, E: Token) -> str:
        '''Texto do Sinal que precede event'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return signal.text.lower()

    def signal_precede_timex3_text(self, T: Token) -> str:
        '''Texto do Sinal que precede timex'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return signal.text.lower()

    def signal_follow_timex3_text(self, T: Token) -> str:
        '''Texto do Sinal que segue timex'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return signal.text.lower()


        
    def signal_precede_event_pos(self, E: Token) -> str:
        '''POS do Sinal que precede event'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return signal.pos_
        
    def signal_follow_event_pos(self, E: Token) -> str:
        '''POS do Sinal que segue event.'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return signal.pos_

    def signal_precede_timex3_pos(self, T: Token) -> str:
        '''POS do Sinal que precede timex'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return signal.pos_
        
    def signal_follow_timex3_pos(self, T: Token) -> str:
        '''POS do Sinal que segue timex'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return signal.pos_


    
    def signal_precede_event_distance_event(self, E: Token) -> str:
        '''Distância do Sinal que precede event até Event'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        distancia = self.lengthInBetween(signal, E)
        return self.__distance_category(distancia)

    def signal_follow_event_distance_event(self, E: Token) -> str:
        '''Distância do Sinal que segue event até Event.'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        distancia = self.lengthInBetween(signal, E)
        return self.__distance_category(distancia)
        
    def signal_precede_timex3_distance_event(self, E: Token, T: Token) -> str:
        '''Distância do Sinal que precede timex até Event'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        distancia = self.lengthInBetween(signal, E)
        return self.__distance_category(distancia)

    def signal_follow_timex3_distance_event(self, E: Token, T: Token) -> str:
        '''Distância do Sinal que segue timex até Event'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        distancia = self.lengthInBetween(signal, E)
        return self.__distance_category(distancia)


    def signal_precede_event_distance_timex3(self, E: Token, T: Token) -> str:
        '''Distância do Sinal que precede event até Timex'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        distancia = self.lengthInBetween(signal, T)
        return self.__distance_category(distancia)
        
    def signal_follow_event_distance_timex3(self, E: Token, T: Token) -> str:
        '''Distância do Sinal que segue event até Timex'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        distancia = self.lengthInBetween(signal, T)
        return self.__distance_category(distancia)
    
    def signal_precede_timex3_distance_timex3(self, T: Token) -> str:
        '''Distância do Sinal que precede timex até Timex'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        distancia = self.lengthInBetween(signal, T)
        return self.__distance_category(distancia)
    
    def signal_follow_timex3_distance_timex3(self, T: Token) -> str:
        '''Distância do Sinal que segue timex até Timex'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        distancia = self.lengthInBetween(signal, T)
        return self.__distance_category(distancia)

    

    def signal_precede_event_comma_between_event(self, E: Token) -> bool:
        '''Há uma vírgula entre o Sinal que precede event e o Event?'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return self.signal_has_comma_token(E, signal)
        
    def signal_follow_event_comma_between_event(self, E: Token) -> bool:
        '''
        Há uma vírgula entre o Sinal que segue event e o Event?
        '''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return self.signal_has_comma_token(E, signal)
        
    def signal_precede_timex3_comma_between_event(self, E: Token, T: Token) -> bool:
        '''Há uma vírgula entre o Sinal que precede timex e o Event?'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return self.signal_has_comma_token(E, signal)
        
    def signal_follow_timex3_comma_between_event(self, E: Token, T: Token) -> bool:
        '''Há uma vírgula entre o Sinal que segue timex e o Event?'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return self.signal_has_comma_token(E, signal)


    

    def signal_precede_event_comma_between_timex3(self, E: Token, T: Token) -> bool:
        '''Há uma vírgula entre o Sinal que precede event e o Timex?'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return self.signal_has_comma_token(T, signal)
        
    def signal_follow_event_comma_between_timex3(self, E: Token, T: Token) -> bool:
        '''Há uma vírgula entre o Sinal que segue event e o Timex?'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return self.signal_has_comma_token(T, signal)
        
    def signal_precede_timex3_comma_between_timex3(self, T: Token) -> bool:
        '''Há uma vírgula entre o Sinal que precede timex e o Timex?'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return self.signal_has_comma_token(T, signal)
        
    def signal_follow_timex3_comma_between_timex3(self, T: Token) -> bool:
        '''Há uma vírgula entre o Sinal que segue timex e o Timex?'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return self.signal_has_comma_token(T, signal)
        
    
    
    

    def signal_precede_event_child_event(self, E: Token) -> bool:
        '''Sinal que precede event é um filho do Event?'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return E.is_ancestor(signal)
        
    def signal_follow_event_child_event(self, E: Token) -> bool:
        '''Sinal que segue event é um filho do Event?'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return E.is_ancestor(signal)
        
    def signal_precede_timex3_child_event(self, E: Token, T: Token) -> bool:
        '''Sinal que precede timex é um filho do Event?'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return E.is_ancestor(signal)
        
    def signal_follow_timex3_child_event(self, E: Token, T: Token) -> bool:
        '''Sinal que segue timex é um filho do Event?'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return E.is_ancestor(signal)
    
    
    def signal_precede_event_child_timex3(self, E: Token, T: Token) -> bool:
        '''Sinal que precede event é um filho de Timex?'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return T.is_ancestor(signal)
        
    def signal_follow_event_child_timex3(self, E: Token, T: Token) -> bool:
        '''Sinal que segue event é um filho de Timex?'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return T.is_ancestor(signal)
        
    def signal_precede_timex3_child_timex3(self, T: Token) -> bool:
        '''Sinal que precede timex é um filho de Timex?'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return T.is_ancestor(signal)
        
    def signal_follow_timex3_child_timex3(self, T: Token) -> bool:
        '''Sinal que segue timex é um filho de Timex?'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return T.is_ancestor(signal)
        
    

    
    def signal_precede_event_is_event_head(self, E: Token) -> bool:
        '''Sinal que precede event  é um pai direto do Event?'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return E.head.i == signal.i
            
    def signal_follow_event_is_event_head(self, E: Token) -> bool:
        '''Sinal que segue event  é um pai direto do Event?'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return E.head.i == signal.i
        
    def signal_precede_timex3_is_event_head(self, E: Token, T: Token) -> bool:
        '''Sinal que precede timex  é um pai direto do Event?'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return E.head.i == signal.i
        
    def signal_follow_timex3_is_event_head(self, E: Token, T: Token) -> bool:
        '''Sinal que segue timex  é um pai direto do Event?'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return E.head.i == signal.i
        
        
        
    def signal_precede_event_is_timex3_head(self, E: Token, T: Token) -> bool:
        '''Sinal que precede event é um pai direto do Timex?'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return T.head.i == signal.i
        
    def signal_follow_event_is_timex3_head(self, E: Token, T: Token) -> bool:
        '''Sinal que segue event é um pai direto do Timex?'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return T.head.i == signal.i
        
    def signal_precede_timex3_is_timex3_head(self, T: Token) -> bool:
        '''Sinal que precede timex é um pai direto do Timex?'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return T.head.i == signal.i
        
    def signal_follow_timex3_is_timex3_head(self, T: Token) -> bool:
        '''Sinal que segue timex é um pai direto do Timex?'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return T.head.i == signal.i


    
    
    def signal_precede_event_dep_advmod_advcl_event(self, E: Token) -> bool:
        '''Sinal que precede event está diretamente relacionado ao Event com advmod ou advcl?'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
            
        dep = self.token1_token2_dep(E, signal)
        return dep in ['advmod', 'advcl']
        
    def signal_follow_event_dep_advmod_advcl_event(self, E: Token) -> bool:
        '''Sinal que segue event está diretamente relacionado ao Event com advmod ou advcl?'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
            
        dep = self.token1_token2_dep(E, signal)
        return dep in ['advmod', 'advcl']
        
    def signal_precede_timex3_dep_advmod_advcl_event(self, E: Token, T: Token) -> bool:
        '''Sinal que precede timex está diretamente relacionado ao Event com advmod ou advcl?'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
            
        dep = self.token1_token2_dep(E, signal)
        return dep in ['advmod', 'advcl']
        
    def signal_follow_timex3_dep_advmod_advcl_event(self, E: Token, T: Token) -> bool:
        '''Sinal que segue timex está diretamente relacionado ao Event com advmod ou advcl?'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
            
        dep = self.token1_token2_dep(E, signal)
        return dep in ['advmod', 'advcl']
        
        
                
    def signal_precede_event_dep_advmod_advcl_timex3(self, E: Token, T: Token) -> bool:
        '''Sinal que precede event está diretamente relacionado ao Timex com advmod ou advcl?'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
            
        dep = self.token1_token2_dep(T, signal)
        return dep in ['advmod', 'advcl']
        
    def signal_follow_event_dep_advmod_advcl_timex3(self, E: Token, T: Token) -> bool:
        '''Sinal que segue event está diretamente relacionado ao Timex com advmod ou advcl?'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
            
        dep = self.token1_token2_dep(T, signal)
        return dep in ['advmod', 'advcl']
        
    def signal_precede_timex3_dep_advmod_advcl_timex3(self, T: Token) -> bool:
        '''Sinal que precede timex está diretamente relacionado ao Timex com advmod ou advcl?'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
            
        dep = self.token1_token2_dep(T, signal)
        return dep in ['advmod', 'advcl']
        
    def signal_follow_timex3_dep_advmod_advcl_timex3(self, T: Token) -> bool:
        '''Sinal que segue timex está diretamente relacionado ao Timex com advmod ou advcl?'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
            
        dep = self.token1_token2_dep(T, signal)
        return dep in ['advmod', 'advcl']

    
    
    
    def signal_precede_event_head_is_event(self, E: Token) -> bool:
        '''O Sinal que precede event  modifica o Event diretamente? (ex: signal é um filho direto do Event?)'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return signal.head.i == E.i
        
    def signal_follow_event_head_is_event(self, E: Token) -> bool:
        '''O Sinal que segue event  modifica o Event diretamente? (ex: signal é um filho direto do Event?)'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return signal.head.i == E.i
        
    def signal_precede_timex3_head_is_event(self, E: Token, T: Token) -> bool:
        '''O Sinal que precede timex  modifica o Event diretamente? (ex: signal é um filho direto do Event?)'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return signal.head.i == E.i
        
    def signal_follow_timex3_head_is_event(self, E: Token, T: Token) -> bool:
        '''O Sinal que segue timex  modifica o Event diretamente? (ex: signal é um filho direto do Event?)'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return signal.head.i == E.i
    
    
    
    def signal_precede_event_head_is_timex3(self, E: Token, T: Token) -> bool:
        '''O Sinal que precede event  modifica o Timex diretamente? (ex: signal é um filho direto do Timex?)'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        return signal.head.i == T.i
    
    def signal_follow_event_head_is_timex3(self, E: Token, T: Token) -> bool:
        '''O Sinal que segue event  modifica o Timex diretamente? (ex: signal é um filho direto do Timex?)'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        return signal.head.i == T.i
    
    def signal_precede_timex3_head_is_timex3(self, T: Token) -> bool:
        '''O Sinal que precede timex  modifica o Timex diretamente? (ex: signal é um filho direto do Timex?)'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        return signal.head.i == T.i
        
    def signal_follow_timex3_head_is_timex3(self, T: Token) -> bool:
        '''O Sinal que segue timex  modifica o Timex diretamente? (ex: signal é um filho direto do Timex?)'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        return signal.head.i == T.i

    
    
    def signal_precede_event_dep_if_child_event(self, E: Token) -> bool:
        '''DEP do Sinal que precede event, se ele for um filho do Event'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        
        if signal.head.i == E.i:
            return self.token_resource(signal, 'dep')
        
        return 'NONE'
        
    def signal_follow_event_dep_if_child_event(self, E: Token) -> bool:
        '''DEP do Sinal que segue event, se ele for um filho do Event'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        
        if signal.head.i == E.i:
            return self.token_resource(signal, 'dep')
        
        return 'NONE'

    def signal_precede_timex3_dep_if_child_event(self, E: Token, T: Token) -> bool:
        '''DEP do Sinal que precede timex, se ele for um filho do Event'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        
        if signal.head.i == E.i:
            return self.token_resource(signal, 'dep')
        
        return 'NONE'
        
    def signal_follow_timex3_dep_if_child_event(self, E: Token, T: Token) -> bool:
        '''DEP do Sinal que segue timex, se ele for um filho do Event'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        
        if signal.head.i == E.i:
            return self.token_resource(signal, 'dep')
        
        return 'NONE'



    def signal_precede_event_dep_if_child_timex3(self, E: Token, T: Token) -> bool:
        '''DEP do Sinal que precede event, se ele for um filho do Timex'''
        signal = self.signal_context_token(E, 'antes')
        if not signal:
            return 'NONE'
        
        if signal.head.i == T.i:
            return self.token_resource(signal, 'dep')
        
        return 'NONE'
        
    def signal_follow_event_dep_if_child_timex3(self, E: Token, T: Token) -> bool:
        '''DEP do Sinal que segue event, se ele for um filho do Timex'''
        signal = self.signal_context_token(E, 'depois')
        if not signal:
            return 'NONE'
        
        if signal.head.i == T.i:
            return self.token_resource(signal, 'dep')
        
        return 'NONE'
        
    def signal_precede_timex3_dep_if_child_timex3(self, T: Token) -> bool:
        '''DEP do Sinal que precede timex, se ele for um filho do Timex'''
        signal = self.signal_context_token(T, 'antes')
        if not signal:
            return 'NONE'
        
        if signal.head.i == T.i:
            return self.token_resource(signal, 'dep')
        
        return 'NONE'
        
    def signal_follow_timex3_dep_if_child_timex3(self, T: Token) -> bool:
        '''DEP do Sinal que segue timex, se ele for um filho do Timex'''
        signal = self.signal_context_token(T, 'depois')
        if not signal:
            return 'NONE'
        
        if signal.head.i == T.i:
            return self.token_resource(signal, 'dep')
        
        return 'NONE'

    
    
    def reichenbach_tense(self, E: Token, T: Token) -> str:
        '''
        Se POS do EVENT é VERB e TIMEX modifica o EVENT (Timex é filho) e TIMEX.tipo é DATE ou TIME então TIMEX = momento de referência (R).
        Se TIMEX = R, então o valor da feature assume os valores: anterior, simples ou posterior conforme o tempo verbal de EVENT equivalente na tabela de Reichenbach.
        O tipo da relação é entre E/R => tempos anteriores: E < R, tempos simples: E = R, tempos posteriores: E > R
        '''
        if not self.pos(E, ['VERB', 'AUX']):
            return 'NONE'
        
        if not self.tipo(T, ['DATE', 'TIME']):
            return 'NONE'
        
        if not self.event_is_ancestor_timex3(E, T):
            return 'NONE'
        
        tempoE = self.tense(E)
        tempo_compostoE = self.tense_compound(E)
        if tempo_compostoE:
            tempo = tempo_compostoE
        else:
            if tempoE == 'NONE':
                return 'NONE'
            tempo = tempoE
        
        tempo_reichenbach = self.__reichenbach_table(tempo)
        if not tempo_reichenbach:
            return 'NONE'
        
        return tempo_reichenbach

    
    def event_modal_verb(self, E: Token) -> str:
        '''Verbo modal antes de EVENT'''
        distancia = 3
        span = reversed(self.spanPrecede(E, distancia))

        for t in span:
            if t.lemma_ in self.list_modal_verbs():
                return t.lemma_.lower()
        return 'NONE'

    def event_has_modal_verb_precede(self, E: Token) -> bool:
        '''Se EVENT tem auxiliares modais antes dele'''
        if self.event_modal_verb(E) == 'NONE':
            return False
        return True

    
    #-------------------------------------------------------------------------------
    # -----FIM FUNÇÕES FEATURES -------------------------------------- -------------
    #-------------------------------------------------------------------------------
    
    
###------------------------------------------------------
#        FIM RULESFUNCTIONS
###------------------------------------------------------



#=========================================================================================================================
# '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
#  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
#  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
#  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
#  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
#  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
#  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
# ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #
#=========================================================================================================================



###------------------------------------------------------
#        CLASS TRANSITIVE_TLINK
###------------------------------------------------------

class TlinkTransitive:
    '''
    Implementa do Fechametno Temporal.
    Adiciona novas relações na relações preditas pela classe TemporalRelation.
    '''
    def __init__(self, tb: TimebankPT):
        self.__tb = tb
        
    def save_tlinks_transitive(self, pares_relacoes_sentenca: dict):
        '''
        Processa e adiciona TLINKs transitivos às relações preditas pelo método.
        São adicionados à estrutura de dados da classe MyTlink.
        
        Args:
            pares_relacoes_sentenca: dicionário no formato {(eventID, timex3ID): typeRel, ('e1', 't2'): 'BEFORE', } contendo todas relações da sentença.
            
        '''
        pares_novos = self.transitive_closure(pares_relacoes_sentenca)
        #print(f'\tNOVOS PARES: {pares_novos}')
        #print('-------------------------------------------')

        if len(pares_novos) > 0:
            task = 'A'
            #cod da regra de transitividade
            rule = 1000
            id_sentenca = self.__tb.id_sentenca_unica[0]
            doc = self.__tb.nome_doc_unico

            for par in pares_novos:
                eventID_transitive, relatedTo_transitive = par
                relType_transitive = pares_novos[par]

                campos_add_transitive = relType_transitive, eventID_transitive, relatedTo_transitive, task, id_sentenca, doc, rule
                self.__tb.my_tlink.add(*campos_add_transitive)
        

    def transitive_closure(self, pairs: dict, retirar_event_event = True) -> dict:
        '''
        Adiciona novos TLINKs utilizando a propriedade de transitividade e simetria das relações. Por exemplo:
            SIMETRIA:
            . Se A←B, então B→A 
            . Se A↔B, então B↔A 
            TRANSITIVIDADE:
            . Se A→B e B→C, então A→C 
            . Se A→B e B↔C, então A→C 
            . Se A→B e A↔C, então C→B 
            . Se A↔B e B↔C, então A↔C 
            . Se A→B e C↔B, então A→C 
            . Se A→B e C↔A, então C→B
            . Se B↔A e B↔C, então A↔C
            . Se A↔B e C↔B, então A↔C
            . Se B↔A e C↔B, então A↔C

        Args:
            pairs: Estrutura de dicionário no seguinte formato: {('e1', 't2'): 'BEFORE', ('e2', 't2'): 'AFTER'}
            retirar_event_event: Se True, retira também as relações EVENT/EVENT: {('e1', 'e2'): 'BEFORE'}

        Return:
            Mesma estrutura de dados da entrada, porém com os TLINKs transitivos adicionados.
            Relações reflexivas e TIMEX3/TIMEX3 são removidas.

        '''
        def only_new_relation() -> dict:
            '''
            Retorna somente as relações novas
            '''
            saida = {}
            diff = set(closure) - set(pairs)
            for par in diff:
                rel = closure[par]
                saida[par] = rel
            return saida

        def union_until_now(dict1, dict2):
            '''Quando as chaves são iguais prevalece o valor o primeiro dicionário'''
            dict_new = dict1.copy()
            for key, value in dict2.items():
                if key not in dict_new:
                    dict_new[key] = value
            return dict_new
        

        #Inverte as relações do tipo AFTER para BEFORE e as relações OVERLAP adiciona mais o par invertido.
        closure = self.__pre_process_pairs(pairs)
        new_relations = {}
        count = 0   #para evitar loop infinito
        while True:
            count += 1
            #print('====>>>>>>>>> COUNT:', count, '<<<<<<<<==============')
            for A1, B1 in closure:
                rel1 = closure[(A1, B1)]
                for A2, B2 in closure:
                    rel2 = closure[(A2, B2)]
                    
                    #FORMA DE INTERPRETAR AS REGRAS DE FECHAMENTO
                    #RELAÇÕES (rel):
                        # → = BEFORE
                        # ↔ = OVERLAP
                    #LÓGICA: Se A →B  e B →C , então A →C 
                    #CÓDIGO: Se A1→B1 e A2→B2 e B1=A2, então A1→B2
                    
                    #DEFINE AS NOVAS RELAÇÕES
                    #Se A→B e B→C, então A→C 
                    if (rel1 == rel2 == 'BEFORE') and (B1 == A2):
                        new_relations[(A1, B2)] = 'BEFORE'

                    #Se A→B e B↔C, então A→C 
                    if (rel1 == 'BEFORE' and rel2 == 'OVERLAP') and (B1 == A2):
                        new_relations[(A1, B2)] = 'BEFORE'

                    #Se A→B e C↔B, então A→C 
                    if (rel1 == 'BEFORE' and rel2 == 'OVERLAP') and (B1 == B2):
                        new_relations[(A1, A2)] = 'BEFORE'

                    #Se A→B e A↔C, então C→B
                    if (rel1 == 'BEFORE' and rel2 == 'OVERLAP') and (A1 == A2):
                        new_relations[(B2, B1)] = 'BEFORE'

                    #Se A→B e C↔A, então C→B
                    if (rel1 == 'BEFORE' and rel2 == 'OVERLAP') and (A1 == B2):
                        new_relations[(A2, B1)] = 'BEFORE'

                    #Se A↔B e B↔C, então A↔C
                    if (rel1 == 'OVERLAP' and rel2 == 'OVERLAP') and (B1 == A2):
                        new_relations[(A1, B2)] = 'OVERLAP'
                        new_relations[(B2, A1)] = 'OVERLAP'

                    #Se B↔A e B↔C, então A↔C
                    if (rel1 == 'OVERLAP' and rel2 == 'OVERLAP') and (A1 == A2):
                        new_relations[(B1, B2)] = 'OVERLAP'
                        new_relations[(B2, B1)] = 'OVERLAP'

                    #Se A↔B e C↔B, então A↔C
                    if (rel1 == 'OVERLAP' and rel2 == 'OVERLAP') and (B1 == B2):
                        new_relations[(A1, A2)] = 'OVERLAP'
                        new_relations[(A2, A1)] = 'OVERLAP'

                    #Se B↔A e C↔B, então A↔C
                    if (rel1 == 'OVERLAP' and rel2 == 'OVERLAP') and (A1 == B2):
                        new_relations[(B1, A2)] = 'OVERLAP'
                        new_relations[(A2, B1)] = 'OVERLAP'

            #União: Quando as chaves são iguais prevalece o valor do segundo dicionário (new_relations)
            #closure_until_now = closure | new_relations
            
            #União: Quando as chaves são iguais prevalece o valor do primeiro dicionário (closure)
            closure_until_now = union_until_now(closure, new_relations)
            #print(f'\tCLOSERE: \nC: {closure}  \nNEW: {new_relations}  \nUNI: {closure_until_now} \n>>>>>>>>IGUAL? {closure_until_now == closure}')

            if (count >= 100): #nunca deveria chegar perto de 100
                raise ValueError('ERRO: Fechamento Temporal em loop infinito.')
                
            #condição de parada do laço while True
            #para quando a adição de uma nova relação não gera mais novas relações transitivas
            if (closure_until_now == closure):
                break
            closure = closure_until_now

        #Todas as relações, inclusive as novas
        #Excluindo relações reflexivas e TIMEX3/TIMEX3. Além de inverter as relações TIMEX3/EVENT para EVENT/TIMEX3.
        closure = self.__pos_process_pairs(closure, retirar_event_event)
        #print(f'\nCOUNT: {count}  POSPRO: {closure} \nSO_NOVA: {only_new_relation()}')

        #retorna somente as relações novas
        return only_new_relation()


    def __pre_process_pairs(self, pairs: dict) -> dict:
        '''
        Inverte as relações do tipo AFTER para BEFORE e as relações OVERLAP adiciona mais o par invertido.

        Args:
            pairs: 
                Estrutura de dicionário no seguinte formato: {('t1', 'e2'): 'AFTER', ('e1', 't2'): 'OVERLAP'}

        Return:
            Mesma estrutura de dados da entrada, porém com os pares e relações invertidas: {('e2', 't1'): 'BEFORE', ('e1', 't2'): 'OVERLAP', ('t2', 'e1'): 'OVERLAP'}
        '''
        saida = {}
        for (x, y) in pairs:
            par = (x, y)
            rel = pairs[(x, y)]

            if rel == 'AFTER':
                par, rel = self.__invert_relation((par, rel))
                saida[par] = rel
            else:
                saida[par] = rel
                if rel == 'OVERLAP':
                    par, rel = self.__invert_relation((par, rel))
                    saida[par] = rel
        return saida

    def __pos_process_pairs(self, pairs: dict, retirar_event_event = True) -> dict:
        '''
        Exclui relações reflexivas e TIMEX3/TIMEX3. Além de inverter as relações TIMEX3/EVENT para EVENT/TIMEX3.

        Args:
            pairs: 
                Estrutura de dicionário no seguinte formato: {('t1', 'e2'): 'AFTER', ('e1', 't2'): 'AFTER'}

            retirar_event_event: 
                Se True, retira também as relações EVENT/EVENT: {('e1', 'e2'): 'BEFORE'}

        Return:
            Mesma estrutura de dados da entrada, porém apenas os pares válidos.

        '''
        closure = pairs.copy()
        for x, y in closure:
            par = (x, y)
            rel = closure[(x, y)]

            #Retira os reflexivo e TIMEX3/TIMEX3 
            if (x == y) or (self.__is_timex3(x) and self.__is_timex3(y)):
                pairs.pop(par)

            #inverter para garantir que EVENT fique sempre primeiro
            if (self.__is_timex3(x) and self.__is_event(y)):
                pairs.pop(par)
                par, rel = self.__invert_relation((par, rel))
                pairs[par] = rel

        #Retirar EVENT/EVENT
        if retirar_event_event:
            closure = pairs.copy()
            for x, y in closure:
                par = (x, y)
                if (self.__is_event(x) and self.__is_event(y)):
                    pairs.pop(par)

        return pairs


    def __invert_relation(self, relation: Literal["((x, y), rel)"]) -> Literal["((y, x), rel)"]:
        '''
        Inverte o tipo da relação e seu par.

        Args:
            relation: par no formato: ((x, y), 'AFTER')')

        Return:
            Retorna par invertido: (y, x), 'BEFORE')

        '''
        par, rel = relation
        if rel == 'AFTER':
            rel = 'BEFORE'
        elif rel == 'BEFORE':
            rel = 'AFTER'

        invert_par = (par[1], par[0])

        return (invert_par, rel)


    def __is_event(self, token):
        '''
        Retorna True se token for do tipo EVENT
        '''
        if token[0] == 'e':
            return True
        return False

    def __is_timex3(self, token):
        '''
        Retorna True se token for do tipo TIMEX3
        '''
        if token[0] == 't':
            return True
        return False


#---------------------------------------------------------------------
#     FIM TRANSITIVE_TLINK
#--------------------------------------------------------------------





#=========================================================================================================================
# '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
#  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
#  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
#  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
#  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
#  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
#  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
# ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #
#=========================================================================================================================

#----------------------------------------------------------------------------------
# Classe TLINK CANDIDATE ML  -> Classifica EVENT candidatos à geração de TLinks
#----------------------------------------------------------------------------------

class TlinkCandidate():
    '''
    Seleciona EVENT com maior probabilidade de estarem anotados no corpus, para geração de TLINKs.
    
    CLASSE DESCONTINUADA. FAZIA SENTIDO APENAS QUANTO SE TENTAVA AVALIAR TAMBÉM OS PARES NÃO ROTULADOS.
    
    '''
    def __init__(self):
        '''
        Descrição das colunas do DataFrame:
            . e_pos         = POS tag do EVENT
            . e_dep_com_pai = Relação de dependência entre o EVENT e seu pai.
            . e_class       = Classe do EVENT
            . e_pai_pos     = POS tag do pai do EVENT
            
        '''
        self.__col_encode = {'e_pos': 'pos', 'e_dep_com_pai': 'dep', 'e_class': 'classe'}  # , 'e_pai_pos': 'pos'   
        self.__col_classe = ['anotado']
        self.__colunas = list(self.__col_encode.keys()) + self.__col_classe
        
        
        subdir = ''
        if os.path.realpath('').find('temporal_relation') < 0: # não encontrou
            subdir = 'temporal_relation\\'
        
        self.__file_model = subdir + 'dtree.model' #Nome do arquivo onde o modelo treinado será salvo
        self.dtree = None
        self.__siglas = Siglas()
        
        if self.__file_model_exist():
            self.__load_model()
        
        self.__approach = 'ML'   # REGRAS OR ML
        
        #se approach = ML
        self.__X_treino  = pd.DataFrame(None)
        self.__X_teste   = pd.DataFrame(None)
        self.__y_treino  = pd.DataFrame(None)
        self.__y_teste   = pd.DataFrame(None)
        self.__threshold = 0.30
        
    @property
    def approach(self):
        '''
        Propriedade que determina qual abordagem será utilizada para seleção de Tlinks Candidatos.
        
        Se 'REGRAS': será considerada as características do EVENT mais frequentemente anotadas para a submissão à predição da relação temporal.
        Se 'ML': será utilizado modelo de machine learning (Decision Tree) pré-treinado para a mesma tarefa.
        
        '''
        return self.__approach
    
    @approach.setter
    def approach(self, valor: str):
        valor = valor.upper()
        approach_validos = ['REGRAS', 'ML']
        if valor not in approach_validos:
            print(f'Approach válidos: {approach_validos}')
            self.__approach = 'REGRAS'
        
        self.__approach = valor
        
    
    @property
    def threshold(self):
        '''
        Propriedade que ajusta o limite da probabilidade para a classificação.
        Valor default = 0.5.
        
        '''
        return self.__threshold
    
    @threshold.setter
    def threshold(self, valor: float):
        if not(0 < valor <= 1):
            print('O valor de threshold deve ser 0 < X <= 1.')
            self.__threshold = 0.5
            
        self.__threshold = valor
    
    
    def predict(self, tokenE: Token) -> bool:
        '''
        Prediz EVENT propenso a está anotado no corpus para geração de Tlinks.
        
        Args:
            tokenE: Token que representa um EVENT na sentença
            
        Return:
            Tupla contendo (Valor predito, probabilidade da predição)
            
        '''
        if not self.__valida_modelo:
            return
        
        df_predict = self.encode_tokenE(tokenE)
        predict_prob = self.dtree.predict_proba(df_predict)
        predict = (predict_prob[:,1] >= self.threshold).astype(bool)
        
        if len(predict) > 0:
            return predict[0]
        else:
            return False

    
    def train(self, df_features: DataFrame):
        '''
        Treina e salva modelo de machine learning (Decision Tree) para selecionar EVENTs com maior probabilidade de estarem anotados.
        Eles serão utilizados para geração de TLINKs.
        
        Args:
            df_features:    Dataframe[colunas], onde colunas = ['e_pos', 'e_dep_com_pai', 'e_class', 'e_pai_pos', 'anotado']
                            Pode ser obtido dos dados anotados do corpus na classe: RelacoesTemporais.df_features.reset_index()[colunas]
        '''
        #Codifica os dados para o modelo ML
        df_codes = self.encode_df(df_features.copy())

        #Dataset sem dummies e com variáveis codificadas
        X = df_codes.loc[:, list(self.__col_encode.keys())]
        y = df_codes.loc[:, self.__col_classe]

        #Dataset dividido sem dummies e com variáveis codificadas
        self.__X_treino, self.__X_teste, self.__y_treino, self.__y_teste = train_test_split(X, y, test_size = 0.2, random_state = 17)
        
        #TREINA E SALVA MODELO
        try:
            self.dtree = DecisionTreeClassifier(criterion="gini", random_state=17,  max_depth=4, splitter='best')
            self.dtree.fit(self.__X_treino, self.__y_treino)
        except Exception as e:
            print(f'Ocorreu um erro no treinamento. ERRO {e}')
        else:
            print('Modelo treinado com sucesso.')

        #Salvar modelo treinado
        self.__save_model()
        
    
    def performance(self):
        '''
        Exibe avaliação dos dados de teste do modelo treinado.
        
        '''
        if not self.__check_model_trained():
            print('Modelo ainda não treinado.')
            return
        
        #Carregar modelo salvo
        if not self.__load_model():
            return
        
        #PREDIÇÃO
        y_pred_dtree = (self.dtree.predict_proba(self.__X_teste)[:,1] >= self.threshold).astype(bool)
        
        cm = confusion_matrix(self.__y_teste, y_pred_dtree)
        df_cm = pd.DataFrame(cm, index = ['NÃO ANOTADO', 'ANOTADO'], columns = ['NÃO ANOTADO', 'ANOTADO'])
        plt.figure(figsize=(5,4))
        sns.heatmap(df_cm, cmap='PuBu', annot=True, fmt="d")
        plt.title('Matriz de confusão do classificador')
        plt.xlabel('__________ PREDITO __________')
        plt.ylabel('____________ REAL ____________')

        print("", self.__X_treino.shape[0], "para treino.\n", self.__X_teste.shape[0], "para testes")
        print('\n')
        print(classification_report(self.__y_teste, y_pred_dtree, target_names=['NÃO ANOTADO', 'ANOTADO'], digits=3))
        print('\n')

    
    def view_tree(self):
        '''
        Exibe árvore de decisão do modelo.
        
        '''
        if not self.__check_model_trained():
            print('Modelo ainda não treinado.')
            return
        
        #Lista de Campos do dataset
        features = list(self.__X_treino)
        classes_names = ['%s' % i for i in self.dtree.classes_]

        dot_data = io.StringIO()
        export_graphviz(self.dtree, out_file=dot_data, feature_names=features, filled=True, rounded=True, special_characters=True, class_names=classes_names)
        graph = pydotplus.graphviz.graph_from_dot_data(dot_data.getvalue())
        
        return Image(graph.create_png()) 
        
    def feature_importances(self):
        '''
        Exibe features mais relevantes para a classificação.
        
        '''
        if not self.__check_model_trained():
            print('Modelo ainda não treinado.')
            return
        
        #Importância de cada variável
        feature_importances = pd.DataFrame(self.dtree.feature_importances_, index=self.__X_treino.columns, columns=['importance']).sort_values('importance', ascending = True)
        feature_importances.plot(kind='barh', figsize=(10,8), grid=False, legend=False)
        plt.title('Importância das Variáveis', fontsize=20)
        print('\n')
        
    ## ----------------------------------------------
    ##             PRIVATES
    ## ----------------------------------------------
    
    def __check_model_trained(self) -> bool:
        '''
        Verifica se o modelo foi treinado.
        '''
        if self.__X_treino.empty or self.__X_teste.empty or self.__y_treino.empty or self.__y_teste.empty:
            return False
        else:
            return True
        
    def __valida_modelo(self):
        '''
        Verifica se existe o objeto do modelo treinado.
        '''
        if self.dtree:
            return True
        else:
            print('Modelo ainda não treinado ou não carregado.')
            return False
    
    def __file_model_exist(self):
        '''
        Verifica se existe o arquivo do modelo salvo .
        '''
        if os.path.isfile(self.__file_model):
            return True
        else:
            return False
        
    def __save_model(self):
        '''
        Salva modelo treinado em arquivo.
        '''
        #Salvar modelo treinado
        if not self.__valida_modelo:
            return False
        
        with open(self.__file_model, 'wb') as model:
            pickle.dump(self.dtree, model)
            
    def __load_model(self):
        '''
        Carrega arquivo do modelo salvo.
        '''
        #Carregar modelo salvo
        if not self.__file_model_exist():
            print('Arquivo salvo do modelo treinado não existe: ' + self.__file_model)
            return False
            
        with open(self.__file_model, 'rb') as model:
            self.dtree = pickle.load(model)
            return True

    def encode_df(self, df: DataFrame) -> DataFrame:
        '''
        Encode features para submissão a modelos de machine learning.
        
        Args:
            df: Dataframe contendo as colunas: 'e_pos', 'e_dep_com_pai', 'e_class', 'e_pai_pos'
                
        '''
        for col_name in self.__col_encode:
            #df.loc[:, col_name] = df.loc[:, col_name].apply(lambda x: self.__siglas.get_key(x, self.__col_encode[col_name]))
            df[col_name] = df[col_name].apply(lambda x: self.__siglas.get_key(x, self.__col_encode[col_name]))
            
            #<<AQUI>>
            #df[col_name] = df[col_name].apply(lambda x: self.__siglas.get_key(x, self.__col_encode[col_name]))
            #WARNING:  df[df.columns[i]] = newvals` or, if columns are non-unique, `df.isetitem(i, newvals)
            
        return df

    def encode_tokenE(self, tokenE: Token):
        '''
        Constrói e codifica DataFrame com features de 'tokenE' para submissão ao modelo de learning.
        
        Args:
            tokenE: Token que representa o EVENT em uma sentença.
            
        '''
        e_pos         = tokenE.pos_
        e_dep_com_pai = tokenE.dep_
        e_class       = tokenE._.classe
        e_pai_pos     = tokenE.head.pos_
        
        #Monta registro para ficar neste formato: {'e_pos': e_pos, 'e_dep_com_pai': e_dep_com_pai, 'e_class': e_class, 'e_pai_pos': e_pai_pos}
        registro = {}
        for k in self.__col_encode.keys():
            registro[k] = eval(k)
        
        df_feats = pd.DataFrame(registro, index = [0])
        df_encode = self.encode_df(df_feats.copy())
        return df_encode


#---------------------------------------------------------------------
#     FIM TLINK CANDIDATE ML
#--------------------------------------------------------------------



#----------------------------------------
# CLASS TOKENDCT
#----------------------------------------
#para task B -> trabalho futuro
class TokenDct:
    '''
    Estrutura de dados para um DCT.
    Útil para a task B.
    '''
    def __init__(self, tb: 'TimebankPT'):
        self.__doc = None
        self.__nome = None
        self.__dct = None
        self.__tid = None
        self.__type = None
        self.__train_test = None
        self.__id_sentenca = None
        self.__tb = tb

    def atualizar(self):
        self.__doc = self.__tb.doc_unico
        doc = self.__doc

        self.__nome = doc._.nome
        self.__dct = doc._.dct
        self.__tid = doc._.dct_tid
        self.__type = doc._.dct_type
        self.__train_test = doc._.train_test
        self.__id_sentenca = doc._.id_sentenca

    def __str__(self):
        if self.__doc:
            return self.dct

    def __repr__(self):
        if self.__doc:
            return self.dct

    @property
    def nome(self):
        return self.__nome

    @property
    def dct(self):
        return self.__dct

    @property
    def tid(self):
        return self.__tid

    @property
    def type(self):
        return self.__type
    
    @property
    def train_test(self):
        return self.__train_test

    @property
    def id_sentenca(self):
        return self.__id_sentenca

# FIM CLASS TokenDct
#-----------------------------------------------



#=========================================================================================================================
# '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
#  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
#  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
#  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
#  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
#  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
#  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
# ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #
#=========================================================================================================================

#---------------------------------------------------------------------
# Classe TEMPORALRELATION - Atribui RT
#--------------------------------------------------------------------

class TemporalRelation():
    '''
    Identifica tipos de relações temporais em sentenças em português.
    
    Args:
        tb: instancia da class TimebankPT.
    '''
    __COD_REGRA      = 0
    __RELTYPE        = 1
    __ORDEM          = 2
    __PREDICADOS     = 3
    __ORIGEM         = 4
    __ACURACIA       = 5
    __ACERTOS        = 6
    __ACIONAMENTOS   = 7
    __TIPO_ORDEM = ['cod_regra', 'relType', 'ordem', 'random', 'origem', 'acuracia', 'acertos', 'acionamentos', 'ordem_origem']

    def __init__(self, tb: TimebankPT):
        
        self.__tb = tb
        self.__id_sentenca = None
        self.__id_sentenca_anterior = None
        
        #Funções que auxiliam na composição das regras, recebe o objeto Doc atual
        self.f = RulesFunctions(self.__tb)
                
        #Cria instancia de classe SetRulesEmpty() apenas para informar ao usuário a forma de inicializar SetRules()
        self.setRules = SetRulesEmpty()
        
        #options
        self.__task = 'A'
        self.__rules = None
        self.__show_result = 'train'
        self.__show_extras = False
        
        self.__processing_type = 'primeira_regra'
        self.__order = ''
        self.__sort_reverse = False
        
        self.__df_pred = None
        self.__rule_triggers = defaultdict(lambda: {'relTypeRule': None, 'relTypePares': []})
        
        self.__df_features = None
        self.tlink_transitive = TlinkTransitive(tb)
        self.tlink_candidate = TlinkCandidate()
        self.__active_tlink_transitive = False
        self.__active_tlink_candidate = False
        
    
    def setRules_start(self, nome_arquivo: str = '', create_dataset: bool = False):
        '''
        Inicia conjunto de regras.
        Para verificar os nomes e parâmetros das funções, é necessário carregar dataset de features.
        
        Args:
            create_dataset: Se True, usa o método 'TimebankPT.features.create_dataset()' para criar dataset de features do zero. Mais lento.
                            Se False, usa o método 'TimebankPT.features.load_dataset()' para carregar um dataset já salvo pelo método TimebankPT.features.save_dataset(nome_arquivo).
            nome_arquivo:   Se create_dataset for False, então nome_arquivo é obrigatório.
                            Representa o caminho do dataset de features salvo.
            
        '''
        if create_dataset:
            self.__tb.features.create_dataset()
        else:
            self.__tb.features.load_dataset(nome_arquivo)
        
        funcs = self.__tb.features.generate_params_functions() 
        self.setRules = SetRules(params_functions=funcs)
        
    @property
    def show_result(self):
        '''
        Determina qual resultado será exibido: treinamento ou teste.
        '''
        return self.__show_result
    
    @show_result.setter
    def show_result(self, train_test: str):
        
        train_test = train_test.lower()
        validos = ['train', 'test']
        
        if train_test not in validos:
            if self.__tb.dev:
                train_test = ['train', 'train_test']
            else:
                train_test = ['train', 'train_test', 'test']
        else:
            if self.__tb.dev:
                if train_test == 'test':
                    train_test = 'train_test'
            else:
                if train_test == 'train':
                    train_test = ['train', 'train_test']

        self.__show_result = train_test
    
    @property
    def show_extras(self):
        '''
        Determina se os resultados das predições de dados não rotulados (extras) serão considerados.
        '''
        return self.__show_extras
    
    @show_extras.setter
    def show_extras(self, value: bool):
        if type(value) != bool:
            raise ValueError('show_extras deve ser booleano')
        
        self.__show_extras = value

    @property
    def id_sentencas_sem_predicao(self):
        '''
        Lista de id_sentenca que ainda não houve predição.
        '''
        if self.df_real_predict.empty: #[ ] monitorar df_real_predict. Verificar if self.df_real_predict is None
            return []
        df = self.df_real_predict
        return df[df['relType_pred'].isna()]['isentenca'].unique().tolist()
        
    
    @property
    def task(self):
        '''
        Define o tipo de tarefa que será processada.
        Pode ser: 'A', 'B', 'C'
        
        '''
        return self.__task
    
    @task.setter
    def task(self, task):
        task = task.upper()
        if task not in ['A', 'B', 'C']:
            print("ERROR: 'task' inválida. \nTask válidas: A, B e C.\n")
            return
            
        self.__task = task

        
    @property
    def rules(self) -> list:
        '''
        Lista de regras que serão passadas à instancia da classe para processamento através do método process_rules().

        Args:
            rules: lista de listas, no formato: 
                [[código regra: float, 
                    tipo de relação temporal: str, 
                    ordem de execução: float, 
                    expressão lógica que representa a regra: str, 
                    origem: algoritmos gerador,
                    acuracia: float,
                    acertos: int,
                    acionamentos: int
                ]]
                As funções que compõe as regras estão em TemporalRelation.f (são acessadas geralmente com o prefixo 'self.f.'). 
                Ex: [249, "OVERLAP", 2, "self.f.is_dependencyType(tokenT, tokenE, 'conj')", 'RIPPER', 0, 0, 0]
                
            #A 'ordem de execução' com números negativos torna a regra inativa.
        '''
        return self.__rules
    
    @rules.setter
    def rules(self, rules: Union[list, DataFrame]):

        if len(rules) == 0:
            self.__rules = []
            return
        
        if type(rules) == DataFrame:
            rules = rules.values.tolist()

        #if type(rules[0]) == self.setRules.Rule: #usar essa classe aqui, exigiria inicializar SetRules sem necessidade.
        if type(rules[0]) not in [str, list]:
            try:
                #converte Rule em string depois em lista
                rules = [eval(str(rule)) for rule in rules]
            except Exception as e:
                print("ERROR: Problema ao converter objeto SetRule.Rule em string. \nERRO:", e)


        if self.check_rules(rules):
            #Grava somente regras ativas
            self.__rules = list(filter(lambda x: x[self.__ORDEM] >= 0, rules))
        else:
            self.__rules = []
    
    
    def add_setRules(self, rules: List[list], reset_cod_regra: bool = True, reset_order: bool = False):
        '''
        Adiciona conjunto de regras provenientes dos objetos TemporalRelation.rules ou TemporalRelation.setRules.rules
        Para adicionar de arquivos em formato texto, use TemporalRelation.load_rules(nome_arquivo)

        Args:
            setRules: conjunto de regras no formato lista de listas dos objetos:
                TemporalRelation.rules ou 
                TemporalRelation.setRules.rules
        '''
        if type(rules) != list:
            raise TypeError('Conjunto de regras deve ser do tipo lista de listas')
        
        if len(rules) == 0:
            print('Conjunto de regras vazio.')
            return

        if type(rules) == DataFrame:
            rules = rules.values.tolist()
            
        if type(rules[0]) not in [str, list]:
            try:
                #converte Rule em string depois em lista
                rules = [eval(str(rule)) for rule in rules]
            except Exception as e:
                print("ERROR: Problema ao converter objeto SetRule.Rule em string em TemporalRelation.add_setRules. \nERRO:", e)

        if len(self.rules) == 0:
            max_elem = 1
        else:
            max_elem = max([elem[0] for elem in self.rules]) + 1

        for elem in rules:
            if reset_cod_regra:
                elem[0] = max_elem
            if reset_order:
                elem[2] = max_elem
            max_elem += 1
            self.rules.append(elem)
        print(f'Total de Regras Adicionadas: {len(rules)}. Total: {len(self.rules)}.')



    def rules_filter_list_cods(self, lista_cod_regras: list):
        '''
        Filtra as regras a serem processadas pelo códigos das regras informadas.
        Esta função altera as self.rules. Para desfazer o filtro, é necessário atribuir as regras novamente: self.rules = listas_das_regras.
        
        Args: 
            lista_cod_regras: lista de códigos das regras.
            
        '''
        if type(lista_cod_regras) != list or not lista_cod_regras or len(lista_cod_regras) == 0:
            print('ERROR: rules_filter deve receber lista contendo códigos das regras que deseja filtrar.')
            return
        
        if (not lista_cod_regras) or (len(lista_cod_regras) == 0):
            self.rules = []

        self.rules = list(filter(lambda x: x[self.__COD_REGRA] in lista_cod_regras, self.rules))
    

    def filter_rules_acuracia(self, value: float):
        ''' Filtra conjunto de regras atual por 'acurácia' maior que 'value' '''
        self.rules = list(filter(lambda x : x[self.__ACURACIA] >= value, self.rules))

    def filter_rules_ordem(self, value: float):
        ''' Filtra conjunto de regras atual por 'ordem' menor que 'value' '''
        self.rules = list(filter(lambda x : x[self.__ORDEM] <= value, self.rules))

    def  filter_rules_primeiras(self, x: int):
        ''' Filtra conjunto de regras atual pelas primeiras 'X' regras por tipo de origem '''
        if type(x) != int or x < 1:
            raise TypeError('X de ser inteiro maior ou igual a 1')
        
        grouped = {}
        for sublista in self.rules:
            chave = sublista[self.__ORIGEM]
            if chave not in grouped:
                grouped[chave] = []
            grouped[chave].append(sublista)

        # selecionar as X primeiras regras de cada grupo
        result = []
        for sublistas in grouped.values():
            result.extend(sublistas[:x])
        self.rules = result
    
    @property
    def df_rules(self):
        '''
        Exibe as regras ativas em formato de tabela.
        '''
        colunas=['cod_regra', 'relType', 'ordem', 'predicados', 'origem', 'acuracia', 'acertos', 'acionamentos']
        
        return pd.DataFrame(self.rules, columns=colunas)

    
    def has_rule_class_default(self) -> str:
        '''
        Verifica se há regras com classe default.
        Se houver, retorna a regra de classe default.
        '''
        rule_default = list(filter(lambda x: x[self.__PREDICADOS] == 'True == True', self.rules))
        if len(rule_default) >= 1:
            return rule_default[0] #retorna regra default
        return False # se não houver regra default
        
    def add_rule_class_default(self, class_default: str):
        '''
        Adicionar regra para classe default em TemporalRelation.rules
        '''
        if self.has_rule_class_default():
            print('Já tem regra para classe default:', self.has_rule_class_default()[1])
        else:
            self.rules.append([2000, class_default, 2000, 'True == True', 'DEFAULT', 0, 0, 0])
            print('Regra default adicionada.')
        
    def remove_rule_class_default(self):
        '''
        Remove todas as regras de classe default, se houver.
        '''
        if self.has_rule_class_default():
            while self.has_rule_class_default() in self.rules:
                print('Regra removida:', self.has_rule_class_default())
                self.rules.remove(self.has_rule_class_default())
        else:
            print('Não há regra de classe default.')
    

    def get_rules(self, cod_rules: list):
        '''
        Retorna lista de regras que possuem 'cod_rules'.
        
        Args:
            cod_rules: lista de códigos de regras

        '''
        list_int = []
        if type(cod_rules) == int:
            list_int.append(cod_rules)
            cod_rules = list_int
            
        if type(cod_rules) != list:
            print('ERRO: cod_rules deve ser uma lista.')
            return []

        return list(filter(lambda x: x[0] in cod_rules, self.rules))

    
    def save_rules(self, nome_arquivo: str):
        '''
        Salva em formato pickle conjunto de regras carregado em TemporalRelation.rules
        '''
        nome_arquivo = self.__tb.check_filename(filename=nome_arquivo, extensao='pickle')

        with open(nome_arquivo, 'wb') as f:
            pickle.dump(self.rules, f)

    def load_rules(self, nome_arquivo: str):
        '''
        Carrega conjunto de regras em formato .pickle ou .txt.
        Se .pickle, deve ter sido salvas pelo método TemporalRelation.save_rules(nome_arquivo).
        Se .txt, deve ter sido salvas pelo método TemporalRelation.save_rules_to_txt(nome_arquivo).

        Args:
            nome_arquivo: o arquivo pode estar em formato pickle ou txt.
        '''
        nome_arquivo = self.__tb.check_filename(filename=nome_arquivo)
        _, ext = os.path.splitext(nome_arquivo.strip())

        try: 
            if ext == '.pickle':
                with open(nome_arquivo, 'rb') as f:
                    self.rules = pickle.load(f)

            elif ext == '.txt':
                with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
                    self.rules = [eval(str(rule)) for rule in arquivo]
            else:
                raise TypeError('ERRO: Extensão do arquivo inválida.')

        except Exception as e:
            print(f'ERRO: Problema ao carregar regras com TemporalRelation.load_rules({nome_arquivo}). ERRO: {e}')
        else:
            print(f'As regras do arquivo {nome_arquivo} foram carregadas com sucesso. \nTotal de regras: {len(self.rules)}')


    def save_rules_to_txt(self, nome_arquivo: str):
        '''
        Salva TemporalRelation.rules em arquivo em formato de texto.
        Pode ser carregado pelo método TemporalRelation.load_rules(nome_arquivo)
        '''
        nome_arquivo = self.__tb.check_filename(filename=nome_arquivo, extensao='txt')

        with open(nome_arquivo, 'w', encoding='utf-8') as f:
            for rule in self.rules:
                f.write(str(rule) + '\n') 

        
    def convert_id_to_token(self, eventID: str, relatedTo: str) -> tuple:
        '''
        Converte ids de EVENT e TIMEX3 para Token.
        
        Args:
            eventID: ID do EVENT no corpus
            relatedTo: ID do TIMEX3 ou EVENT no corpus
        
        Return
            Tupla contendo tokens que representam a entidade EVENT e/ou TIMEX3.
            
        '''
        if not self.task:
            print("'TemporalRelation.task' não definido.")
            return (None, None)
        
        token_dct = TokenDct(self.__tb)
        token_eventID = self.__tb.my_tlink.idtag_to_token(eventID)
        if self.task == 'A':
            token_relatedTo = self.__tb.my_tlink.idtag_to_token(relatedTo)
        elif self.task == 'B':
            token_dct.atualizar()
            token_relatedTo = token_dct
        elif self.task == 'C':
            token_relatedTo = self.__tb.my_tlink.idtag_to_token_next(relatedTo)
            
        return (token_eventID, token_relatedTo)
    
    
    
    
    ############ OPÇÕES
    
    def get_sort_rules(self) -> str:
        '''
        Obtém a ordem atual das regras.
        
        '''
        if type(self.__order) == str:
            return self.__order
        else:
            return self.__TIPO_ORDEM[self.__order]

    def sort_rules(self, order: str, reverse = False):
        '''
        Ordena as regras.
        A ordem atual pode ser obtida pelo método 'get_sort_rules()'
        
        Args:
            order: 'cod_regra', 'relType', 'ordem', 'random', 'origem', 'acuracia', 'acertos', 'acionamentos', 'ordem_origem'
                    se as regras forem processadas, permite que seja uma lista de cod_regras
                Se 'order' for 'ordem_origem': 
                    Intercala origem conforme sua ordem. Útil quando filtrado com filter_rules_primeiras_x().
                    Ex: CBA, CBA, CBA, IDS, IDS, IDS  --> CBA, IDS, CBA, IDS, CBA, IDS
        '''
        def order_list(lista_cod_regras):
            '''
            Retorna lista de código de regra de self.rules conforme ordem de 'lista_cod_regras'

            Args:
                lista_cod_regras: lista de cod_regras na ordem que se deseja executar as regras
            '''
            rules = []
            for sort_rule in lista_cod_regras:
                for rt_rule in self.rules:
                    if rt_rule[0] == sort_rule:
                        rules.append(rt_rule)

            #adiciona as regras que não possuem resultados
            cod_rules = [r[0] for r in rules]
            for rule in self.rules:
                if rule[0] not in cod_rules:
                    rules.append(rule)

            return rules
    
        #se order for do tipo list
        #Ordena código da regra de self.rules conforme ordem do lista order
        if type(order) == list:
            self.__order = 'lista'
            self.rules = order_list(order)
            return
        
        self.__sort_reverse = reverse
        
        if order not in self.__TIPO_ORDEM:
            print('ERROR: Ordens válidas: ' + str(self.__TIPO_ORDEM))
            print("Default: 'ordem'")
            order = ''

        if order == '':
            self.__order = ''
            return
        
        if order == 'cod_regra':
            self.__order = self.__COD_REGRA
        
        elif order == 'relType':
            self.__order = self.__RELTYPE
            
        elif order == 'ordem':
            self.__order = self.__ORDEM
            
        elif order == 'origem':
            self.__order = self.__ORIGEM
        
        elif order == 'acuracia':
            self.__order = self.__ACURACIA
        
        elif order == 'acertos':
            self.__order = self.__ACERTOS

        elif order == 'acionamentos':
            self.__order = self.__ACIONAMENTOS
            
        elif order == 'random':
            self.__order = 'random'
        
        elif order == 'ordem_origem':
            self.__order = 'ordem_origem'


        if self.__order == 'random':
            random.shuffle(self.rules)
        elif self.__order == 'ordem_origem':
            self.__sort_rules_ordem_origem()
        else:
            self.rules.sort(key = lambda x: x[self.__order], reverse = reverse)

    def get_sort_reverse(self):
        '''
        Obtém a informação sobre se a ordem das regras está ascendente ou descendente.
        '''
        return self.__sort_reverse
        
    def __sort_rules_ordem_origem(self):
        '''
        Intercala origem conforme sua ordem. Útil quando filtrado com filter_rules_primeiras_x(x).
        Ex: se as regra estiverem esta ordem: CBA, CBA, CBA, IDS, IDS, IDS, ficará assim: CBA, IDS, CBA, IDS, CBA, IDS
        '''
        # cria um dicionário para armazenar as sublistas de cada tipo de elemento[3]
        d = {}
        for sublist in self.rules:
            key = sublist[self.__ORIGEM]
            if key not in d:
                d[key] = []
            d[key].append(sublist)

        # cria uma lista vazia para armazenar a saída ordenada
        result = []

        # enquanto houver sublistas em qualquer um dos dicionários
        while any(d.values()):
            # percorre o dicionário e adiciona a primeira sublista de cada chave à lista de saída e remove-a do dicionário
            for key in list(d.keys()):
                if d[key]:
                    result.append(d[key][0])
                    del d[key][0]
        self.rules = result

    
    @property
    def processing_type(self):
        '''
        Propriedade que define a forma como as regras serão processadas.
        Se igual a  'votacao' -> Todas as regras são processadas para todos os pares de relação. A relação temporal mais frequente é retornada. (default)
                    'primeira_regra' -> A relação temporal do par é retornada na primeira regra acionada. Para o par atual, as outras regras não são processadas.

        '''
        return self.__processing_type
    
    @processing_type.setter
    def processing_type(self, tipo: str):
        TIPO_PROCESSAMENTO = ['primeira_regra', 'votacao']
        tipo = tipo.lower()
        if tipo not in TIPO_PROCESSAMENTO:
            print('ERROR: Tipo de Processamento inválido. Tipos Válidos: ', str(TIPO_PROCESSAMENTO))
            self.__processing_type = 'votacao'
        else:
            self.__processing_type = tipo
        
    
    def status(self):
        '''
        Exibe o status do processamento das regras.
        '''
        if not self.rules:
            quant_regras = 'REGRAS AINDA NÃO INFORMADAS'
        else:
            quant_regras = len(self.rules)
        
        if not self.task:
            task = 'NÃO INFORMADA'
        else:
            task = self.task
        
        print("STATUS PROCESSAMENTO:")
        print('{0:<27} {1}'.format('Task:', task))
        print('{0:<27} {1}'.format('Quant. Regras Ativas:', quant_regras))
        print('{0:<27} {1}'.format('Ordenada por:', self.get_sort_rules()))
        print('{0:<27} {1}'.format('Ordem reversa:', self.get_sort_reverse()))
        print('{0:<27} {1}'.format('Tipo Processamento:', self.processing_type))
        print('{0:<27} {1}'.format('Tlink Candidato Ativo?', self.active_tlink_candidate))
        print('{0:<27} {1}'.format('Tlink Transitivo Ativo?', self.active_tlink_transitive))
        print('{0:<27} {1}'.format('Tlink Candidato Approach:', self.tlink_candidate.approach if self.active_tlink_candidate else ''))
        print('{0:<27} {1}'.format('Tlink Candidato Threshold:', self.tlink_candidate.threshold if (self.active_tlink_candidate and self.tlink_candidate.approach in ['ML']) else ''))
        print('{0:<27} {1}'.format('Regras processadas?', 'SIM' if self.__check_process_rules() else 'NÃO'))

    def status_resumido(self):
        '''
        Exibe o status resumido do processamento das regras.
        '''
        if not self.rules:
            quant_regras = 'Regras não informadas'
        else:
            quant_regras = str(len(self.rules)) + ' regras ativas'
        
        return {'processing_type': self.processing_type.upper(), 
                'quant_regras': quant_regras, 
                'show_result':self.show_result,
                'pares_candidatos': 'Pares Candidatos' if self.active_tlink_candidate else 'Sem Pares Candidato',
                'fechamento': 'Fechamento' if self.active_tlink_transitive else 'Sem Fechamento'}
        
    def status_resumido_str(self):
        '''
        Exibe o status resumido do processamento das regras.
        '''
        status = self.status_resumido()
        return f"STATUS: {status['processing_type']} | {status['quant_regras']} | ({status['show_result']})\n\t{status['pares_candidatos']} | {status['fechamento']}"
    
    @property
    def active_tlink_transitive(self):
        '''
        Propriedade booleana que informa à classe se serão adicionados TLINKs transitivos aos TLINKs preditos pelo sistema.
        '''
        return self.__active_tlink_transitive
    
    @active_tlink_transitive.setter 
    def active_tlink_transitive(self, valor: bool):
        self.__active_tlink_transitive = valor
    
    
    @property
    def active_tlink_candidate(self):
        '''
        DESCONTINUADO.

        Propriedade booleana que informa à classe se serão ou não selecionados EVENTs com maior probabilidade se estarem anotados no corpus para posterior geração de TLINKs.
        '''
        return self.__active_tlink_candidate
    
    @active_tlink_candidate.setter 
    def active_tlink_candidate(self, valor: bool):
        self.__active_tlink_candidate = valor
    
    
    ############## FIM OPÇÕES
    
    
    def __check_process_rules(self):
        '''
        Verifica se as regras foram processadas.
        '''
        if self.__tb.my_tlink.to_df.empty:
            print('ERROR: Regras ainda não processadas. Utilize o método process_rules().')
            return False
        return True
    
    
    def check_rules(self, rules: list) -> bool:
        '''
        Checa a consistências da lista de regras. 
        
        Args: 
            rules: lista que contem as regras.
            
        '''
        #check estrutura inicial
        if type(rules[0]) != list:
            print('ERROR: rules deve ser uma lista de listas, no formato: [[código regra, tipo de relação temporal, ordem de execução, expressão lógica que representa a regra, origem, acuracia, acertos, acionamentos]')
            return False
        
        unico = []
        repetido = []
        
        #atribui uma sentença qualquer e tokens quaisquer para avalia a expressão que representa a regra
        self.__tb.id_sentenca = 10
        tokenE = self.__tb.doc_unico[9]
        tokenT = self.__tb.doc_unico[1]
        
        for rule in rules:
            #Regras com código repetido
            if rule[self.__COD_REGRA] not in unico:
                unico.append(rule[self.__COD_REGRA])
            else:
                repetido.append(rule[self.__COD_REGRA])

            #checa toda estrutura
            if 6 < len(rule) > 8: # manter compatibilidade com regras que não tem 'acertos' e 'acionamentos'
                print('ERROR: COD = {0}. Cada regra deverá conter: [código regra, tipo de relação temporal, ordem de execução, expressão lógica que representa a regra, origem, acuracia, acertos, acionamentos].\nValor informado: {1}'.format(rule[self.__COD_REGRA], rule))
                return False
            if type(rule[self.__COD_REGRA]) not in [int, float]:
                print('ERROR: COD = {0}. Código da regra deve ser numérico.'.format(rule[self.__COD_REGRA]))
                return False
            if rule[self.__RELTYPE] not in self.__tb.my_tlink.relType_validos: 
                print('ERROR: COD = {0}. Tipo da Relação Temporal inválida: {1}. Valores válidos: {2}.'.format(rule[self.__COD_REGRA], rule[self.__RELTYPE], str(self.__tb.my_tlink.relType_validos)))
                return False
            if type(rule[self.__ORDEM]) not in [int, float]:
                print('ERROR: COD = {0}. Ordem de processamento da regra inválido: {1}. Deve ser numérico.'.format(rule[self.__COD_REGRA], rule[self.__ORDEM]))
                return False
            if type(rule[self.__PREDICADOS]) != str:
                print('ERROR: COD = {0}. Expressão que representa a regra deve ser do tipo string.'.format(rule[self.__COD_REGRA]))
                return False
            if type(rule[self.__ORIGEM]) != str:
                print('ERROR: COD = {0}. Origem da regra deve ser do tipo string.'.format(rule[self.__COD_REGRA]))
                return False
            if type(rule[self.__ACURACIA]) not in [int, float]:
                print('ERROR: COD = {0}. Acurácia da regra deve ser do tipo float.'.format(rule[self.__COD_REGRA]))
                return False
            if len(rule) > 6: # manter compatibilidade com regras que não tem 'acertos' e 'acionamentos'
                if type(rule[self.__ACERTOS]) not in [int]:
                    print('ERROR: COD = {0}. Quantidade de acertos da regra deve ser do tipo inteiro.'.format(rule[self.__COD_REGRA]))
                    return False
            if len(rule) > 7:  # manter compatibilidade com regras que não tem 'acertos' e 'acionamentos'
                if type(rule[self.__ACIONAMENTOS]) not in [int]:
                    print('ERROR: COD = {0}. Quantidade de acionamentos da regra deve ser do tipo inteiro.'.format(rule[self.__COD_REGRA]))
                    return False
            
            try:
                if rule[self.__ORDEM] >= 0:  #somente as regras ativas
                    result = eval(rule[self.__PREDICADOS])
            except Exception as e:
                print('ERROR: COD = {0}. Expressão que representa a Regra não é válida: {1}. \nERRO: {2}'.format(rule[self.__COD_REGRA], rule[self.__PREDICADOS], e))
                return False

        if len(repetido) > 0:
            print('ERROR: Códigos de regras repetidos. CÓD REPETIDO:', repetido)
            return False

        return True

    
    def __trata_predict_votacao(self, predict: dict):
        '''
        Retorna o tipo de relação mais votado e o código da primeira regra do tipo mais votado.
        
        '''
        tipo_x_cods = defaultdict(list)
        tipo_x_quant = defaultdict(int)

        #Agrupa cod regras por tipo de relação
        for cod, tipo in predict.items():
            tipo_x_cods[tipo].append(cod)

        #Conta a quantidade de regras casadas para cada tipo de relação
        for tipo, cods in tipo_x_cods.items():
            tipo_x_quant[tipo] = len(cods)

        contador = Counter(tipo_x_quant)
        tipo_mais_votado = contador.most_common(1)[0][0]
        cods_rules = tipo_x_cods[tipo_mais_votado]
        
        return tipo_mais_votado, cods_rules[0]
    
    def relTypePar(self, tokenE: Token, tokenT: Token) -> str:
        '''
        Retorna tipo da relação temporal anotada no Corpus entre tokenE e tokenT, se houver.
        Busca informação do relType gravada no tokenE.
        Só recebe pares dos dados de treino.
        '''
        #Relações de todas as tasks gravadas em tokenE
        relacoes = tokenE._.relType
        #tid das relações de task 'A'
        tid_A = list(filter(lambda x: relacoes[x]['task'] == 'A', relacoes.keys()))
        id_tag_timex = tokenT._.id_tag
        if not relacoes or not tid_A:
            return ''
        list_A = {}
        for key in tid_A:
            relatedTo = relacoes[key]['relatedTo']
            relType = relacoes[key]['relType']
            list_A[relatedTo] = relType
        #print('\nDENTRO:', 'sent', tokenE._.id_sentenca, 'tid_A', tid_A, 'timex', id_tag_timex, 'list_A', list_A)
        return list_A[id_tag_timex]

    def __rule_triggers_update(self, codRule, relTypeRule, relTypePar):
        '''
        Atualiza estrutura de dados que armazena métricas das regras.
        É chamado durante o processamentos de regras por votação.
        '''
        if relTypePar: #só considera pares anotados
            self.__rule_triggers[codRule]['relTypeRule'] = relTypeRule
            self.__rule_triggers[codRule]['relTypePares'].append(relTypePar)

    def __rule_triggers_metrics(self) -> dict:
        '''
        Calcular métricas das regras com base do dados coletados por __rule_triggers_update()
        '''
        self.__rule_triggers.clear()

        print('Processando conjunto de regras...')
        self.task = 'A'
        self.processing_type = 'votacao'
        self.active_tlink_candidate  = False
        self.active_tlink_transitive = False
        self.process_rules()

        print(self.__rule_triggers)

        metrics = {}
        for codRegra, metricsRule in self.__rule_triggers.items():
            relTypeRule = metricsRule['relTypeRule']
            relTypePares = metricsRule['relTypePares']
            acertos = relTypePares.count(relTypeRule)
            acionamentos = len(relTypePares)
            acuracia = acertos / acionamentos if acionamentos > 0 else 0

            metrics[codRegra] = {'acertos': acertos, 'acionamentos': acionamentos, 'acuracia': acuracia}
        return metrics

    def __calculate_metrics_helper(self):
        '''
        Calcula métricas do conjunto de regras atribuido a TemporalRelation.rules.
        '''
        rules = self.rules
        if not rules:
            raise ValueError('ERRO: TemporalRelation.rules está vazio. Atribui um conjunto de regras a esta propriedade.')
        
        rule_triggers_metrics = self.__rule_triggers_metrics()

        #Atualiza métricas das regras
        for rule in rules:
            cod_regra = rule[0]
            if cod_regra in rule_triggers_metrics:
                acuracia = rule_triggers_metrics[cod_regra]['acuracia']
                acertos = rule_triggers_metrics[cod_regra]['acertos']
                acionamentos = rule_triggers_metrics[cod_regra]['acionamentos']
                rule[-3:] = [acuracia, acertos, acionamentos]
        print('Métricas atualizadas disponível em TemporalRelation.rules.')

    def calculate_metrics(self, setRules: SetRules = None):
        '''
        Calcula métricas do conjunto de regras.

        Args:
            setRules: Instância do objeto SetRules (opcional).
                Se informado calcula métricas de TemporalRelation.SetRules.rules e de TemporalRelation.rules.
                Se não informado calcula apenas de TemporalRelation.rules.
        '''
        if setRules:
            if not isinstance(setRules, SetRules):
                print('setRules deve ser uma instância de TemporalRelation.SetRules')
                return 
            
            if not setRules.rules:
                print('Não foi atribuido conjunto de regras para setRules.')
                return

            self.rules = setRules.rules
            self.__calculate_metrics_helper()
            
            setRules.clear()
            setRules.add_setRules_ojb('my_rules', self.rules, verbosy=False) 
            print('Métricas atualizadas e disponível também TemporalRelation.SetRules.rules.')
        else:
            self.__calculate_metrics_helper()


    def __predict_tlink_rules(self, tokenE: Token, tokenT: Token) -> tuple:
        '''
        Recebe todos os pares de possível relação temporal e retorna o tipo de relação baseado na regra.
        Chamado a cada par.

        Args:
            tokenE: token do tipo EVENT
            tokenT: token do tipo TIMEX3

        Return:
            Tupla contendo tipo da relação temporal predita (relType) e
            Código da regra que prediz a relação. 
            Se o tipo processamento (TemporalRelation.processing_type) for 'primeira_regra', ou o código da primeira regra do tipo de relação mais votado.

        '''
        verbosy = False
        if verbosy: print(('Rel Anotado:{5} -> {0:<5} E: {1} {2:<15} T: {3} {4:<15}').format(tokenE._.id_sentenca, tokenE._.id_tag, tokenE.text, tokenT._.id_tag, tokenT.text, self.relTypePar(tokenE, tokenT)))

        predict = defaultdict(str)
        
        for rule in self.rules:
            if eval(rule[self.__PREDICADOS]):
                if self.processing_type == 'votacao':
                    codRule = rule[self.__COD_REGRA]
                    relTypeRule = rule[self.__RELTYPE]
                    
                    #atualiza dados para cálculo de métricas das regras apenas para DADOS DE TREINO
                    if tokenE.doc._.train_test == 'train':
                        relTypePar = self.relTypePar(tokenE, tokenT)
                        self.__rule_triggers_update(codRule, relTypeRule, relTypePar)
                    
                    #predições de um par event-time (relType de todas as regras acionadas)
                    predict[codRule] = relTypeRule  #votacao

                if self.processing_type == 'primeira_regra':
                    return rule[self.__RELTYPE], rule[self.__COD_REGRA]  #primeira_regra

        if self.processing_type == 'votacao' and predict:
            #Retorna o tipo de relação mais votado e o código da primeira regra do tipo mais votado.
            if verbosy: print('ANTES:', predict)
            relTypePredict = self.__trata_predict_votacao(predict) #votação
            if verbosy: print('DEPOIS:', relTypePredict)
            
            if verbosy: print('rule_triggers:', self.__rule_triggers)
            return relTypePredict
            

    
    def __process_tlinks(self):
        '''
        Processa e salva TLINKs preditos em tb.my_tlinks,
        Calcula e salva TLINKs transitivos e seleciona TLINKs com maior probabilidade de estarem anotados no corpus.
        
        '''
        #Processa predição para cada sentenca
        for id_sentenca in self.__id_sentenca:
            #Atribui cada sentenca recebida como argumento para a classe TimebankPT, 
            #isso faz as funções abaixo responderem conforme a sentença selecionada
            self.__tb.id_sentenca = id_sentenca
            
            pares_relacoes_sentenca = {}
                        
            #Recebe os pares Event x Timex3 da sentença e 
            #Processa predição para cada par de entidade da sentenca
            for eventID, relatedTo in self.__tb.my_tlink.lista_id_timebank(self.task)['pares']:
                #converte ids de EVENT e TIMEX3 para Token
                token_eventID, token_relatedTo = self.convert_id_to_token(eventID, relatedTo) 
                
                #===================TLINKCANDIDATE===========================
                #Verifica se o EVENT é um candidato para geração de TLINK
                #Se não for retorna ao inicio do FOR
                if self.active_tlink_candidate:
                    #Alimenta self.df_features com atributos morfossintáticos e POS tagger
                    #Usado para alimentar modelo de ML de TLinkCandidate
                    self.__process_features(token_eventID, token_relatedTo)

                    if not self.__process_tlinks_candidate(token_eventID):
                        continue
                #============================================================
                
                #REALIZA A PREDIÇÃO
                #Retorna a tupla (tlink predito, código da regra que previu)
                predict_tlink_rules = self.__predict_tlink_rules(token_eventID, token_relatedTo)
                
                #Se salvar com sucesso o tlink predito pelas regras, acumula o tlink do par (EVENT, TIMEX3) em pares_relacoes_sentenca
                if self.__save_predict_tlink_rules(eventID, relatedTo, predict_tlink_rules):
                    #unpack tupla
                    relType_predict, _ = predict_tlink_rules
                    
                    #===================TLINKTRANSITIVE==================================
                    #Junta os pares da sentença para posterior cálculo da transitividade
                    if self.active_tlink_transitive:
                        pares_relacoes_sentenca[(eventID, relatedTo)] = relType_predict
                    #=====================================================================
                            
            #===================TLINKTRANSITIVE==================================
            #FOR dentro de id_sentenca
            #Calcula e salva novos pares transitivos em tb.my_tlink
            if self.active_tlink_transitive:
                #print(f'{id_sentenca} pares: {pares_relacoes_sentenca}')
                self.tlink_transitive.save_tlinks_transitive(pares_relacoes_sentenca)
            #====================================================================
        
        
    def __save_predict_tlink_rules(self, eventID: str, relatedTo: str, predict_tlink_rules: tuple):
        '''
        Salva a relação predita em tb.my_tlink
        
        Args:
            eventID: id do EVENT
            reletedTo: id do TIMEX3
            predict_tlink_rules: Tupla (relType, cod_rule) predito.
            
        Return:
            Retorna True se salvo com sucesso, se não, retorna False
            
        '''
        #Valida predict_tlink_rules
        if self.__check_predict_tlinks_rules(predict_tlink_rules):
            relType_predict, cod_rule_predict = predict_tlink_rules
            
            #Salva relação
            campos_add = relType_predict, eventID, relatedTo, self.task, self.__tb.id_sentenca_unica[0], self.__tb.nome_doc_unico, cod_rule_predict
            self.__tb.my_tlink.add(*campos_add)
            
            return True
        
        return False
        
        
    def __check_predict_tlinks_rules(self, predict_tlink_rules) -> bool:
        '''
        Verifica se o tlink predito pelas regras é válido.
        
        Args:
            predict_tlink_rules: Tupla (relType, cod_rule) predito.
            
        Return:
            Retorna True se for válido, se não, retorna False
            
        '''
        if not predict_tlink_rules:
            return False
        
        if type(predict_tlink_rules) not in [tuple, list] or len(predict_tlink_rules) != 2:
            print('ERROR: O retorno da Regra deve retornar tupla contendo: Tipo Relação, CódigoRegra. Retorno: {0}'.format(predict_tlink_rules))
            return False
        
        relType, cod_rule = predict_tlink_rules
        if relType not in self.__tb.my_tlink.relType_validos:
            print('ERROR: A REGRA {1} está passando tipo de relação temporal inválida: {0}. \nTipo válidos: {2}'.format(relType, cod_rule, self.__tb.my_tlink.relType_validos) )
            return False
            
        return True
    
    
    def __process_tlinks_candidate(self, tokenE: Token) -> bool:
        '''
        Seleciona EVENT candidato à geração de TLINK com maior probabilidade de estarem anotados no corpus.
        EVENT com menor probabilidade de não estarem anotados não serão processados, isso é, não serão preditos o tipo de relação temporal que ele tem com o TIMEX3.
        
        Utiliza regras ou modelo de machine learning pré-treinado, dependendo da propriedade TlinkCandidate.approach, que assume o valor 'REGRAS' ou 'ML'.
        
        Args:
            tokenE: token que representa o evento a ser analisado.
        
        '''
        
        #-----DECISION TREE------
        if self.tlink_candidate.approach == 'ML':
            precidt = self.tlink_candidate.predict(tokenE)
            return precidt
        
        #-----REGRAS MANUAIS------
        elif self.tlink_candidate.approach == 'REGRAS':
            #if self.f.pos(tokenE, ['VERB']) and self.f.dep(tokenE, ['ROOT', 'CCOMP', 'ADVCL', 'XCOMP', 'ACL', 'ACL:RELCL']):
            #    return True
            #if self.f.pos(tokenE, ['NOUN']) and self.f.dep(tokenE, ['OBJ', 'OBL']):
            #    return True
            
            if self.f.pos(tokenE, ['VERB']) or self.f.pos(tokenE, ['NOUN']):
                return True

        return False

    
    def process_rules(self, id_sentencas = None):
        '''
        Realiza predição de relações temporais em todos os pares entre EVENT e TIMEX3 de todas as sentenças, conforme a tarefa.
        Preenche estrutura 'self.__tb.my_tlink' com dados da predição.
        
        Args:
            id_sentencas: Se não for informado, processa todas as sentenças cobertas pela task atual.

        '''
        #VALIDAÇÕES
        if not self.task:
            print("ERROR: É necessário informar a tarefa através da propriedade 'task' da instancia da classe. \nTask válidas: A, B e C.")
            return 
        
        if not self.rules:
            raise ValueError('ERRO: É necessário atribuir um conjunto de regras a TemporalRelation.rules.')

        #Salva sentença atual
        id_sentenca_atual = self.__tb.id_sentenca
        
        #Define sentenças que serão processadas
        if id_sentencas:
            self.__id_sentenca = self.__tb.trata_lista(id_sentencas)
        else:
            self.__id_sentenca = self.__tb.id_sentencas_task(self.task) 
            
            
        #Limpa estrutura de dados que receberá as predições
        self.__tb.my_tlink.clear()

        #Adiciona TLINKs preditos em tb.my_tlink, 
        self.__process_tlinks()
        
        #Monta dataframe de valores preditos
        self.__df_pred = self.__tb.my_tlink.to_df
        
        #Seleciona campo conforme tipo da tarefa
        relatedTo = ''
        if self.task in ['A', 'B']:
            relatedTo = 'relatedToTime'
        elif self.task == 'C':
            relatedTo = 'relatedToEvent'
        else:
            print("ERROR: task '{0}' não existente".format(self.task))
            return
        
        #Organiza dataframe
        self.__df_pred = self.__df_pred[['lid', 'relType', 'eventID', relatedTo, 'task', 'isentenca', 'doc', 'rule']]
        self.__df_pred.rename(columns={'lid':'lid_pred', 'relType':'relType_pred', relatedTo:'relatedTo'}, inplace=True)
                
        #Restaura sentenca atual
        self.__tb.id_sentenca = id_sentenca_atual
    
    
    def __sort_rules_accuracy(self, show_accuracy: bool = False):
        '''
        <<FUNÇÃO NÃO MAIS UTILIZADA>>
        Retorna lista de cod_regra ordenada por acurácia.
        As regras são processadas individualmente, ou seja, todos os pares são processados por apenas uma regra de cada vez.
        É ineficiente, mas o objetivo é evitar interferências de uma regra sobre outra.
        
        O resultado poderá ser utilizado para processar o conjunto de teste em ordem de acurácia, ex: TemporalRelation.sort_rules(sort_rules_accuracy: list)
        Será implementado outra maneira mais eficiente no futuro.
        '''
        def trata_acuracia(lista):
            #Retorna a acurácia da lista
            if math.isnan(lista[1]):
                lista[1] = 0
            return lista[1]
        
        tempo_inicial = time.time() # em segundos

        rules = self.rules
        cod_rules_accuracy = []
        for rule in rules:
            self.rules = [rule]
            self.process_rules()
            cod_rules_accuracy.append([rule[0], self.df_resultado_por_regras['pct_acerto_anotado'].mean()])

        cod_rules_accuracy.sort(key = trata_acuracia, reverse = True)
        
        if not show_accuracy:
            cod_rules_accuracy = [cod_rule[0] for cod_rule in cod_rules_accuracy]
        
        self.rules = rules
        
        tempo_final = time.time() # em segundos
        print(f"Tempo processamento: {(tempo_final - tempo_inicial)/60} minutos")
        
        return cod_rules_accuracy
        
    
    @property
    def df_features(self):
        '''
        DESCONTINUADO.

        Dataframe contendo features dos tokens EVENT e TIMEX3 que poderão ser TLinks.
        Utilizado para análises manuais na seleção de TLinkCandidate 
        e para alimentar dados para treinamento do modelo de machine learning
        
        '''
        return self.__df_features
    
    def __process_features(self, tokenE: Token, tokenT: Token):
        '''
        DESCONTINUADO.

        Alimenta self.df_features com atributos sintáticos e POS tagger
        Usado em TLinkCandidate.
        '''
        #Salvar features de todas as relações do corpus
        eventID = tokenE._.id_tag
        relatedTo = tokenT._.id_tag
        id_sentenca = self.__tb.id_sentenca
        nome_doc = self.__tb.nome_doc_unico
        train_test = self.__tb.get_train_test(nome_doc)
        
        e_text  = tokenE.text
        t_text  = tokenT.text
        e_root  = self.f.dep(tokenE, 'ROOT')
        e_dep_t = self.f.dependencyType(tokenE, tokenT)
        e_pos   = tokenE.pos_
        t_pos   = tokenT.pos_
        e_class = tokenE._.classe

        e_dep_com_pai = tokenE.dep_
        e_pai_text = tokenE.head.text
        e_pai_pos = tokenE.head.pos_
        e_pai_ent = tokenE.head.ent_type_

        t_dep_com_pai = tokenT.dep_
        t_pai_text = tokenT.head.text
        t_pai_pos = tokenT.head.pos_
        t_pai_ent = tokenT.head.ent_type_

        e_dist_t = abs(tokenE.i - tokenT.i - 1)
        if e_dist_t in range(0,6):
            e_dist_t_desc = 'Muito Perto'
        elif e_dist_t in range(6, 11):
            e_dist_t_desc = 'Perto'
        elif e_dist_t in range(11, 16):
            e_dist_t_desc = 'Longe'
        elif e_dist_t >= 16:
            e_dist_t_desc = 'Muito Longe'

        anotado = False
        df = self.__tb.df.tlink_completo[['eventID', 'relatedToTime', 'doc', 'task', 'relType']]
        df = df[(df['eventID'] == eventID) & (df['relatedToTime'] == relatedTo) & (df['doc'] == nome_doc) & (df['task'] == 'A')]
        if not df.empty:
            anotado = True

        colunas={'train_test': train_test, 'anotado': anotado, 'task': self.task, 'eventID': eventID, 'e_text': e_text, 'relatedTo': relatedTo, 't_text': t_text, 'isentenca': id_sentenca, 'doc': nome_doc, 
                'e_class': e_class, 'e_root': e_root, 'e_pos': e_pos, 't_pos': t_pos, 'e_dep_t': e_dep_t, 
                'e_dep_com_pai': e_dep_com_pai, 'e_pai_text': e_pai_text, 'e_pai_pos': e_pai_pos, 'e_pai_ent': e_pai_ent, 
                't_dep_com_pai': t_dep_com_pai, 't_pai_text': t_pai_text, 't_pai_pos': t_pai_pos, 't_pai_ent': t_pai_ent, 'e_dist_t': e_dist_t, 'e_dist_t_desc': e_dist_t_desc}
        new = pd.DataFrame(colunas, index = [0])
        self.__df_features = pd.concat([self.__df_features, new])
    
        
    def id_sentencas_show_result(self, task: str):
        '''
        Retorna id_sentenca contempladas pela tarefa 'task' e filtrada conforme TemporalRelation.show_result 
        que consiste em selecionar dados de traino ou de teste para exibir o resultado
        
        Args:
            task: filtrar conforme task
        '''
        if not task:
            print('Task ainda não foi informada.')
            return 
        
        query = self.__tb.query_filtro_task('A') + ' and train_test == @self.show_result'
        return self.__tb.df.tlink_join_completo.query(query)['isentenca'].unique().tolist()
        
    def process_resume(self):
        '''
        <<Precisa desbagunçar isso aqui>>
        
        Exibe resultado do processamentos do processamento das regras
        
        '''
        if not self.__check_process_rules():
            #print('ERROR: Regras ainda não processadas. Execute o método process_rules().')
            return
        
        acertos = self.df_resultado_por_regras.Acertos.sum()
        total_anotadas = self.df_resultado_por_regras['Anotado'].sum()
        total_extras = self.df_resultado_por_regras['Não Anotado'].sum()
        total_geral = total_anotadas + total_extras
        pct_acerto_anotadas = '{0:,.1f}%'.format(acertos / total_anotadas * 100)
        pct_acerto = '{0:,.1f}%'.format(acertos / total_geral * 100)
        
        total_sentenca_task_a = len(self.id_sentencas_show_result('A'))
        total_sentenca_sem_predicao = len(self.id_sentencas_sem_predicao)
        total_relacoes_task_a = len(self.df_real) 
        total_relacoes_sem_predicao = total_relacoes_task_a - self.df_real_predict.relType_pred.value_counts().sum()  #total_relacoes_task_a - total_anotadas
        pct_cobertura = '{0:,.1f}%'.format(total_anotadas / total_relacoes_task_a * 100)
        
        quant_regras_processadas = self.df_resultado_por_regras.shape[0]
        
        print(f'RESUMO PROCESSAMENTO DO DADOS DE {self.show_result}')
        print(self.status_resumido_str())
        print('{0:<24} {1:>5} de {2:>7} ({3})'.format('Acurácia Anotadas:', acertos, total_anotadas, pct_acerto_anotadas))
        print('{0:<24} {1:>17}'.format('Total Não Anotadas:', total_extras))
        print('{0:<24} {1:>5} de {2:>7} ({3})'.format('Acurácia Não Anotadas:', acertos, total_geral, pct_acerto))
        
        print('{0:<25} {1:>5} de {2:>7,.1f} ({3})'.format('\nTAXA COBERTURA:', total_anotadas, total_relacoes_task_a, pct_cobertura))
        
        print('\nQuant Regras Processadas: ', quant_regras_processadas)
        print('Total Sentenças Anotadas Task A:  ', total_sentenca_task_a)
        print('Total Sentenças Anotadas sem predição: ', total_sentenca_sem_predicao)
        print('Total Relações Anotadas Task A: ', total_relacoes_task_a)
        print('Total Relações Anotadas sem predição: ', total_relacoes_sem_predicao)
        print('\n')
        
        
    @property
    def df_pred(self):
        '''
        Exibe em DataFrame as predição de relações temporais processadas pelo método process_rules().

        '''
        if not self.__check_process_rules():
            #return False
            return pd.DataFrame(data=None) 
        
        df = self.__df_pred
        df['train_test'] = df['doc'].apply(self.__tb.get_train_test)
        #Filtra para exibir resultado de treino ou de teste
        df = df.query("train_test == @self.show_result")
        
        return df.sort_values(['isentenca', 'eventID', 'relatedTo'])
        
    
    @property
    def df_real(self):
        '''
        Sentenças anotadas no corpus que atendem aos critérios de cada task.
        
        '''
        relatedTo = ''
        if self.task in ['A', 'B']:
            relatedTo = 'relatedToTime'
        elif self.task == 'C':
            relatedTo = 'relatedToEvent'
        else:
            print("ERROR: task '{0}' não existente".format(self.task))
            return
        
        #Filtra resultado conforme task
        query = self.__tb.query_filtro_task(self.task)
        df = self.__tb.df.tlink_join_completo.query(query + " and isentenca in " + str(self.__id_sentenca))
        
        #Renomeia colunas
        rename_colunas = {relatedTo:'relatedTo', 'relType':'relType_real', 'lid':'lid_real'}
        df = df.rename(columns=rename_colunas)
        #Seleciona colunas
        df = df[['lid_real', 'relType_real', 'eventID', 'relatedTo', 'task', 'isentenca', 'doc', 'train_test']]
        #Ordena colunas
        df = df.sort_values(['isentenca', 'eventID', 'relatedTo'])
        #Filtra para exibir resultado de treino ou de teste
        df = df.query("train_test == @self.show_result")
        return df 
    
    def df_real_predict_id_sentenca(self, id_sentenca = None, extra : bool = False):
        '''
        Une dados real do corpus com os dados previsto.
        
        Args:
            id_sentenca: Se informada, filtra por id_sentenca.
            extra: Se True, exibe as previsões que não estão anotadas no corpus, se False (default), exibe apenas as previsões que há correspondência no corpus.
            
        '''
        def esta_na_lista(relType_pred, relType_real):
            '''
            Calcula coluna 'acertou'.
            
            Args: 
                relType_pred = Tipo de relação prevista
                relType_real = Tipo de relação anotada
            '''
            if not relType_real or type(relType_real) != str:
                return False
            if relType_pred == relType_real:
                return True
            relType_real = relType_real.split('-OR-')
            return relType_pred in relType_real
        
        def relType_real_sem(relType_pred, relType_real, acertou):
            if relType_real in ['BEFORE-OR-OVERLAP','OVERLAP-OR-AFTER']:
                if acertou:
                    return relType_pred
            return relType_real
        
        
        if not self.__check_process_rules():
            return pd.DataFrame(data=None) 
        
        if extra:
            how = 'outer'
        else:
            how = 'left'
        
        #Une df_real com df_pred
        real_predict = self.df_real.merge(self.df_pred, how = how, on=['task', 'doc', 'train_test', 'isentenca', 'eventID', 'relatedTo'])
        
        #Calcula predições corretas
        real_predict['acertou'] = [ esta_na_lista(row[0], row[1]) for i, row in real_predict[['relType_pred', 'relType_real']].iterrows() ]
        real_predict['relType_real'] = [ relType_real_sem(row[0], row[1], row[2]) for i, row in real_predict[['relType_pred', 'relType_real', 'acertou']].iterrows() ]
        
        #Filtra resultado conforme task
        query = "task == '" + self.task + "'" 
        real_predict = real_predict.query(query)
        
        #Busca valores de eventID
        df_eventID  = self.__tb.df.event_completo[['doc', 'eid', 'text']].rename(columns={'eid':'eventID'})
        real_predict_join = real_predict.merge(df_eventID, how = 'left', on=['doc', 'eventID'])
        
        #Seleciona colunas
        colunas = ['lid_real', 'doc', 'train_test', 'isentenca', 'task', 'relType_real', 'eventID', 'text_event', 'relatedTo', 'text_relatedTo', 'value', 'relType_pred', 'rule', 'acertou']
            
        #Busca valores de relatedTo conforme task
        if self.task in ['A', 'B']:
            df_timex3 = self.__tb.df.timex3_completo[['doc', 'tid', 'text', 'value']].rename(columns={'tid':'relatedTo'})
            real_predict_join = real_predict_join.merge(df_timex3, how = 'left', on=['doc', 'relatedTo'], suffixes=('_event', '_relatedTo'))
            
        elif self.task == 'C':
            df_eventRT = df_eventID.rename(columns={'eventID':'relatedTo'}) 
            real_predict_join = real_predict_join.merge(df_eventRT, how = 'left', on=['doc', 'relatedTo'], suffixes=('_event', '_relatedTo'))
            #Não tem 'value' se task C
            colunas.remove('value')
        
        #Aplica seleção de colunas
        real_predict_join = real_predict_join[colunas]
        #Ordena colunas
        real_predict_join = real_predict_join.sort_values(['acertou', 'isentenca', 'eventID', 'relatedTo'])
        
        #Permite receber id_sentenca como argumento
        if id_sentenca:
            real_predict_join = real_predict_join.query("isentenca == " + str(id_sentenca))
        
        return real_predict_join
    
    @property
    def df_real_predict(self):
        '''
        Retorna dataframe com dados real do corpus e com dados previstos unidos.
        Apenas aquelas previsões em que há correspondência no corpus.
        '''
        return self.df_real_predict_id_sentenca(extra = False)
    
    @property
    def df_real_predict_extra(self):
        '''
        Retorna dataframe com dados real do corpus e com dados previstos unidos.
        Inclui também as previsões que não estão anotadas no corpus.
        '''
        return self.df_real_predict_id_sentenca(extra = True)

    @property
    def df_rules_applied_id_sentenca(self):
        '''
        Dataframe com as predições realizadas nas sentenças atribuídas a TimebankPT.id_sentenca
        '''
        id_sentenca = self.__tb.id_sentenca
        if not id_sentenca:
            raise ValueError('Não foi atribuído valor para TimebankPT.id_sentenca')
        
        return self.df_real_predict.query('isentenca == @id_sentenca')
    
    @property
    def explain_applied_rules(self):
        '''
        Explica as predições realizadas nas sentenças atribuídas a TimebankPT.id_sentenca
        '''
        print("{0:>4} {1:<18} {2}".format('COD', 'RelType', 'Regra') )
        for rule in self.get_rules(self.df_rules_applied_id_sentenca['rule'].tolist()):
            print("{0:>4} {1:<18} {2}".format(str(rule[0]), rule[1], rule[3].replace('self.f.', '')) )


    @property
    def y(self):
        #extras = predições não anotadas no corpus
        if self.show_extras:
            real_predict = self.df_real_predict_extra
        else:
            real_predict = self.df_real_predict
        
        #Todas predições que foram avaliadas por regras
        real_predict = real_predict[~real_predict['rule'].isna()]
        
        y_test = real_predict[['doc', 'eventID', 'relatedTo', 'relType_real']].set_index(['doc', 'eventID', 'relatedTo']).fillna('') #.tolist()
        y_pred = real_predict[['doc', 'eventID', 'relatedTo', 'relType_pred']].set_index(['doc', 'eventID', 'relatedTo']).fillna('')  #.tolist()
        return y_test, y_pred

    def save_results(self, nome_arquivo: str):
        '''
        Salva arquivo contendo dataset de resultados TemporalRelation.df_real_predict processado pelo método 'process_rules()'.
        Utiliza ext .parquet, mas não é necessário informá-la em nome_arquivo
        '''
        arquivo = self.__tb.check_filename(filename=nome_arquivo, extensao='parquet')
        try:
            self.show_result = 'all'
            self.df_real_predict.to_parquet(arquivo)
        except Exception as e:
            print(f'Erro ao salvar arquivo {arquivo} em save_results(). ERRO: {e}')
        else:
            print(f"Dataset salvo em {arquivo}")

    def load_results(self, nome_arquivo: str):
        '''
        Retorna dataset de resultados salvo pelo método 'save_results()'.
        Utiliza ext .parquet, mas não é necessário informá-la em nome_arquivo
        '''
        arquivo = self.__tb.check_filename(filename=nome_arquivo, extensao='parquet', check_if_exist = True)
        
        try:
            return pd.read_parquet(arquivo)
        except Exception as e:
            print(f'Erro ao carregar dataset de resultados {arquivo} em load_results(). ERRO: {e}')


    def relations_incorrect_class(self, train_test: Literal["train", "test"], *df_real_predicts: pd.DataFrame) -> pd.DataFrame:
        '''
        Recebe vários DataFrame df_real_predict, cada um representando um resultado de setRules diferente.
        Retorna DataFrame com a quantidade de todas as relações incorretas.
        Só considera dados rotulados.

        Args:
            train_test: informar quais dados serão exibidos: 'train' ou 'test'.
            df_real_predict: Vários DataFrames TimebankPT.TemporalRelation.df_real_predict contendo resultados do processamento das regras.
        '''
        
        if type(train_test) != str or train_test not in ['train', 'test']:
            #print("INFORMAR QUAIS DADOS SERÃO EXIBIDOS: 'train' ou 'test'")
            raise ValueError("ERROR: primeiro argumento deve ser 'train' ou 'test'.")
        print(f"EXIBINDO DADOS DE {'TREINO' if train_test == 'train' else 'TESTE'}")
        df_combinados = pd.concat(list(df_real_predicts), axis=0)
        df_combinados = df_combinados.query('train_test == @train_test and acertou == False and task == @self.task')
        df_combinados = df_combinados.groupby(['isentenca', 'doc', 'eventID', 'text_event', 'relatedTo', 'text_relatedTo', 'relType_real']).size().reset_index(name='incorretos').sort_values(['incorretos', 'isentenca', 'eventID'], ascending=[False, True, True])
        return df_combinados


    def graph_pred(self, id_sentenca = None):
        '''
        Exibe as relações temporais de forma gráfica da predições realizadas por este sistema.
        '''
        def cabecalho(isentenca):
            dct = self.__tb.dct_doc[0]
            nome = self.__tb.nome_doc[0]
            sentenca = self.__tb.sentenca_texto[0]
            texto = '\n-------------------------------------------------------------------\nDCT: {0}   DOC: {1}   ID_SENTENCA: {2} \n{3}\n'
            return texto.format(dct, nome, isentenca, sentenca)
            
            
        def graph_pred_helper(isentenca):
            pred_tlink = self.__tb.MyTlink(self.__tb)
            pred_tlink.clear()

            relType_true = self.df_real_predict.query("isentenca == " + str(isentenca))
            relType_true_com_predicao = relType_true[~relType_true['relType_pred'].isna()]
            for index, row in relType_true_com_predicao.iterrows():
                pred_tlink.add(row['relType_pred'], row['eventID'], row['relatedTo'], 'A', row['isentenca'], row['doc'], row['rule'])
            
            print(cabecalho(isentenca))
            pred_tlink.graph_rt()
            display(relType_true)
        
        
        if id_sentenca is None:
            id_sents = self.__tb.id_sentenca
            for_id_sents = id_sents
        else:
            self.__id_sentenca_anterior = self.__tb.id_sentenca
            self.__tb.id_sentenca = self.__tb.trata_lista(id_sentenca)
            for_id_sents = self.__tb.id_sentenca

        
        for id_sent in for_id_sents:
            self.__tb.id_sentenca = id_sent
            graph_pred_helper(id_sent)

            
        if id_sentenca is None:
            self.__tb.id_sentenca = id_sents
        else:
            self.__tb.id_sentenca = self.__id_sentenca_anterior
        

    def cm(self):
        '''
        Matrix de Confusão das predições considerando o contexto:
        self.show_result, self.processing_type, self.active_tlink_transitive
        '''
        y_test, y_pred = self.y
        
        #Uni label únicos de ambos dataset
        label = list(set(y_test['relType_real'].unique().tolist() + y_pred['relType_pred'].unique().tolist()))
        cm = confusion_matrix(y_test, y_pred, labels=label)
        df_cm = pd.DataFrame(cm, index = label, columns = label)

        plt.figure(figsize=(4, 3))
        sns.heatmap(df_cm, cmap='PuBu', annot=True, fmt="d")
        plt.title('Matriz de confusão')
        plt.xlabel('__________ PREDITO __________')
        plt.ylabel('____________ REAL ____________')
        print('\n')

        print(classification_report(y_test, y_pred, zero_division=0, digits=3))
        
        
    def ct(self):
        '''
        Cross Tab dos resultados do processamento das regras considerando o contexto:
        self.show_result, self.processing_type, self.active_tlink_transitive
        '''
        y_test, y_pred = self.y
        
        y_test = y_test.reset_index()['relType_real']
        y_pred = y_pred.reset_index()['relType_pred']

        return pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Pred'], margins=True)


    def metrics(self):
        '''
        Resultado do processamento das regras considerando o contexto:
        self.show_result, self.processing_type, self.active_tlink_transitive
        '''
        return {'f1_score'       : round(f1_score(*self.y, average='weighted'), 5), 
                'precision_score': round(precision_score(*self.y, average='weighted'), 5),
                'recall_score'   : round(recall_score(*self.y, average='weighted'), 5),
                'accuracy'       : round(accuracy_score(*self.y), 5),
                'support'        : classification_report(*self.y, output_dict=True)['weighted avg']['support']
                }
    
    @property
    def f1_score(self):
        '''
        Resultado do processamento das regras considerando o contexto:
        self.show_result, self.processing_type, self.active_tlink_transitive
        '''
        return self.metrics()['f1_score']
    
    @property
    def precision_score(self):
        '''
        Resultado do processamento das regras considerando o contexto:
        self.show_result, self.processing_type, self.active_tlink_transitive
        '''
        return self.metrics()['precision_score']

    @property
    def recall_score(self):
        '''
        Resultado do processamento das regras considerando o contexto:
        self.show_result, self.processing_type, self.active_tlink_transitive
        '''
        return self.metrics()['recall_score']
    
    @property
    def accuracy(self):
        '''
        Resultado do processamento das regras considerando o contexto:
        self.show_result, self.processing_type, self.active_tlink_transitive
        '''
        return self.metrics()['accuracy']

    @property
    def support(self):
        '''
        Resultado do processamento das regras considerando o contexto:
        self.show_result, self.processing_type, self.active_tlink_transitive
        '''
        return self.metrics()['support']
    
    


    #FIXME: colocar resultados em classe
    @property
    def quant_anotado(self):
        '''Quantidade de predições, considerando apenas dados rotulados'''
        df = self.df_real_predict
        df = df[~df['relType_pred'].isna()]
        return df.shape[0]
        
    @property
    def quant_acerto(self):
        '''Total de classificações corretas, considerando apenas dados rotulados'''
        df = self.df_real_predict
        return df[df['acertou'] == True].shape[0]
    
    @property
    def quant_erro(self):
        '''Total de classificações incorretas, considerando apenas dados rotulados'''
        return self.quant_anotado - self.quant_acerto
    
    @property
    def quant_nao_anotado(self):
        '''Quantidade de predições, considerando apenas dados não rotulados'''
        df = self.df_real_predict_extra
        return df[df['relType_real'].isna()].shape[0]
    
    @property
    def pct_acerto(self):
        '''Taxa de classificações corretas, considerando também dados não rotulados'''
        if (self.quant_anotado + self.quant_nao_anotado) == 0:
            return 0
        return self.quant_acerto / (self.quant_anotado + self.quant_nao_anotado) * 100
    
    @property
    def pct_acerto_anotado(self):
        '''ACURÁCIA: Taxa de classificações corretas, considerando apenas dados rotulados'''
        if self.quant_anotado == 0:
            return 0
        return self.quant_acerto / self.quant_anotado * 100
    
    @property
    def pct_erro(self):
        '''Taxa de classificações incorretas, considerando apenas dados rotulados'''
        if self.quant_anotado == 0:
            return 0
        return self.quant_erro / self.quant_anotado * 100

    @property
    def pct_cobertura(self) -> float:
        '''Taxa de cobertura'''
        total_relacoes_task_a = len(self.df_real) 
        return self.quant_anotado / total_relacoes_task_a * 100
        


    @property
    def df_resultado_geral(self):
        '''
        Retorna resumo do resultado geral.
        
        '''
        #Cria datasets
        df_resultado_geral = pd.DataFrame(data=None, columns = ['Erros', 'Acertos', 'Anotado', 'Não Anotado', '% Acerto', '% Acerto Anotado'])
        
        quant_total = self.quant_anotado
        quant_acerto = self.quant_acerto
        quant_previsao_extra = self.quant_nao_anotado
        quant_erro = self.quant_erro 
        pct_acerto = self.pct_acerto
        pct_acerto_anotado = self.pct_acerto_anotado
        #pct_erro = self.pct_erro

        df_total = pd.DataFrame({'Erros':quant_erro, 'Acertos': quant_acerto, 'Anotado': quant_total, 'Não Anotado': quant_previsao_extra, '% Acerto':'{0:,.2f}%'.format(pct_acerto), '% Acerto Anotado':'{0:,.2f}%'.format(pct_acerto_anotado)}, index=[0])
        df_resultado_geral = pd.concat([df_resultado_geral, df_total], axis=0, ignore_index=True) 
                
        return df_resultado_geral
    
    def __df_resultado_gererico(self, indice: list):
        '''
        Resultado das predições das relações temporais
        
        Args:
            indice: lista dos campos que compõem o índice do dataframe
        '''
        #Constrói df vazio para não faltar campos caso o resultado não contenha todos os campos
        df0 = pd.DataFrame(data=None, columns=[*indice, 'Erros', 'Acertos'])
        df0 = df0.set_index([*indice])

        #Data inicial agrupado
        df_extra = self.df_real_predict_extra
        if not df_extra.empty:
            df_extra_pred = df_extra[~df_extra['relType_pred'].isna()] #somente os que houve predição (SÓ EXTRA + SÓ TOTAL)
            df1 = df_extra_pred.groupby([*indice, 'acertou'])['lid_real'].count().unstack().fillna(0)
            df1 = df1.rename(columns={False:'Erros', True:'Acertos'})
            #df_resultado = df0.append(df1).fillna(0) #obsoleto
            df_resultado = pd.concat([df0, df1], axis = 0).fillna(0)
        else:
            df_resultado = df0.fillna(0)

        #Calcula o campo Anotado (Total de anotado)
        df_total = df_resultado.sum(axis=1).reset_index().rename(columns={0:'Anotado'})
        df_total = df_total.set_index([*indice])

        #Uni o campo Anotado do dataframe principal
        df_final = df_resultado.merge(df_total, how='left', on=[*indice])
        df_final.columns.set_names(names='Resultado', inplace=True)

        if df_extra.empty:
            df_final['Não Anotado'] = 0

        #NÃO ANOTADO
        #Acrescenta a quantidade de previsões que não estão anotadas no corpus: campo Não Anotado
        if not df_extra.empty:
            df_total_extra = df_extra[df_extra['relType_real'].isna()]
            df_total_extra = df_total_extra.groupby([*indice, 'acertou'])['isentenca'].count().unstack().fillna(0)
            df_total_extra = df_total_extra.rename(columns={False:'Não Anotado'})

            #CRIAR NÃO ANOTADO VAZIO
            df_final_com_extra = pd.DataFrame(data=None, columns=[*indice, 'Erros', 'Acertos', 'Anotado', 'Não Anotado'])
            df_final_com_extra = df_final_com_extra.set_index([*indice])

            #df_final_com_extra = df_final_com_extra.append(df_final.merge(df_total_extra, how='left', on=[*indice]) ) #obsoleto
            
            df_final_com_extra = pd.concat( [df_final_com_extra, df_final.merge(df_total_extra, how='left', on=[*indice])] )
            df_final_com_extra.columns.set_names(names='Resultado', inplace=True)

            colunas = ['Erros', 'Acertos', 'Anotado', 'Não Anotado']
            df_final_com_extra = df_final_com_extra[colunas].fillna(0)
            df_final = df_final_com_extra

        #--------------------------------
        #Calcular % de Acertos
        df_final['pct_acerto'] = df_final['Acertos'] / (df_final['Anotado'] + df_final['Não Anotado']) * 100
        df_final['pct_acerto_anotado'] = df_final['Acertos'] / df_final['Anotado'] * 100  

        #Ordena por maiores acertos
        df_final.sort_values(['pct_acerto_anotado', *indice], ascending=False, inplace=True)
        #df_final = df_final.sort_index()

        #Formata em percentual. Deve executar após ordenação, pois a formatação converte 'pct_acerto' em string
        df_final['pct_acerto'] = df_final['pct_acerto'].map(lambda x: round(x, 2))
        df_final['pct_acerto_anotado'] = df_final['pct_acerto_anotado'].map(lambda x: round(x, 2))

        return df_final.fillna(0)
    
    
    @property
    def df_resultado_por_sentenca(self):
        indice = ['doc', 'isentenca']
        return self.__df_resultado_gererico(indice = indice)
    
    @property
    def df_resultado_por_documento(self):
        indice = ['doc']
        return self.__df_resultado_gererico(indice = indice)
    
    @property
    def df_resultado_por_regras(self):
        indice = ['task', 'rule', 'relType_pred']
        return self.__df_resultado_gererico(indice = indice)
    
    @property
    def df_resultado_por_classe(self):
        indice = ['task', 'relType_pred']
        return self.__df_resultado_gererico(indice = indice)
    
    @property
    def df_resultado_por_task(self):
        indice = ['task']
        return self.__df_resultado_gererico(indice = indice)

    @property
    def df_resultado_regras_por_classe(self):
        return self.df_real_predict.value_counts(['rule', 'relType_real']).sort_index().unstack().fillna(0)

    #---------------------------------------------------------------------
    #     FIM TEMPORALRELATION
    #--------------------------------------------------------------------


#=========================================================================================================================
#=========================================================================================================================
# '##::: ##:'########:'##:::::'##:::::'######::'##::::::::::'###:::::'######:::'######:: #
#  ###:: ##: ##.....:: ##:'##: ##::::'##... ##: ##:::::::::'## ##:::'##... ##:'##... ##: #
#  ####: ##: ##::::::: ##: ##: ##:::: ##:::..:: ##::::::::'##:. ##:: ##:::..:: ##:::..:: #
#  ## ## ##: ######::: ##: ##: ##:::: ##::::::: ##:::::::'##:::. ##:. ######::. ######:: #
#  ##. ####: ##...:::: ##: ##: ##:::: ##::::::: ##::::::: #########::..... ##::..... ##: #
#  ##:. ###: ##::::::: ##: ##: ##:::: ##::: ##: ##::::::: ##.... ##:'##::: ##:'##::: ##: #
#  ##::. ##: ########:. ###. ###:::::. ######:: ########: ##:::: ##:. ######::. ######:: #
# ..::::..::........:::...::...:::::::......:::........::..:::::..:::......::::......::: #
#=========================================================================================================================
#=========================================================================================================================


###------------------------------------------------------
#        PIPELINE SPACY - TAGS TIMEBANKPT
###------------------------------------------------------
@Language.factory("pipe_timebankpt")
def pipe_timebankpt(tb_dict: dict, nlp: Language, name: str):
    return PipeTimebankPT(tb_dict, nlp)

class PipeTimebankPT():
    '''
    Implementa pipeline do spaCy que processa anotações do corpus TimeBankPT trazendo para a estrutura do spaCy as tag EVENT e TIMEX3.
    Adicionar antes do pipe nlp.add_pipe("merge_entities")

    Arg:
        tb_dict: Dicionário de dados contendo dados do TimebankPT necessários para este pipeline do spaCy.

    Return: 
        Objeto Doc do spaCy processado contendo as tag do TimebankPT (EVENT e TIMEX3).
        
    '''
    def __init__(self, tb_dict: dict, nlp: Language):
        
        self.__tb_dict = tb_dict
        
        #Regista atributos das tags DCT ao Doc
        if not Doc.has_extension("dct"):
            Doc.set_extension("dct", default='')
        if not Doc.has_extension("nome"):
            Doc.set_extension("nome", default='')
        if not Doc.has_extension("id_sentenca"):
            Doc.set_extension("id_sentenca", default='')
        if not Doc.has_extension("dct_tid"):
            Doc.set_extension("dct_tid", default='')
        if not Doc.has_extension("dct_type"):
            Doc.set_extension("dct_type", default='')
        if not Doc.has_extension("train_test"):
            Doc.set_extension("train_test", default='')

        #Regista atributos das tags EVENT e TIMEX3 ao Token
        if not Token.has_extension("id_sentenca"):
            Token.set_extension('id_sentenca', default='')
        if not Token.has_extension("id_tag"):
            Token.set_extension('id_tag', default='')
        
        #Regista atributos apenas de EVENT ao Token
        if not Token.has_extension("classe"):
            Token.set_extension('classe', default='')
        if not Token.has_extension("aspecto"):
            Token.set_extension('aspecto', default='')
        if not Token.has_extension("polarity"):
            Token.set_extension('polarity', default='')
        if not Token.has_extension("tense"):
            Token.set_extension('tense', default='')
        if not Token.has_extension("pos"):
            Token.set_extension('pos', default='')
        if not Token.has_extension("relType"):
            Token.set_extension('relType', default='')
        
        #Regista atributos apenas de TIMEX3 ao Token
        if not Token.has_extension("tipo"):
            Token.set_extension('tipo', default='')
        if not Token.has_extension("value"):
            Token.set_extension('value', default='')
            
        if not Token.has_extension("value_group"):
            Token.set_extension('value_group', default='')
        if not Token.has_extension("anchorTimeID"):
            Token.set_extension('anchorTimeID', default='')
        if not Token.has_extension("temporalFunction"):
            Token.set_extension('temporalFunction', default='')

            
            
    def __call__(self, doc: Doc) -> Doc:
        '''
        Retorna Doc processado contendo informações do TimebankPT.
        
        '''
        def relType_event(lista_tlink, eid):
            '''
            Retorna todas as relações do evento 'eid'.
            
            return {'l34': {'tid': 't225', 'task': 'A', 'relType': 'AFTER'},
                    'l35': {'tid': 't227', 'task': 'A', 'relType': 'AFTER'}
                    }
            '''
            #Índice das colunas
            tag            = 0
            lid            = 1
            task           = 2
            relType        = 3
            eventID        = 4
            relatedToTime  = 5
            relatedToEvent = 6
            doc            = 7
            
            tlinks = {}
            for item in list(filter(lambda x : x[eventID] == eid, lista_tlink)):  #filtra por eid
                rel = {}
                if item[task] in ['A', 'B']:
                    rel['relatedTo'] = item[relatedToTime]
                elif item[task] == 'C':
                    rel['relatedTo'] = item[relatedToEvent]
                    
                rel['task']    = item[task]
                rel['relType'] = item[relType]
                tlinks[item[lid]] = rel
                
            return tlinks
        
        #Se a sentença não existir no TimeBankPT, sai desse pipe e retorna o Doc original.
        #Mas os atributos devem ser registrados em __init__ para evitar erro ao serem chamados externamente
        dados_sentenca = self.__tb_dict.get(doc.text)

        if not dados_sentenca:
            return doc
        
        id_sentenca  = dados_sentenca['isentenca']
        dct_doc      = dados_sentenca['dct']
        nome_doc     = dados_sentenca['doc'] 
        dct_tid      = dados_sentenca['tid'] 
        dct_type     = dados_sentenca['type'] 
        train_test   = dados_sentenca['train_test']
        lista_event  = dados_sentenca['lista_event']
        lista_timex3 = dados_sentenca['lista_timex3']
        lista_tlink  = dados_sentenca['lista_tlink']
        
        ent_timebank = []
        ent_doc = []

        #Atribui valor para os novos atributos do Doc
        doc._.dct = dct_doc
        doc._.nome = nome_doc
        doc._.id_sentenca = id_sentenca
        doc._.dct_tid = dct_tid
        doc._.dct_type = dct_type
        doc._.train_test = train_test

        with doc.retokenize() as retokenizer:

            #EVENT
            for tag, isentenca, eid, text, start, end, classe, aspecto, tense, pos, polarity in lista_event:

                #Se text já estiver contido em alguma Entidade, esta entidade deverá ser removida
                #isso dará prioridade à entidades do tipo EVENT
                encontrou, ent = self.__find_substring_in_list_span(text, doc.ents)
                if encontrou:
                    ents = list(doc.ents)
                    ents.remove(ent)
                    doc.ents = tuple(ents)

                    #print('id:', isentenca, tag, '  Ent del:', ent)

                span = doc.char_span(start, end, 'EVENT') 
                if span:
                    #Adicionar EVENT a uma lista para adicionar às ENTIDADES
                    ent_timebank.append(span)
                    #Junta tokens formados por mais de uma palavra
                    retokenizer.merge(span)
                    #Atribui as características do EVENT ao token
                    for token in span:
                        token._.id_sentenca = isentenca
                        token._.id_tag = eid
                        token._.classe = classe
                        token._.aspecto = aspecto
                        token._.polarity = polarity
                        token._.tense = tense
                        token._.pos = pos
                        #todas as relações do evento 'eid'
                        token._.relType = relType_event(lista_tlink, eid)

                #else:
                    #print('id:', isentenca, tag,  '  None span:', text)


            #TIMEX3
            for tag, isentenca, tid, text, start, end, tipo, value, value_group, anchorTimeID, temporalFunction in lista_timex3:

                #Se text já estiver contido em alguma Entidade do spaCy, esta entidade deverá ser removida
                #isso dará prioridade à entidades do tipo TIMEX3
                encontrou, ent = self.__find_substring_in_list_span(text, doc.ents)
                if encontrou:
                    ents = list(doc.ents)
                    ents.remove(ent)
                    doc.ents = tuple(ents)

                    #print('id:', isentenca, tag, '  Ent del:', ent)

                span = doc.char_span(start, end, 'TIMEX3')
                if span:
                    #Adicionar TIMEX3 a uma lista para adicionar às ENTIDADES
                    ent_timebank.append(span)
                    #Junta tokens formados por mais de uma palavra
                    retokenizer.merge(span)
                    #Atribui as características do TIMEX3 ao token
                    for token in span:
                        token._.id_sentenca = isentenca
                        token._.id_tag = tid
                        token._.tipo = tipo
                        token._.value = value
                        
                        token._.value_group = value_group
                        token._.anchorTimeID = anchorTimeID
                        token._.temporalFunction = temporalFunction
                        
                #else:
                    #print('id:', isentenca, tag, '  None span:', text)

                #print('2: ', 'span:', span, '  text:', text, start, end, doc.ents, '  find:', self.__find_substring_in_list_span(text, doc.ents), ' char_span:', doc.char_span(start, end, 'EVENT') )

            #ADICIONA ENTIDADES
            #Prioriza as novas entidades EVENT e TIMEX3
            #Retira entidades de doc.ents se ela ou parte dela estiver no timebank
            for ent in doc.ents:
                encontrou_ents = False
                encontrou_timebank = False
                #adiciona ent a ent_doc se ent_timebank_item não estiver em doc.ents
                for ent_timebank_item in ent_timebank: 
                    has_substring, frase_lista = self.__find_substring_in_list_span(ent_timebank_item.text, doc.ents)
                    if has_substring:
                        encontrou_timebank = True
                        break
                for doc_ent_item in doc.ents:
                    has_substring, frase_lista = self.__find_substring_in_list_span(doc_ent_item.text, ent_timebank)
                    if has_substring:
                        encontrou_ents = True
                        break

                if (not encontrou_timebank) and (not encontrou_ents):
                    ent_doc.append(ent)

                #else:
                    #print('id:', doc._.id_sentenca, ent.label_, '  Ent not add:', ent.text)

            ent_doc.extend(ent_timebank)
            doc.ents = tuple(ent_doc)

        return doc

    def __find_substring_in_list_span(self, substring: str, list_span):
        '''
        Localiza uma substring ou palavra em uma lista de string

        Args:
            substring: texto que deseja localizar
            list_span: Lista de Span onde o texto será localizado.

        Returns: 
            Booleano do resultado da busca e a 

            Frase/palavra da lista de Span onde a subtring foi encontrada, em formato Span

        '''
        for span in list_span:
            if substring.find(' ') >= 0:  # >=0 se encontrar
                if span.text.find(substring) >= 0: 
                    return True, span
            for palavra in span.text.split():
                if palavra == substring:
                    return True, span
        return False, ''

    
#---------------------------------------------------------
#  Fim class - PipeTimebankPT
#---------------------------------------------------------
