'''
Autor: Dárcio Santos Rocha® 
e-mail: darcio.rocha@ufba.br
Mestrando em Ciências da Computação - UFBA
Agosto/2021

Importa tags do corpus TimebankPT para a estrutura do spaCy.

'''
#----------------------------------------------------------
#INSTALAÇÕES

#conda install -c conda-forge spacy
#conda install tabulate
#python -m spacy download pt_core_news_lg
#!python -m spacy info
#----------------------------------------------------------
'''
TODO:
    . Verificar a importancia de noun_chunks: [(p.text, p.label_) for p in tb.doc[0].noun_chunks]
    
    . No pipe para atribuir relações temporais, adicionar os atributos aos tokens
        . token._.relatedTo = []
        . token._.relType = []
        . token._.task = []
        Ex:
        #relatedTo ['t158', 't161']
        #relType ['after', 'before']
        #task = ['A', 'A']
        
    . Adaptar parse para o TimeBank em inglês
    . verificar se \b é melhor que \W na func search -> reg = "(\W|^)(" + tratar_palavras_busca(palavras) + ")(\W|$)"
    
    . The last feature in this category is the Temporal Relation between the Document Creation Time and the Temporal Expression in the target sentence. 
    . The value of this feature could be “greater than”, “less than”, “equal”, or “none”. 
    . Adicionar informação de train/test na estrutura de dados
    
    . ele primeiro ordena expressões temporais anotadas de acordo com seu valor normalizado (por exemplo, a data 1989-09-29  é ordenada como precedendo  1989-10-02). 
      Ou seja, exploramos as anotações timex3 a fim de enriquecer o conjunto de relações temporais com as quais trabalhamos, e mais especificamente fazemos uso do atributo de valor dos elementos TIMEX3.  


#SALVAR SAIDA DO DISPLACY EM ARQUIVO HTML
html = spacy.displacy.render(doc, style='ent', jupyter=False, page=True)
f = open("teste.html", "w", encoding='utf-8')
f.write(html)
f.close()
'''


import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
pd.set_option('max_columns', None)
pd.set_option('max_colwidth', 200)
#pd.set_option('max_rows', 40)
#pd.set_option("colheader_justify", "left")  # Não funcionou
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import spacy
from spacy.language import Language
from spacy.tokens import Token, Span, Doc

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import pickle  #save model

#Visualização Decision Tree
from IPython.display import Image  
from sklearn.tree import export_graphviz
#!pip install pydotplus
#!conda install graphviz
import pydotplus
from tabulate import tabulate
from treelib import Node, Tree

import os
import sys
import io
import math
import time
import random
import xml.etree.ElementTree as et 
import glob
import re
import html
from itertools import product, combinations
from collections import defaultdict, Counter
import types


class Functions:
    '''
    Funções genéricas utilizadas em outras classes
    
    '''
    def explicar_spacy(self, elemento):
        '''
        Retorna descrição explecativa sobre elementos POS e DEP do spaCy.
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
    
    
    def train_test(self, nome_arquivo: str) -> bool:
        '''
        Retorna se o arquivo é de teste ou de treino, baseado no subdiretório onde cada tipo está armazenado.

        Args:
            nome_arquivo: nome do arquivo que represente um documento do corpus.

        '''
        nome_inverso = nome_arquivo[::-1]
        ini = nome_inverso.find('\\')
        fim = nome_inverso.find('\\', ini + 1)
        tipo = nome_inverso[fim-1:ini:-1].lower()
        
        return tipo
    
    
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
        dep    : ávore de dependência
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
        
        #Featues em ordem de frequencia para testar o formato da árvore
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
        Obtem a chave do dicionário 'tipo_siglas' passando o valor como argumento.
        
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
        Obtem o valor da chave do dicionário 'tipo_siglas' passando a chave como argumento.
        
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
    Importa dados dos arquivos do corpus TimebankPT e fornece vários métodos para manipular o conteúdo do corpus.
    
    Args:
        path_tml: caminho do corpus TimebankPT no formato: 'c:\diretorio\*\*.txt'
        add_timebank: adiciona tags (EVENT E TIMEX3) do corpus TimebankPT ao pipeline do spaCy. Default é True
        
    '''
    def __init__(self, path_tml, add_timebank = True, lang = 'pt'):

        if not os.path.exists(os.path.dirname(path_tml).replace('*', '')):
            print('ERROR: Path dos arquivos .tml não existe.\n' + path_tml)
            return
        
        self.path_tml = path_tml
        
        #O self do parâmetro é para passar a class TimebankPT para Df, Print e MyTlink
        self.df = self.Df(self)
        self.print = self.Print(self)
        self.my_tlink = self.MyTlink(self)
        
        self.__dados_pipe = None
        self.__id_sentenca = []
        self.__sentenca_texto = []
        self.__nome_doc = None
        self.__dct_doc = None
        self.__train_test = None
        self.__lingua = 'PORTUGUÊS'
        
        #objeto Doc do spaCy
        self.__doc = None
        
        self.siglas = Siglas()
        
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
                "\n   . TLink: " + str(self.df.quant_tlink_total )
    
    
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
            
    def add_pipe_timebank(self):
        '''
        Adiciona o pipe que adiciona tags dos timebankPT ao Doc no spaCy
        
        '''
        if not self.nlp.has_pipe("pipe_timebankpt"):
            insere_antes = 'merge_entities'
            if not self.nlp.has_pipe(insere_antes):
                self.nlp.add_pipe(insere_antes)
            
            #Recupera dados do TimebankPT para serem fornecidos ao pipeline 
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
        Caso a sentenca passada exista no TimabankPT, atribui a id_sentenca à classe com set_id_sentenca()
        
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
        else:  # se não, o pipe processa a sentanca passada e atualiza o objeto Doc
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
        
        sentenca_texto = self.sentenca_texto    #self.sentenca_texto é atribuido em set_id_setenca() ou em set_sentenca_texto()
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
        Retorna lista com a primeira id_sentenca da lista de senteças setadas em set_id_sentenca()
        
        '''
        id_sentenca = self.id_sentenca
        
        if len(id_sentenca) == 0:
            return []
        
        id_sentenca_list = []
        id_sentenca_list.append(id_sentenca[0])
        id_sentenca = id_sentenca_list
        return id_sentenca
    
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
        Retorna query que filtra sentencas conforme task
        
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

  
    def id_sentencas_task(self, task):
        '''
        Retorna id_sentenca contempladas com cada tipo de task
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
            id_sentenca: sobrepõe id_sentenca atribuida em set_id_sentenca()
            
            todas_do_documento: Se True, retorna o texto de todas as sentenças do documento de get_id_sentenca(). Se False, retorna apenas o texto da sentença de get_id_sentenca()
                Se True e id_sentenca for uma lista com sentenças pertencentes a mais de um documento, concidera apenas as senteças do documento do primeiro item da lista id_sentenca

            com_tags: Se True, retorna o texto da sentença com as tags TimeML. Se False, retorna texto puro.

        Return:
            lista de sentencas
            
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
            
            formato_dataframe: se True, retorna o dataframe filtrado por lista_termos, se não, retorna lista de sentencas que atendem ao critério de pesquisa
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
            print('É necessário atribuir id_sentenca à instancia da classs TimebankPT.')
            return
        
        if type(self.doc) == list:
            doc = self.doc[0]
            
        if type(doc) == Doc:
            return doc
        
    
    def get_doc_root(self, doc: Doc = None) -> Token:
        '''
        Retorn o root do Doc.
        Se root for pontuação, verificar se não há outro root que não seja pontuação.
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
        
        list_roots = []
        for root in roots:
            if not (root.is_punct or root.is_bracket):
                list_roots.append(root)
        
        return list_roots[0]
        
    
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
        Retorna lista com nome do documento da id_sentença setada em set_id_sentenca() ou str da id_setenca do parametro
        
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
    
    
    def get_dct_doc(self, nome_doc = None):
        '''
        Retorna lista com Data de Criação do Documento(DCT) da id_sentença setada em set_id_sentenca() ou str da id_sentenca do parametro
        
        Args:
            nome_doc: se informado, sobrepõe as id_sentenca atribuida em set_id_sentenca()
        
        Return:
            lista com DCT.
            
        '''
        if nome_doc is None:
            if self.id_sentenca:
                return self.__dct_doc
            else:
                #print('ERROR: É necessário atribuir id_sentenca à instancia da class TimebankPT.')
                return []
        else:
            return self.df.documento_completo.query("doc == '" + str(nome_doc) + "'")['dct'].tolist()[0]
        
    def get_train_test(self, id_sentenca):
        '''
        Retorna qual o grupo de desenvolvimento que a sentença pertença.
        
        Args:
            id_sentenca: id da sentença
        
        Return:
            train:      Se é uma sentença de treino.
            train_test: Se é uma sentença de teste para o conjunto de treino (Dev).
            test:       Se é uma sentença de teste global. Utilizado apenas no trabalho final.
            
        '''
        return self.df.sentenca_completo.query("isentenca == " + str(id_sentenca) + "")['train_test'].tolist()[0]
        

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
            id_sentenca: ID da sentença armazenada no DataFrame. O ID da senteça não conta nos arquivos TimeML, foram criados na função timeml_to_df.
                    Se for passada um lista de id_sentenca, será conciderada apenas o primeiro item da lista.
        '''
        if id_sentenca is None:
            return False
        
        df_id_sentenca = self.df.sentenca_completo.query("isentenca == " + str(id_sentenca) + "")
        return not df_id_sentenca.empty


    
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
        '''
        def __init__(self, tb: 'TimebankPT'):  #o 'tb' é recebido no self do parâmetro da instanciação da class, ex: df = Df(self)
            '''
            Inicializa classe recebendo instancia de TimebankPT (tb) e carrega dados dos arquivos do corpus para Dataframes.
            
            Args:
                tb: Recebe instancia da classe TimebankPT
                
            '''
            self.__tb = tb
            self.__recursivo = False
            self.__carregar_df_completo()
            
        
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
                resursivo: Se True, é exibido também as sentenças que possuem relações temporais dependentes.
                           Se False, exibe apenas as sentenças informadas.
            
            '''
            recursivo_atual = self.__recursivo
            
            if recursivo_atual != recursivo:
                self.__recursivo = recursivo
                self.atualizar_filtros()
        
        
        def __add_task_in_event(self):
            #Adiciona task em EVENT
            ## IMPLEMENTAR
            self.__df_event['taskA'] = 'True'
            self.__df_event['taskB'] = 'True'
            self.__df_event['taskC'] = 'True'
        
        
        def __carregar_df_completo(self):
            '''
            Carrega os DataFrames completos, sem filtros.
            '''
            self.__df_event, self.__df_timex3, self.__df_tlink, self.__df_sentenca, self.__df_doc = self.__timeml_to_df()
            self.__add_task_in_event()
            
            
            
        def atualizar_filtros(self):
            '''
            Carrega os DataFrames filtrados conforme parametros.
            É chamada senpre que uma propriedade da classe é alterado, por exemplo, set_id_sentenca, set_sentenca_texto, recursivo.
            
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
            
            return self.tlink_join_completo.query("isentenca in " + str(id_sentenca)) 
        
        @property
        def tlink_join_doc(self):
            '''
            Retorna DataFrame contendo todos atributos de TLINK e suas chaves extrangeiras, porém apenas os registros do documento atual.
            
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
            df_join_eventID = df_tlink.merge(df_event_eventID, on=['eventID', 'doc'], how='outer', suffixes=('_TLINK_L', '_EVENTID_R')) #left
            df_tlink = None
            
            #JOIN relatedToTime
            df_join_event_relatedToTime = df_join_eventID.merge(df_timex3_relatedToTime, on=['relatedToTime', 'doc'], how='left', suffixes=('_EVENTID_L', '_RTOTIME_R'))
            df_join_eventID = None
            
            #JOIN relatedToEvent
            df_join_event_relatedToTime_relatedToEvent = df_join_event_relatedToTime.merge(df_event_relatedToEvent, on=['relatedToEvent', 'doc'], how='left', suffixes=('_RTOTIME_L', '_RTOEVENT_R'))
            df_join_event_relatedToTime = None
            
            #JOIN anchorTimeID
            col = ['lid', 'relType', 'task', 'doc', 
                   'eventID', 'isentenca_EVENTID_L', 'text_EVENTID_L', 'class_RTOTIME_L', 'tense_RTOTIME_L', 'pos_RTOTIME_L', 'aspect_RTOTIME_L',
                   'relatedToTime', 'isentenca_RTOTIME_R', 'tag_RTOTIME_L', 'type_RTOEVENT_L', 'value_RTOEVENT_L', 'text_RTOTIME_R',
                   'relatedToEvent', 'isentenca_RTOEVENT_L', 'text_RTOEVENT_L', 'class_RTOEVENT_R', 'tense_RTOEVENT_R', 'pos_RTOEVENT_R', 'aspect_RTOEVENT_R', 
                   'anchorTimeID', 'isentenca_ANCHOR_R', 'tag', 'type_ANCHOR_R', 'value_ANCHOR_R', 'text_ANCHOR_R']
            df_join_event_relatedToTime_relatedToEvent_anchorTimeID = df_join_event_relatedToTime_relatedToEvent.merge(df_timex3_anchorTimeID, on=['anchorTimeID', 'doc'], how='left', suffixes=('_RTOEVENT_L', '_ANCHOR_R'))[col]

            #Renomeia cnome das colunas para facilitar a visualização
            col_rename = {'isentenca_EVENTID_L': 'isentenca', 'text_EVENTID_L': 'text', 'class_RTOTIME_L': 'class', 'tense_RTOTIME_L': 'tense', 'pos_RTOTIME_L': 'pos', 'aspect_RTOTIME_L': 'aspect',
                   'isentenca_RTOTIME_R': 'isentenca_rt', 'tag_RTOTIME_L': 'tag_rt', 'type_RTOEVENT_L': 'type_rt', 'value_RTOEVENT_L': 'value_rt', 'text_RTOTIME_R': 'text_rt',
                   'isentenca_RTOEVENT_L': 'isentenca_re', 'text_RTOEVENT_L': 'text_re', 'class_RTOEVENT_R': 'class_re', 'tense_RTOEVENT_R': 'tense_re', 'pos_RTOEVENT_R': 'pos_re', 'aspect_RTOEVENT_R': 'aspect_re', 
                   'isentenca_ANCHOR_R': 'isentenca_at', 'tag': 'tag_at', 'type_ANCHOR_R': 'type_at', 'value_ANCHOR_R': 'value_at', 'text_ANCHOR_R': 'text_at'}
            df_join_event_relatedToTime_relatedToEvent_anchorTimeID.rename(columns=col_rename, inplace=True)
            
            return df_join_event_relatedToTime_relatedToEvent_anchorTimeID.sort_values(['isentenca', 'task', 'eventID', 'relatedToTime', 'relatedToEvent'])
        
        
        #Retorna a quentidade de registro do dataframe
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
            Retorna dicionario contendo os dados necessários para o processamento do pipeline do spaCy: pipe_timebankpt.
            
            Return:
                {
                    'texto da sentença': 
                    {
                        'isentenca': 'id sentença',
                        'doc': 'nome do arquivo doc',
                        'dct': 'data de criação do documento',
                        'lista_event': [[], [], []],
                        'lista_timex3': [[], []]
                    },
                    
                    'Repetidamente, ele resiste.':
                    {  
                        'isentenca': '254', 
                        'doc': 'ABC19980120.1830.0957', 
                        'dct': '1998-01-20', 
                        'lista_event': [['EVENT', '2', 'e1', 'previram', 14, 22, 'I_ACTION', 'NONE'], ['EVENT', '2', 'e86', 'queda', 29, 34, 'OCCURRENCE', 'NONE']], 
                        'lista_timex3': [['TIMEX3', '10', 't94', 'quase quarenta anos', 8.0, 27.0, 'DURATION', 'P40Y']]
                    }
                }

            '''
            col = ['isentenca', 'sentenca', 'doc', 'dct', 'tid', 'type']
            df_sentenca = self.sentenca_completo.merge(self.documento_completo, on='doc', how='left')[col]
            df_sentenca.set_index('sentenca', inplace=True)

            dict_pipe = {}
            for sentenca, row in df_sentenca.iterrows():
                dados = {'isentenca': row['isentenca'], 'tid':row['tid'], 'doc': row['doc'], 'dct': row['dct'], 'type': row['type'], 
                         'lista_event': self.__get_lista_event(row['isentenca']), 
                         'lista_timex3': self.__get_lista_timex3(row['isentenca'])}
                dict_pipe.update({sentenca: dados})

            return dict_pipe

        def __get_lista_event(self, id_sentenca):
            '''
            Retorna lista de todos os eventos (EVENT) da sentença passada em id_sentenca. Considera apenas a primeira id_sentenca se for uma lista
            
            '''
            lista = []
            
            if not id_sentenca:
                return []
            
            id_sentenca = self.__tb.trata_lista(id_sentenca)
            
            if len(id_sentenca) == 0:
                return []
            
            id_sentenca = id_sentenca[0]
            
            df_event_sentenca = self.event_completo.query("isentenca == " + str(id_sentenca) + "")
            if not df_event_sentenca.empty:
                lista = df_event_sentenca[['tag', 'isentenca', 'eid', 'text', 'p_inicio', 'p_fim', 'class', 'aspect', 'tense', 'pos']].values.tolist()
            
            return lista

        def __get_lista_timex3(self, id_sentenca):
            '''
            Retorna lista de todas as expressões temporais (TIMEX3) da sentença passa em id_sentenca. Considera apenas a primeira id_sentenca se for uma lista

            '''
            lista = []

            if not id_sentenca:
                return []
            
            id_sentenca = self.__tb.trata_lista(id_sentenca)

            if len(id_sentenca) == 0:
                return []
            
            id_sentenca = id_sentenca[0]

            df_timex3_sentenca = self.timex3_completo.query("isentenca == " + str(id_sentenca) + "")
            if not df_timex3_sentenca.empty:
                lista = df_timex3_sentenca[['tag', 'isentenca', 'tid', 'text', 'p_inicio', 'p_fim', 'type', 'value']].values.tolist()

            return lista
        
        
        #-------------------------------
        # FUNÇÕES PRIVADAS DA CLASSE DF
        #-------------------------------
        
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
                train_test = self.__tb.train_test(arq_timeml)
                xtree = et.parse(arq_timeml)
                root = xtree.getroot()

                nome_arquivo = os.path.basename(arq_timeml)
                doc, ext = os.path.splitext(nome_arquivo)

                for node in list(root):
                    #DCT -> há apenas um DCT para cada documento
                    if node.tag == 'TIMEX3':
                        node.attrib['tag'] = 'DCT'
                        node.attrib['text'] = node.text
                        node.attrib['doc'] = doc
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
                           
                        #retira senteças com conteúdo inválido, como poucos caracteres e entre parenteses: (sp/eml)
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
                                #como a pesquisa é feita na substring iniciada em i_aux, é necessário adicionar seu valor em inicio para que a posição seja sempre em referência ao primeiro caracter da sentença
                                inicio = encontrou.start() + i_aux

                            #Corrigir problemas com HÍFENS
                            #Adicior o sufixo com hifem, se houver, ao elem_text
                            #ex: se elem_text = declarar
                            #mas na sentença = declarar-se então adiciona o sufixo em elem_text
                            #É necessário porque o TOKEN é formado também com o sufixo
                            pos_proximo_char = inicio + len(elem_text)
                            proximo_char = s[pos_proximo_char]
                            tamanho_sufixo = 0
                            if proximo_char == '-':
                                #Se o próximo char do elem for um hífen
                                #a partir do hífem, busca o próximo char que não seja letra ou hifem ou seja fim da linha
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

                            if elem.tag == 'EVENT':
                                event.append(elem.attrib)
                            if elem.tag == 'TIMEX3':
                                timex3.append(elem.attrib)
                            #print(elem.attrib)
                            
                    if node.tag == 'TLINK':
                        node.attrib['tag'] = node.tag
                        node.attrib['doc'] = doc
                        tlink.append(node.attrib)
                        #print(node.attrib)
            
            col_timex3 = ['tid', 'isentenca', 'tag', 'text', 'type', 'value', 'anchorTimeID', 'temporalFunction', 'functionInDocument', 'mod', 'beginPoint', 'endPoint', 'quant', 'freq', 'p_inicio', 'p_fim', 'doc']
            df_timex3 = pd.DataFrame(timex3, columns=col_timex3) 
            df_timex3 = df_timex3[col_timex3]
            df_timex3 = df_timex3.sort_values(['doc', 'isentenca', 'p_inicio'])
            
            df_event = pd.DataFrame(event)
            df_event = df_event.astype({'isentenca': 'int'})
            df_event = df_event[['eid', 'isentenca', 'tag', 'text', 'class', 'stem', 'aspect', 'tense', 'polarity', 'pos', 'p_inicio', 'p_fim', 'doc']]
            df_event = df_event.sort_values(['doc', 'isentenca', 'p_inicio'])
            
            df_tlink = pd.DataFrame(tlink)
            df_tlink = df_tlink[['lid', 'tag', 'task', 'relType', 'eventID', 'relatedToTime', 'relatedToEvent', 'doc']]
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


            #TLINK -> Busca Event e Timex3
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


            #Lista Timex3 apenas DCT -> Geralmente cada documeto conteu apenas um DCT, mas deixei como lista assim mesmo
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
        
        '''
        def __init__(self, tb: 'TimebankPT'):  #o 'tb' é recebido no self do parâmetro da instanciação da class, ex: df = Df(self)
            '''
            Inicializa classe recebendo instancia da classe TimebankPT (tb)
            
            Args:
                tb: Instancia da classe TimebankPT.
                
            '''
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
                self.__tb.id_sentenca = self.__id_sentenca_anterior
                if self.__sentenca_anterior:
                    self.__tb.sentenca_texto = self.__sentenca_anterior
            
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
            Retorna dicionario contendo os campos do spaCy que serão impressos na tela
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
            Retorna dicionario contendo os campos do spaCy que serão impressos na tela
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
            Retorna dicionario contendo os campos do spaCy que serão impressos na tela
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
            Retorna dicionario contendo os campos do spaCy que serão impressos na tela
            '''
            return {
                'i'           :token.i,
                'Token'       :token.orth_,
                'ENT'         :token.ent_type_,
                'POS'         :token.pos_,
                'Desc POS'    :self.__tb.explicar_spacy(token.pos_), 
                'PAI'         :token.head, 
                'POS PAI'     :token.head.pos_,
                'DEP'         :token.dep_, 
                'id_sentenca' :token._.id_sentenca,  
                'id_tag'      :token._.id_tag,
                'classe'      :token._.classe, 
                'aspecto'     :token._.aspecto, 
                'tempo'       :token._.tense,
                'pos'         :token._.pos,
                'tipo'        :token._.tipo,
                'valor'       :token._.value
            }
        
        def __campos_tokens(self, token: Token):
            '''
            Retorna dicionario contendo os campos do spaCy que serão impressos na tela
            '''
            return {
                'i'         :token.i,
                'Token'     :token.orth_,
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
            
            
        def graph_dfs(self, id_sentenca = None, mostrar_mais = True):
            '''
            Imprime arvore sintática utilizanto a Busca por Profundidade
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
                dfs(self.__tb.get_doc_root(doc))
                i += 1
            
            self.__recupera_sentenca_anterior(id_sentenca)
            
        
        def graph_treelib(self, id_sentenca = None):
            '''
            Imprime arvore sintática utilizanto a Busca por Profundidade com a biblioteca treelib
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
                
                root = self.__tb.get_doc_root(doc)
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
        Recebe as Relações Temporais descobertas pelo método aqui proposto.
        Salva-as em arquivo no mesmo formato das tag TLINK do corpus TimebankPT.
        Fornece impressão gráfica das relações.

        '''
        def __init__(self, tb: 'TimebankPT'):
            '''
            Recebe instancia da classe TimebankPT e e inicializa estrutura de dados para o gráfico das Relações Temporais.

            Args:
                tb: Instancia da classe TimebankPT.

            '''
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
            
            self.relType_validos = ['BEFORE', 'AFTER', 'OVERLAP']
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
                task: Tipo da tarefa do Tempeval. 
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
            
            if relType not in self.relType_validos:
                print("ERROR: A relação: '{0}' de eventID: '{2}', relatedTo: '{3}' e Task: '{4}' não é um tipo de relação válida. \n       Valores válidos: {1}".format(relType, str(self.relType_validos), eventID, relatedTo, task) )
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
                print("WARNING: Está tendo adicionar: id_sentenca: {3}, eventID: '{0}', relatedTo: '{1}', task: '{2}', rule {4}. POREM JÁ EXISTE COMO: relType: '{5}', lid: '{6}', rule: '{7}'.".format(eventID, relatedTo, task, isentenca, rule, text_encontrado[self.RELTYPE], text_encontrado[self.LID], text_encontrado[self.RULE]))

        
        def remove(self, relType, eventID, relatedTo, task, isentenca, doc, rule, lid = None):
            '''
            Remove TLink da estrutuda de dados.
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

                sobrescrever: Se True, sobrescreve o arquivo file_tlink se ele existir, se não existir, cria-o.
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
            except:
                print('Ocorreu erro ao salvar o arquivo ' + file_tlink)
            else:
                print('Dados salvo com sucesso em ' + file_tlink)

        def load_from_file(self, file_tlink, modo = 'w'):
            '''
            ######### PROVAVELMENTE ESTE MÉTODO SERÁ EXCLUIDO -> ANALISAR ISSO DEPOIS
            
            Carrega dados do arquivo salvo dados pelo método save_to_file()

            Args:
                file_tlink: Arquivo tml contendo tags TLINK criado pelo método save_to_file()

                modo: se 'w' (write), limpa as carga anterior de self.to_list, sobrescreve conteudo já carragado.
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
            except:
                print('Ocorreu erro ao carregar o arquivo ' + file_tlink)
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
            Retorna DataFrame de MyTlink contendo os dados principais das chaves extrangeiras.

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
                task: A. EVENT-TIMEX3 (intra-sentença)
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
                text: É uma lista com os atrubutos [lid, relType, eventID, relatedToTime, relatedToEvent, task, isentenca, doc, rule]
                
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
            Remove o TLink com id = lid da estrutuda de dados.
            
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
            Retorna estrura de dados no padrão do displaCy dep para composição manual de árvore de depencia customizada.
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
            Carrega a estrutua de dados __struct_words com todas as palavras (words) da sentença.
            
            '''
            if self.__tb.doc_unico:
                for token in self.__tb.doc_unico:
                    self.__add_words(token.text, token.pos_)

        def __load_arcs(self):
            '''
            Carrega a estrutua de dados __struct_words com todos os arcos (Relações Temporais) extraídas.
            
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
            Adiciona uma 'word' que é composta por text e tag à estrutua de dados __struct_words.
            
            '''
            lista = self.__struct_words
            text = {'text': text, 'tag':tag}
            lista.append(text)

        def __add_arcs(self, start, end, label):
            '''
            Adiciona um 'arc' que é composta por start, end e label à estrutua de dados __struct_arcs.
            Um arcs representa uma relação temporal.
            
            '''
            #valida label
            label = label.upper()
            label_validos = [*self.relType_validos, 'BEFORE-OR-OVERLAP', 'OVERLAP-OR-AFTER']
            if label not in label_validos:
                print('ERROR: {0} de start: {2} e end: {3} não é um tipo de relação válida. \n       Valores válidos: {1}'.format(label, str(label_validos), start, end) )
                return

            #Valida dir
            if label in ['AFTER', 'OVERLAP-OR-AFTER']:
                direcao = 'left'
            elif label in ['BEFORE', 'BEFORE-OR-OVERLAP']:
                direcao = 'rigth'
            else:  #['OVERLAP', 'VAGUE']
                direcao = 'rigth'

            #valida start e end
            start_ori = start
            if type(start) == str:
                start = self.__idtag_to_i(start)
                if start is None:
                    print("ERROR: id_tag '{0}' não encontrada".format(start_ori))
                    return

            end_ori = end
            if type(end) == str:
                end = self.__idtag_to_i(end)
                if end is None:
                    print("ERROR: id_tag '{0}' não encontrada".format(end_ori))
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
            
            for ent in self.__tb.doc_unico.ents:
                if ent.label_ in ('EVENT', 'TIMEX3'):
                    for token in ent:
                        struct.append({'i':token.i, 'id_tag':token._.id_tag, 'token':token.text, 'ent_type':token.ent_type_})
            #DCT
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
    Funções auxiliares das regras para extrair Relações Temporais.
    
    '''
    def __init__(self, tb: 'TimeBankPT'):
        '''
        Inicializa a classe.
        
        Args:
            doc: objeto Doc atual da class TimebankPT.
            
        '''
        self.__tb = tb
        
        #Criei outro nlp para melhorar performance, neste pipe só precisa do lemmatizer, os outros pipes foram excluídos
        self.__nlp_lemma = spacy.load('pt_core_news_lg', exclude= ['tok2vec', 'morphologizer', 'parser', 'attribute_ruler', 'ner'])
        
    
    def search(self, palavras, frase, lemma: bool = False):
        '''
        Verifica se 'palavras' inteiras estão presentes em 'frase'.

        Args:
            palavras: Pode ser list, str, Token ou expressão regular.
                      Pesquisa por palavras inteiras.
                      Regex gerado: (\W|^)(palavras_separadas_por_pipe)(\W|$)
            frase: Texto onde as 'palavras' serão encontradas.
                   Pode ser list, str, Doc, Span e Token
            lemma: Se True, lematiza palavras e frase.

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
        Converte 'lista' em string minúculas.

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

        return lista.lower()

    def str_to_list(self, palavra: str) -> list:
        '''
        Se 'palavra' for string, converte em lista unitária de string maiúsculas.
        Se não, converte em maíusculas.
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
            frase: Texto que deseja lematizar. 
                   Pode ser Doc, Span, Token, list ou str.

        Return:
            Se frase for Doc, Span, Token ou str, retorna string lematizada.
            Se frase for list, retorna lista com elementos únicos lematizados.

        '''
        if type(frase) in [Span, Token, Doc]:
            if type(frase) == Doc:
                frase = frase[0:len(frase)]  #-> to Span
            return frase.lemma_   # -> str

        elif type(frase) == str:
            doc = self.__nlp_lemma(frase.lower())
            span = doc[0:len(doc)]
            return span.lemma_    # -> str

        elif type(frase) in [list, tuple]:
            lista_lemma = []
            for palavra in frase:
                doc = self.__nlp_lemma(palavra.lower())
                span = doc[0:len(doc)]
                lista_lemma.append(span.lemma_)
            return list(set(lista_lemma))  # -> list única

        else:
            print('ERROR: função __lemma recebeu tipo desconhecido.')
            return

    def __doc_ids(self, ids) -> "List Doc":
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
                 se dict no formato {'antes':[], 'depois':[]} contatena texto que vem antes com o que vem depois
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
            frase = docs.text 
            
        return frase.strip()
        

    def __idsContexto(self, token: Token, distancia = 5, contexto = None) -> list:
        '''
        Retorna lista de i dos tokens dentro da distância conforme contexto (antes ou depois do token).
        Se contexto 'antes', o 'token' é incluído. Se 'depois', é adicionado mais um token no final.

        Args:
            token: objeto Token do spaCy
            contexto: 'antes' ou 'depois' do token
            distancia: Quantidade de tokens antes ou depois do token. 
                       Se 'max' (ou qualquer string), extende até o final ou o início da sentença, conforme contexto.

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
        Returna True se t1 estiver posicionado na frase antes de t2. Senão retorna False.
        
        '''
        if t1.i < t2.i:
            return True
        return False

    def t1AfterT2(self, t1: Token, t2: Token) -> bool:
        '''
        Returna True se t1 estiver posicionado na frase depois de t2. Senão retorna False.
        
        '''
        return t1BeforeT2(t2, t1)
    
    
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
                       Se 'max' (ou qualquer string), extende até o final ou o início da sentença, dependendo do contexto.
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
                       Se 'max' (ou qualquer string), extende até o final ou o início da sentença, dependendo do contexto.
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
                       Se 'max' (ou qualquer string), extende até o início da sentença, conforme contexto.
        '''
        return self.hasPosInContext(token, pos, distancia, 'antes')

    def hasPosInContextFollow(self, token: Token, pos: list, distancia = 5) -> bool:
        '''
        Verifica se existe a classe gramatical 'pos' a uma distância de até 5 tokens depois do 'token'.

        Args:
            token: Objeto Token do spaCy.
            pos: Classe gramatical - POS Tag.
            distancia: Quantidade de tokens depois de 'token'. 
                       Se 'max' (ou qualquer string), extende até o fim da sentença, conforme contexto.
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
            distancia: Quantidade de tokens antes de 'token'. 
                       Se 'max' (ou qualquer string), extende até o início da sentença, conforme contexto.
        '''
        return self.hasNoVerbInContext(token, distancia, 'antes')

    def hasNoVerbInContextFollow(self, token: Token, distancia = 5) -> bool:
        '''
        Verifica se existe VERBO a uma distância de até 5 tokens depois do 'token'.

        Args:
            token: objeto Token
            distancia: Quantidade de tokens depois de 'token'. 
                       Se 'max' (ou qualquer string), extende até o fim da sentença, conforme contexto.
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
        tempo1 = token1.morph.get('Tense')
        tempo2 = token2.morph.get('Tense')

        if not (tempo1 or tempo2):
            return False
        return tempo1 == tempo2

    def tenseVerb(self, token: Token, tense: list) -> bool:
        '''
        Verifica se 'token' possui tempo verbal 'tense'.

        Args:
            token: Token
            tense: str ou list
                   tempo verbal válidos: 'FUT', 'IMP', 'PAST', 'PQP', 'PRES'

        '''
        tense_valid = ['FUT', 'IMP', 'PAST', 'PQP', 'PRES']
        tense = self.str_to_list(tense)

        for t in tense:
            if t not in tense_valid:
                print('ERROR: Regras contendo tempo verbal inválido. Valores válido: ' + str(tense_valid))
                return False

        tense_atual = list(map(str.upper, token.morph.get('Tense')))
        if tense_atual:
            tense_atual = tense_atual[0]

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
        Verifica se 'token' possue um dos tipos de dependência da lista 'dep'.

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
            distancia: Quantidade de tokens antes de 'token'. 
                       Se 'max' (ou qualquer string), extende até o início da sentença, conforme contexto.
        '''
        return self.hasDepInContext(token, dep, distancia, 'antes')

    def hasDepInContextFollow(self, token: Token, dep: list, distancia = 5) -> bool:
        '''
        Verifica se existe a dependência 'dep' a uma distância de até 5 tokens depois do 'token'.

        Args:
            token: objeto Token
            distancia: Quantidade de tokens depois de 'token'. 
                       Se 'max' (ou qualquer string), extende até o fim da sentença, conforme contexto.
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


    #--------------------------------------
    #MORPH

    def morph(self, token: Token, morph: tuple) -> bool:
        '''
        Verifica se 'token' possue o elemento morph representado pela tupla (key, value) da análise morfológica.

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
                    #Para os cosos em que desejo encontrar um key com value vazio, ex: ('Tense', '') ou ('VerbForm', '')
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
            keyvalue: Tuple que representa o par da morfologia que deseja (key, value). 
                      Ex:  (Tense, Fut) ou (VerbForm, Inf).
            distancia: Quantidade de tokens antes de 'token'. 
                       Se 'max' (ou qualquer string), extende até o início da sentença, conforme contexto.
        '''
        return self.hasMorphInContext(token, keyvalue, distancia, 'antes')

    def hasMorphInContextFollow(self, token: Token, keyvalue: tuple, distancia = 5):
        '''
        Verifica se existe na análise morfológica o 'key' de valor 'value' a uma distância de até 5 tokens depois de 'token'.

        Args:
            token: objeto Token do spaCy.
            keyvalue: Tuple que representa o par da morfologia que deseja (key, value). 
                      Ex:  (Tense, Fut) ou (VerbForm, Inf).
            distancia: Quantidade de tokens depois de 'token'. 
                       Se 'max' (ou qualquer string), extende até o fim da sentença, conforme contexto.

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

    def hasPastParticipleInContext(self, token: Token, distancia = 5, contexto: 'antes ou depois' = None):
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
        Verifica se 'token' é uma das classes de evetos da lista 'classe'.

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
    def contextBy(self, token: Token, tipo: "'str', 'str_lemma', 'token', 'digito', 'pos', 'dep', 'morph'", valor = None, distancia = 'max', contexto: 'antes, depois' = None) -> bool:
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

    def precededBy(self, token: Token, tipo: "string: 'str', 'str_lemma', 'token', 'digito', 'pos', 'dep', 'morph'", valor = None, distancia = 'max') -> bool:
        '''
        Procura elementos na sentença antes do 'token', conforme o 'tipo'.

        Args:
            token: objeto Token do spaCy.
            tipo: string e pode ser:
                  str      -> verifica se existe palavras especificada em valor 
                           -> valor: str ou list; PODE SER OMITIDO.
                  str_lemma -> verifica se existe palavras lematizadas especificada em valor 
                            -> valor: str ou list;
                  token    -> verifica se 'token' precede o outro token especificado em valor 
                           -> valor: Token; PODE SER OMITIDO.
                  digito   -> verifica se há digitos ou pos = 'NUM' -> Não tem valor;
                           -> Se valor for informado, ele será a distância.
                  pos      -> verifica se há a classe gramatical especificada em valor 
                           -> valor: list, ex: ['VERB', 'NOUM'];
                  dep      -> verifica se há na árvore de dependencia o elemento especificado em valor 
                           -> valor: list, ex: ['nsubj', 'nmod'];
                  morph    -> verifica se há na análise morfológica o elemento especificado em valor. 
                           -> valor: tuple (key, value), ex: ('Tense', 'Fut'). Só é permitido uma tupla.
            valor: valor do elemento que será procurado, conforme o tipo.
            distancia: Se inteiro, é quantidade de tokens antes de 'token' onde a pesquisa será realizada.
                       Se string, a pesquisa será realizada em todos os tokens que precedem 'token'.
        '''
        return self.contextBy(token, tipo, valor, distancia, contexto = 'antes')

    def followedBy(self, token: Token, tipo: "string: 'str', 'str_lemma', 'token', 'digito', 'pos', 'dep', 'morph'", valor = None, distancia = 'max') -> bool:
        '''
        Procura elementos na sentença depois do 'token', conforme o 'tipo'.

        Args:
            token: objeto Token do spaCy.
            tipo: string e pode ser:
                  str       -> verifica se existe palavras especificada em valor 
                            -> valor: str ou list; PODE SER OMITIDO.
                  str_lemma -> verifica se existe palavras lematizadas especificada em valor 
                            -> valor: str ou list;
                  token     -> verifica se 'token' vem depois do outro token especificado em valor 
                            -> valor: Token; PODE SER OMITIDO.
                  digito    -> verifica se há digitos ou pos = 'NUM' -> Não tem valor;
                            -> Se valor for informado, ele será a distância.
                  pos       -> verifica se há a classe gramatical especificada em valor 
                            -> valor: list, ex: ['VERB', 'NOUM'];
                  dep       -> verifica se há na árvore de dependencia o elemento especificado em valor 
                            -> valor: list, ex: ['nsubj', 'nmod'];
                  morph     -> verifica se há na análise morfológica o elemento especificado em valor 
                            -> valor: tuple (key, value), ex: ('Tense', 'Fut'). Só é permitido uma tupla.
            valor: valor do elemento que será procurado, conforme o tipo.
            distancia: Se inteiro, é quantidade de tokens depois de 'token' onde a pesquisa será realizada.
                       Se string, a pesquisa será realizada em todos os tokens que vem depois 'token'.
        '''
        return self.contextBy(token, tipo, valor, distancia, contexto = 'depois')

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
            distancia: Se inteiro, é quantidade de tokens antes de 'token'.
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
            distancia: Se inteiro, é quantidade de tokens depois de 'token'.
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
    Adiciona TLINKs transitivos na relações preditas pela classe RelacaoTemporal.
    '''
    def __init__(self, tb: 'TimeBankPT'):
        self.__tb = tb
        
    def save_tlinks_transitive(self, pares_relacoes_sentenca: dict):
        '''
        Processa e adiciona TLINKs transitivos às relações preditas pelo método.
        São adicionados à estrutura de dados da classe MyTlink.
        
        Args:
            pares_relacoes_sentenca: dicionário no formato {(eventID, timex3ID): typeRel, ('e1', 't2'): 'BEFORE', } contendo todas relações da sentença.
            
        '''
        pares_novos = self.transitive_closure(pares_relacoes_sentenca)
        if len(pares_novos) > 0:
            task = 'A'
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

        #Inverte as relações do tipo AFTER para BEFORE e as relações OVERLAP adiciona mais o par invertido.
        closure = self.__pre_process_pairs(pairs)
        new_relations = {}

        while True:
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

            #retira os repetidos
            closure_until_now = closure | new_relations

            #condição de parado do laço while True
            #pára quando a adição de uma nova relação não gera mais novas relações transitivas
            if closure_until_now == closure:
                break
            closure = closure_until_now

        #Todas as relações, inclusive as novas
        #Excluindo relações reflexivas e TIMEX3/TIMEX3. Além de inverter as relações TIMEX3/EVENT para EVENT/TIMEX3.
        closure = self.__pos_process_pairs(closure, retirar_event_event)

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


    def __invert_relation(self, relation: '((x, y), rel)') -> '((y, x), rel)':
        '''
        Inverte o tipo da relação e seu par.

        Args:
            relation: par no formato: ((x, y), 'AFTER')')

        Return:
            Retorn par invertido: (y, x), 'BEFORE')

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
        Retorn True se token for do tipo TIMEX3
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
        
        self.__file_model = 'dtree.model' #Nome do arquivo onde o modelo treinado será salvo
        self.dtree = None
        self.__siglas = Siglas()
        
        if self.__file_model_exist():
            self.__load_model()
        
        self.__approach = 'ML'   # REGRAS_TREE, REGRAS OR ML
        
        #se approach = ML
        self.__X_treino = pd.DataFrame(None)
        self.__X_teste = pd.DataFrame(None)
        self.__y_treino = pd.DataFrame(None)
        self.__y_teste = pd.DataFrame(None)
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
        approach_validos = ['REGRAS', 'ML', 'REGRAS_TREE']
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
            df_features: Dataframe[colunas], onde colunas = ['e_pos', 'e_dep_com_pai', 'e_class', 'e_pai_pos', 'anotado']
                         Pode ser obtido dos dados anotados do corpus na classe: RelacoesTemporais.df_features.reset_index()[colunas]
                         
        '''
        #Codifica os dados para o modelo ML
        df_codes = self.encode_df(df_features.copy())

        #Dataset sem dummies e com variáveis codificadas
        X = df_codes.loc[:, self.__col_encode]
        y = df_codes.loc[:, self.__col_classe]

        #Dataset dividido sem dummies e com variáveis codificadas
        self.__X_treino, self.__X_teste, self.__y_treino, self.__y_teste = train_test_split(X, y, test_size = 0.2, random_state = 17)
        
        #TREINA E SALVA MODELO
        try:
            self.dtree = DecisionTreeClassifier(criterion="gini", random_state=17,  max_depth=4, splitter='best')
            self.dtree.fit(self.__X_treino, self.__y_treino)
        except:
            print('Ocorreu um erro no treinamento.')
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
        
        #Importancia de cada variável
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
        Encode features para submissão a modelos de machine learnig.
        
        Args:
            df: Dataframe contendo as colunas: 'e_pos', 'e_dep_com_pai', 'e_class', 'e_pai_pos'
                
        '''
        for col_name in self.__col_encode:
            df.loc[:, col_name] = df.loc[:, col_name].apply(lambda x: self.__siglas.get_key(x, self.__col_encode[col_name]))
        return df

    def encode_tokenE(self, tokenE: Token):
        '''
        Constroi e codifica DataFrame com features de 'tokenE' para submissão ao modelo de learning.
        
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
# Classe RELACAOTEMPORAL - Atribui RT
#--------------------------------------------------------------------

class RelacaoTemporal():
    '''
    Atribui relações temporais às sentencas.
    
    '''
    __COD     = 0
    __TIPO    = 1
    __ORD     = 2
    __EXPRESS = 3
    __TIPO_ORDEM = ['codigo_regra', 'tipo_relacao', 'ordem', 'random']
        
    def __init__(self, tb: 'TimebankPT'):
        '''
        Iniciaza classe RelacaoTemporal.
        
        Args:
            tb: instancia da class TimebankPT
            
        '''
        self.__tb = tb
        self.__id_sentenca = None
        
        #Funções que auxiliam na composição das regras, recebe o objeto Doc atual
        self.f = RulesFunctions(tb)
        
        #options
        self.__task = None
        self.__rules = None
        
        self.__processing_type = 'votacao'
        self.__order = 'ordem'
        self.__sort_reverse = False
        
        self.__df_pred = None
        
        self.__df_features = None
        self.tlink_transitive = TlinkTransitive(tb)
        self.tlink_candidate = TlinkCandidate()
        self.__active_tlink_transitive = True
        self.__active_tlink_candidate = True
        
    #----------------------------------------
    # CLASS TOKENDCT
    
    #para task B
    class TokenDct:
        def __init__(self, tb: TimebankPT):
            self.__doc = None
            self.__nome = None
            self.__dct = None
            self.__tid = None
            self.__type = None
            self.__id_sentenca = None
            self.__tb = tb

        def atualizar(self):
            self.__doc = self.__tb.doc_unico
            doc = self.__doc

            self.__nome = doc._.nome
            self.__dct = doc._.dct
            self.__tid = doc._.dct_tid
            self.__type = doc._.dct_type
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
        def id_sentenca(self):
            return self.__id_sentenca
    
    # FIM CLASS TokenDct
    #-----------------------------------------------
    
    @property
    def id_sentencas_sem_predicao(self):
        '''
        Lista de ed_sentenca que ainda não houve predição.
        '''
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
                   [[código regra: float, tipo de relação temporal: str, ordem de execução: float, expressão lógica que representa a regra: str], ]
                   As funções auxiliares das regras são acessadas com o prefixo 'self.f'. 
                   Ex: [249, "OVERLAP", 2, "self.f.is_dependencyType(tokenT, tokenE, 'conj')"]
                   
                   #A 'ordem de execução' com números negativos torna a regra inativa.
            
        '''
        return self.__rules
    
    @rules.setter
    def rules(self, rules: list):
        if self.__check_rules(rules):
            #Grava somente regras ativas
            self.__rules = list(filter(lambda x: x[self.__ORD] >= 0, rules))
        else:
            self.__rules = []

    @property
    def rules_filter(self):
        '''
        Filtra as regras a serem processadas pelo códigos das regras informadas.
        Esta função altera as self.rules. para desfazer o filtro, é necessário é necessário atribuir as regras novamente: self.rules = listas_das_regras.
        
        Args: 
            lista_cod_regras: lista de códigos das regras.
            
        '''
        return self.rules
    
    @rules_filter.setter
    def rules_filter(self, lista_cod_regras: list):
        if type(lista_cod_regras) != list or not lista_cod_regras or len(lista_cod_regras) == 0:
            print('ERROR: rules_filter deve receber lista contendo códigos das regras que deseja filtrar.')
            return
        
        if (not lista_cod_regras) or (len(lista_cod_regras) == 0):
            self.rules = []

        self.rules = list(filter(lambda x: x[self.__COD] in lista_cod_regras, self.rules))
    
    @property
    def df_rules(self):
        '''
        Exibe as regras ativas em formato de tabela.
        
        '''
        colunas=['código', 'tipo', 'ordem', 'expressão']
        
        return pd.DataFrame(self.rules, columns=colunas)

    
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

    
    
    ############ OPÇÕES
    
    def get_sort_rules(self) -> str:
        '''
        Obtem a ordem atual das regras.
        
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
            Order: 'codigo_regra', 'tipo_relacao', 'ordem', 'random'
                   se as regras forem processadas, permite que seja uma lista de cod_regras
            
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
    
        #se order que do tipo list
        #Ordena código da regra de self.rules conforme ordem do lista order
        if type(order) == list:
            self.__order = 'lista'
            self.rules = order_list(order)
            return
        
        self.__sort_reverse = reverse
        
        if order not in self.__TIPO_ORDEM:
            print('ERROR: Ordens válidas: ' + str(self.__TIPO_ORDEM))
            print("Default: 'ordem'")
            order = 'ordem'

        if order == 'codigo_regra':
            self.__order = self.__COD
        
        elif order == 'tipo_relacao':
            self.__order = self.__TIPO
            
        elif order == 'ordem':
            self.__order = self.__ORD
        
        elif order == 'random':
            self.__order = 'random'
            
        if self.__order == 'random':
            random.shuffle(self.rules)
        else:
            self.rules.sort(key = lambda x: x[self.__order], reverse = reverse)

    def get_sort_reverse(self):
        '''
        Obtem a informação sobre se a ordem das regras está ascendente ou descendente.
        '''
        return self.__sort_reverse
        
    
    @property
    def processing_type(self):
        '''
        Propriedade que define a forma como as regras serão processadas.
        Se igual a 'votacao' -> Todas as regras são processadas para todos os pares de relação. A relação temporal mais frequente é retornada. (default)
                   'peneira' -> A relação temporal do par é retornada na primeira regra que casar. Para o par atual, as outras regras não são processadas.

        '''
        return self.__processing_type
    
    @processing_type.setter
    def processing_type(self, tipo: str):
        TIPO_PROCESSAMENTO = ['peneira', 'votacao']
        if tipo not in TIPO_PROCESSAMENTO:
            print('ERROR: Tipo de Processamento inválido. Tipos Válidos: ', str(TIPO_PROCESSAMENTO))
            self.__processing_type = 'votacao'
        self.__processing_type = tipo
        
    
    def status(self):
        '''
        Exibe as principais configurações da instancia da classe.
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
        print('{0:<27} {1}'.format('Tipo Processamento:', self.processing_type))
        print('{0:<27} {1}'.format('Ordenada por:', self.get_sort_rules()))
        print('{0:<27} {1}'.format('Ordem reversa:', self.get_sort_reverse()))
        print('{0:<27} {1}'.format('Tlink Transitivo Ativo?', self.active_tlink_transitive))
        print('{0:<27} {1}'.format('Tlink Candidato Ativo?', self.active_tlink_candidate))
        print('{0:<27} {1}'.format('Tlink Candidato Approach:', self.tlink_candidate.approach if self.active_tlink_candidate else ''))
        print('{0:<27} {1}'.format('Tlink Candidato Threshold:', self.tlink_candidate.threshold if (self.active_tlink_candidate and self.tlink_candidate.approach in ['ML', 'REGRAS_TREE']) else ''))
        print('{0:<27} {1}'.format('Regras processadas?', 'SIM' if self.__check_process_rules() else 'NÃO'))
        
    
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
        Propriedade booleana que informa à classe se serão ou não selecionados EVENTs com maior probabilidade se estaram anotados no corpus para posterior geração de TLINKs.
        
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
    
    
    def __check_rules(self, rules: list) -> bool:
        '''
        Checa a consistências da lista de regras. 
        
        Args: 
            rules: lista que contem as regras.
            
        '''
        #check estrutura inicial
        if type(rules[0]) != list:
            print('ERROR: rules deve ser uma lista de listas, no formato: [[código regra, tipo de relação temporal, ordem de execução, expressão lógica que representa a regra], ]')
            return False
        
        unico = []
        repetido = []
        
        #atribui uma sentença qualquel e tokens quaisquer para avalia a expressão que representa a regra
        self.__tb.id_sentenca = 10
        tokenE = self.__tb.doc_unico[9]
        tokenT = self.__tb.doc_unico[1]
        
        for rule in rules:
            #Regras com código repetido
            if rule[self.__COD] not in unico:
                unico.append(rule[self.__COD])
            else:
                repetido.append(rule[self.__COD])

            #checa toda estrutura
            if len(rule) != 4:
                print('ERROR: COD = {0}. Cada regra deverá conter: [código regra, tipo de relação temporal, ordem de execução, expressão lógica que representa a regra].\nValor informado: {1}'.format(rule[self.__COD], rule))
                return False
            if type(rule[self.__COD]) not in [int, float]:
                print('ERROR: COD = {0}. Código da Regra deve ser numérico.'.format(rule[self.__COD]))
                return False
            if rule[self.__TIPO] not in self.__tb.my_tlink.relType_validos:
                print('ERROR: COD = {0}. Tipo da Relação Temporal inválida: {1}. Valores válidos: {2}.'.format(rule[self.__COD], rule[self.__TIPO], str(self.__tb.my_tlink.relType_validos)))
                return False
            if type(rule[self.__ORD]) not in [int, float]:
                print('ERROR: COD = {0}. Ordem de processamento da Regra inválido: {1}. Deve ser numérico.'.format(rule[self.__COD], rule[self.__ORD]))
                return False
            if type(rule[self.__EXPRESS]) != str:
                print('ERROR: COD = {0}. Expressão que representa a Regra deve ser do tipo string.'.format(rule[self.__COD]))
                return False
            try:
                if rule[self.__ORD] >= 0:  #somente as regras ativas
                    result = eval(rule[self.__EXPRESS])
            except:
                print('ERROR: COD = {0}. Expressão que representa a Regra não é válida: {1}'.format(rule[self.__COD], rule[self.__EXPRESS]))
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
            print("'RelacaoTemporal.task' não definido.")
            return (None, None)
        
        token_dct = self.TokenDct(self.__tb)
        token_eventID = self.__tb.my_tlink.idtag_to_token(eventID)
        if self.task == 'A':
            token_relatedTo = self.__tb.my_tlink.idtag_to_token(relatedTo)
        elif self.task == 'B':
            token_dct.atualizar()
            token_relatedTo = token_dct
        elif self.task == 'C':
            token_relatedTo = self.__tb.my_tlink.idtag_to_token_next(relatedTo)
            
        return (token_eventID, token_relatedTo)
    
    
    def __predict_tlink_rules(self, tokenE: Token, tokenT: Token) -> tuple:
        '''
        Recebe todos os pares de possível relação temporal e retorna o tipo de relação baseado na regra.

        Args:
            tokenE: token do tipo EVENT
            tokenT: token do tipo TIMEX3

        Return:
            Tupla contendo tipo da relação temporal predita (relType) e
            Código da regra que prediz a relação. 
            Se o tipo processamento (rt.processing_type) for 'peneira', ou o código da primeira regra do tipo de relação mais votado.

        '''
        #print(('E: {0:<15} T: {1:<15} : {2}').format(tokenE.text, tokenT.text, rt.f.followedBy(tokenT, 'token', tokenE) ))   

        predict = defaultdict(str)
        
        for rule in self.rules:
            if eval(rule[self.__EXPRESS]):
                if self.processing_type == 'votacao':
                    predict[rule[self.__COD]] = rule[self.__TIPO] #votacao
                if self.processing_type == 'peneira':
                    return rule[self.__TIPO], rule[self.__COD]  #peneira

        if self.processing_type == 'votacao' and predict:
            #Retorna o tipo de relação mais votado e o código da primeira regra do tipo mais votado.
            return self.__trata_predict_votacao(predict) #votação

    
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
                
                #Alimenta self.df_features com atributos sintáticos e POS tagger
                #Pode ser usado para alimentar modelo de ML de TLinkCandidate
                self.__process_features(token_eventID, token_relatedTo)
                
                #===================TLINKCANDIDATE===========================
                #Verifica se o EVENT é um candidato para geração de TLINK
                #Se não for retorna ao inicio do FOR
                if self.active_tlink_candidate:
                    if not self.__process_tlinks_candidate(token_eventID):
                        continue
                #============================================================
                
                #Recebe a tupla (tlink predito, código da regra que preveu)
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
        Seleciona EVENT canditado à geração de TLINK com maior probabilidade de estarem anotados no corpus.
        EVENT com menor probabilidade de não estarem anotados não serão processados, isso é, não serão preditos o tipo de relação temporal que ele tem com o TIMEX3.
        
        Utiliza regras ou modelo de machine learning pré-treinado, dependendo da propriedade TlinkCandidate.approach, que assume o valor 'REGRAS' ou 'ML'.
        
        Args:
            tokenE: token que representa o evento a ser analizado.
        
        '''
        #----REGRAS EXTRAIDAS DA ÁRVORE DE DECISÃO------
        if self.tlink_candidate.approach == 'REGRAS_TREE':
            siglas = self.__tb.siglas
            f = self.f
            threshold = self.tlink_candidate.threshold
            
            #le = less equal
            #os percentuais dos threshold representa a quantidade de pares anotados que foram calculados pelos respectivos nós na árvore de decisão
            if f.dep(tokenE, siglas.dep_le(33.0)):
                if f.classe(tokenE, siglas.classe_le(5.5)):
                    if f.classe(tokenE, siglas.classe_le(3.5)):
                        if f.pos(tokenE, siglas.pos_le(4.0)):
                            return threshold <= 0.11   #11%
                        else:
                            return threshold <= 0.39   #39%
                    else:
                        if f.dep(tokenE, siglas.dep_le(19.5)):
                            return threshold <= 0.76   #76%
                        else:
                            return threshold <= 0.42   #42%
                else:
                    if f.pos(tokenE, siglas.pos_le(7.5)):
                        if f.pos(tokenE, siglas.pos_le(4.5)):
                            return threshold <= 0.03   #3%
                        else:
                            return threshold <= 0.31   #31%
                    else:
                        if f.pos(tokenE, siglas.pos_le(13.5)):
                            return threshold <= 0.0    #0%
                        else:
                            return threshold <= 0.33   #33%
            else:
                if f.dep(tokenE, siglas.dep_le(34.5)):
                    if f.classe(tokenE, siglas.classe_le(3.5)):
                        if f.pos(tokenE, siglas.pos_le(10.5)):
                            return threshold <= 0.01   #1%
                        else:
                            return threshold <= 0.68   #68%
                    else:
                        if f.pos(tokenE, siglas.pos_le(11.0)):
                            return threshold <= 0.28   #28%
                        else:
                            return threshold <= 0.95   #95%
                else:
                    if f.classe(tokenE, siglas.classe_le(5.5)):
                        if f.classe(tokenE, siglas.classe_le(3.5)):
                            return threshold <= 0.38   #38%
                        else:
                            return threshold <= 0.97   #97%
                    else:
                        return threshold <= 0.0        #0%
        
        #-----DECISION TREE------
        elif self.tlink_candidate.approach == 'ML':
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
        
        #Salva sentença atual
        id_sentenca_atual = self.__tb.id_sentenca
        
        #Define sentenças que serão processadas
        if id_sentencas:
            self.__id_sentenca =  self.__tb.trata_lista(id_sentencas)
        else:
            self.__id_sentenca = self.__tb.id_sentencas_task(self.task) 
            
            
        #Limpa estrutura de dados que receberá as predições
        self.__tb.my_tlink.clear()

        #Adiciona TLINKs preditos em tb.my_tlink, 
        #avaliar TlinksCandidate e TlinkTransitive
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
    
    
    def sort_rules_accuracy(self, show_accuracy: bool = False):
        '''
        Retorna lista de cod_regra ordenada por acurácia.
        As regras são processadas individualmente, ou seja, todos os pares são processados por apenas uma regra de cada vez.
        O objetivo é evitar interferências de uma regra sobre outra.
        
        O resultado poderá ser utilizado para processar o conjunto de teste em ordem de acurácia, ex: rt.sort_rules(sort_rules_accuracy: list)
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
            cod_rules_accuracy.append([rule[0], self.df_resultado_por_regras['pct_acerto'].mean()])

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
        Dataframe contendo features dos tokens EVENT e TIMEX3 que poderão ser TLinks.
        Utilizado para análises manuais na seleção de TLinkCandidate 
        e para alimentar dados para treinamento do modelo de machine learning
        
        '''
        return self.__df_features
    
    def __process_features(self, tokenE: Token, tokenT: Token):
        '''
        Alimenta self.df_features com atributos sintáticos e POS tagger
        '''
        #Salvar features de todas as relações do corpus
        eventID = tokenE._.id_tag
        relatedTo = tokenT._.id_tag
        id_sentenca = self.__tb.id_sentenca
        nome_doc = self.__tb.nome_doc_unico
        train_test = self.__tb.get_train_test(id_sentenca)
        
        e_text = tokenE.text
        t_text = tokenT.text
        e_root = self.f.dep(tokenE, 'ROOT')
        e_dep_t = self.f.dependencyType(tokenE, tokenT)
        e_pos = tokenE.pos_
        t_pos = tokenT.pos_
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
        
        
    def process_resume(self):
        '''
        <<Precisa desbabunçar isso aqui>>
        
        Exibe resultado do processamentos do processamento das regras
        
        '''
        if not self.__check_process_rules():
            #print('ERROR: Regras ainda não processadas. Execute o método process_rules().')
            return
        
        acertos = self.df_resultado_por_regras.Acertos.sum()
        total_anotadas = self.df_resultado_por_regras.Total.sum()
        total_extras = self.df_resultado_por_regras.Extras.sum()
        total_geral = total_anotadas + total_extras
        pct_acerto_anotadas = '{0:,.1f}%'.format(acertos / total_anotadas * 100)
        pct_acerto = '{0:,.1f}%'.format(acertos / total_geral * 100)
        
        total_sentenca_task_a = len(self.__tb.id_sentencas_task('A'))
        total_relacoes_task_a = len(self.df_real) 
        pct_cobertura = '{0:,.1f}%'.format(total_anotadas / total_relacoes_task_a * 100)
        
        quant_regras_processadas = self.df_resultado_por_regras.shape[0]
        
        print('RESUMO RELAÇÕES:')
        print('{0:<18} {1:>5} de {2:>7} ({3})'.format('Apenas Anotadas:', acertos, total_anotadas, pct_acerto_anotadas))
        print('{0:<18} {1:>16}'.format('Total Extras:', total_extras))
        print('{0:<18} {1:>5} de {2:>7} ({3})'.format('Acurácia:', acertos, total_geral, pct_acerto))
        
        print('{0:<19} {1:>5} de {2:>7,.1f} ({3})'.format('\nCOBERTURA:', total_anotadas, total_relacoes_task_a, pct_cobertura))
        
        print('\nQuant Regras Processadas: ', quant_regras_processadas)
        print('Total Sentenças Task A:  ', total_sentenca_task_a)
        print('Total Relações Anotadas Task A: ', total_relacoes_task_a)
        print('\n')
        
        
    @property
    def df_pred(self):
        '''
        Exibe em DataFrame as predição de relações temporais processadas pelo método process_rules().

        '''
        if not self.__check_process_rules():
            return False
        
        return self.__df_pred.sort_values(['isentenca', 'eventID', 'relatedTo'])
        
    
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
        df = df[['lid_real', 'relType_real', 'eventID', 'relatedTo', 'task', 'isentenca', 'doc']]
        #Ordena colunas
        df = df.sort_values(['isentenca', 'eventID', 'relatedTo'])
        return df 
    
    def df_real_predict_id_sentenca(self, id_sentenca = None, extra:bool = False):
        '''
        Une dados real do corpus com os dados previsto.
        
        Args:
            id_sentenca: Se informada, filtra por id_sentenca.
            extra: Se True, exibe as previsões que não estão anotadas no corpus, se False (default), exibe apenas as previsões que há correspondência no corpus.
            
        '''
        def esta_na_lista(relType_pred, relType_real):
            '''
            Trata o OR das relações reais BEFORE-OR-OVERLAP e OVERLAP-OR-AFTER, permitindo que relações BEFORE, AFTER e OVERLAP seja aceitas com válidas para aquelas.
            
            Args: 
                relType_pred = Tipo de relação prevista
                relType_real = Tipo de relação anotada
            '''
            if not relType_real or type(relType_real) != str:
                return False
            relType_real = relType_real.split('-OR-')
            return relType_pred in relType_real
        
        def relType_real_sem_OR(relType_pred, relType_real, acertou):
            '''
            Calcula coluna relType_real_sem_OR. Derivada da coluna relType_real, porém com os tipos de relação que contem OR 
                (BEFORE-OR-OVERLAP','OVERLAP-OR-AFTER) modificados, estes serão igual ao relType_pred corretos.
            Poderá ser utilizados para calcular resultados, por exemplo, matrix de confusão
            '''
            if relType_real in ['BEFORE-OR-OVERLAP','OVERLAP-OR-AFTER']:
                if acertou:
                    return relType_pred
                else:
                    if relType_real == 'BEFORE-OR-OVERLAP':
                        return 'BEFORE'
                    elif relType_real == 'OVERLAP-OR-AFTER':
                        return 'AFTER'
            else:
                return relType_real
        
        
        if not self.__check_process_rules():
            return pd.DataFrame(data=None) 
        
        if extra:
            how = 'outer'
        else:
            how = 'left'
        
        #Uni df_real com df_pred
        real_predict = self.df_real.merge(self.df_pred, how = how, on=['task', 'doc', 'isentenca', 'eventID', 'relatedTo'])
        
        #Calcula predições corretas
        real_predict['acertou'] = [ esta_na_lista(row[0], row[1]) for i, row in real_predict[['relType_pred', 'relType_real']].iterrows() ]
        
        #Calcula coluna relType_real_sem_OR
        real_predict['relType_real_sem_OR'] = [ relType_real_sem_OR(row[0], row[1], row[2]) for i, row in real_predict[['relType_pred', 'relType_real', 'acertou']].iterrows() ]
        
        #VERIFICAR se este filtro poderá ser retirado após processar as outras tarefas
        #Filta resultado conforme task
        query = "task == '" + self.task + "'" 
        real_predict = real_predict.query(query)
        
        #Busca valores de eventID
        df_eventID  = self.__tb.df.event_completo[['doc', 'eid', 'text']].rename(columns={'eid':'eventID'})
        real_predict_join = real_predict.merge(df_eventID, how = 'left', on=['doc', 'eventID'])
        
        #Seleciona colunas
        colunas = ['lid_real', 'doc', 'isentenca', 'task', 'relType_real', 'relType_real_sem_OR', 'eventID', 'text_event', 'relatedTo', 'text_relatedTo', 'value', 'relType_pred', 'rule', 'acertou']
            
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
        real_predict_join = real_predict_join.sort_values(['isentenca', 'eventID', 'relatedTo'])
        
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
        
        
    def graph_pred(self, id_sentenca = None):
        '''
        Exibe as relações temporais de forma gráfica da primeira sentença setada.
        
        '''
        if not id_sentenca:
            id_sentenca = self.__id_sentenca[0]
        else:
            id_sentenca = self.__tb.trata_lista(id_sentenca)[0]
        
        relType_true = self.df_real_predict.query("acertou == True and isentenca == " + str(id_sentenca))
        self.__tb.my_tlink.clear()
        for index, row in relType_true.iterrows():
            self.__tb.my_tlink.add(row['relType_pred'], row['eventID'], row['relatedTo'], 'A', row['isentenca'], row['doc'], row['rule'])
        self.__tb.my_tlink.graph_rt()

    def cm(self, extras = True):
        '''
        Matrix de Confusão das predições.
        
        Args:
            extras: se True (default), exibe as predições que não estão anotadas no corpus
                    se False, não considera as predições que não estão anotadas.
        '''
        #extras = predições não anotadas no corpus
        if extras:
            real_predict = self.df_real_predict_extra
        else:
            real_predict = self.df_real_predict
        #Todas predições que foram avaliadas por regras
        real_predict = real_predict[~real_predict['rule'].isna()]
        
        y_teste = real_predict[['doc', 'eventID', 'relatedTo', 'relType_real_sem_OR']].set_index(['doc', 'eventID', 'relatedTo']).fillna('') #.tolist()
        y_pred  = real_predict[['doc', 'eventID', 'relatedTo', 'relType_pred']].set_index(['doc', 'eventID', 'relatedTo']).fillna('')  #.tolist()

        
        #Uni label únicos de ambos dataset
        label = list(set(y_teste['relType_real_sem_OR'].unique().tolist() + y_pred['relType_pred'].unique().tolist()))
        cm = confusion_matrix(y_teste, y_pred, labels=label)
        df_cm = pd.DataFrame(cm, index = label, columns = label)

        plt.figure(figsize=(5,4))
        sns.heatmap(df_cm, cmap='PuBu', annot=True, fmt="d")
        plt.title('Matriz de confusão')
        plt.xlabel('__________ PREDITO __________')
        plt.ylabel('____________ REAL ____________')
        print('\n')

        print(classification_report(y_teste, y_pred, zero_division=0, digits=3))
        
        
    def ct(self, extras = True):
        '''
        Cross Tab.
        
        Args:
            extras: se True (default), exibe as predições que não estão anotadas no corpus
                    se False, não considera as predições que não estão anotadas.
        '''
        #extras = predições não anotadas no corpus
        if extras:
            real_predict = self.df_real_predict_extra
        else:
            real_predict = self.df_real_predict
        #Todas predições que foram avaliadas por regras
        real_predict = real_predict[~real_predict['rule'].isna()]
        
        y_teste = real_predict['relType_real_sem_OR'].fillna('') #.tolist()
        y_pred = real_predict['relType_pred'].fillna('')  #.tolist()

        return pd.crosstab(y_teste, y_pred, rownames=['Real'], colnames=['Pred'], margins=True)
        
        
    @property
    def __quant_total(self):
        df = self.df_real_predict
        df = df[~df['relType_pred'].isna()]
        return df.shape[0]
        
    @property
    def __quant_acerto(self):
        df = self.df_real_predict
        return df[df['acertou'] == True].shape[0]
     
    @property
    def __quant_erro(self):
        return self.__quant_total - self.__quant_acerto
    
    @property
    def __quant_previsao_extra(self):
        df = self.df_real_predict_extra
        return df[df['relType_real'].isna()].shape[0]
    
    @property
    def __pct_acerto(self):
        if self.__quant_total == 0:
            return 0
        return self.__quant_acerto / (self.__quant_total + self.__quant_previsao_extra) * 100
    
    @property
    def __pct_erro(self):
        if self.__quant_total == 0:
            return 0
        return self.__quant_erro / self.__quant_total * 100

    @property
    def df_resultado_geral(self):
        '''
        Retorna resumo do resultado geral.
        
        '''
        #Cria datadets
        df_resultado_geral = pd.DataFrame(data=None, columns = ['Erros', 'Acertos', 'Total', 'Extra', '% Acerto'])
        
        quant_total = self.__quant_total
        quant_acerto = self.__quant_acerto
        quant_previsao_extra = self.__quant_previsao_extra
        quant_erro = self.__quant_erro 
        pct_acerto = self.__pct_acerto
        #pct_erro = self.__pct_erro

        #Adicionar resultado em dataset
        df_resultado_geral = df_resultado_geral.append({'Erros':quant_erro, 'Acertos': quant_acerto, 'Total': quant_total, 'Extra': quant_previsao_extra, '% Acerto':'{0:,.2f}%'.format(pct_acerto)}, ignore_index=True)
                
        return df_resultado_geral
    
    def __df_resultado_gererico(self, indice: list):
        '''
        Resultado das predições das relações temporais
        
        Args:
            indice: lista dos campos que compõem o índice do dataframe
            Extras: Se True exibe o campo Extras, são as relações preditas que não estão anotadas no corpus.
            
        '''
        #Constroi df vazio para não faltar campos caso o resultado não contenha todos os campos
        df0 = pd.DataFrame(data=None, columns=[*indice, 'Erros', 'Acertos'])
        df0 = df0.set_index([*indice])

        #Data inicial agrupado
        df_extra = self.df_real_predict_extra
        if not df_extra.empty:
            df_extra_pred = df_extra[~df_extra['relType_pred'].isna()] #somente os que houve predição (SÓ EXTRA + SÓ TOTAL)
            df1 = df_extra_pred.groupby([*indice, 'acertou'])['lid_real'].count().unstack().fillna(0)
            df1 = df1.rename(columns={False:'Erros', True:'Acertos'})
            df_resultado = df0.append(df1).fillna(0)
        else:
            df_resultado = df0.fillna(0)

        #Calcula o campo Total
        df_total = df_resultado.sum(axis=1).reset_index().rename(columns={0:'Total'})
        df_total = df_total.set_index([*indice])

        #Uni o campo Total do dataframe principal
        df_final = df_resultado.merge(df_total, how='left', on=[*indice])
        df_final.columns.set_names(names='Resultado', inplace=True)

        if df_extra.empty:
            df_final['Extras'] = 0

        #EXTRAS
        #Acrescenta a quantidade de previsões que não estão anotadas no corpus: campo Extras
        if not df_extra.empty:
            df_total_extra = df_extra[df_extra['relType_real'].isna()]
            df_total_extra = df_total_extra.groupby([*indice, 'acertou'])['isentenca'].count().unstack().fillna(0)
            df_total_extra = df_total_extra.rename(columns={False:'Extras'})

            #CRIAR EXTRAS VAZIO
            df_final_com_extra = pd.DataFrame(data=None, columns=[*indice, 'Erros', 'Acertos', 'Total', 'Extras'])
            df_final_com_extra = df_final_com_extra.set_index([*indice])

            df_final_com_extra = df_final_com_extra.append(df_final.merge(df_total_extra, how='left', on=[*indice]) )
            df_final_com_extra.columns.set_names(names='Resultado', inplace=True)

            colunas = ['Erros', 'Acertos', 'Total', 'Extras']
            df_final_com_extra = df_final_com_extra[colunas].fillna(0)
            df_final = df_final_com_extra

        #--------------------------------
        #Calcular % de Acertos
        df_final['pct_acerto'] = (df_final['Acertos'] / (df_final['Total'] + df_final['Extras']) * 100)

        #Ordena por maiores acertos
        df_final.sort_values(['pct_acerto', *indice], ascending=False, inplace=True)
        #df_final = df_final.sort_index()

        #Formata em percentual. Deve executar após ordenação, pois a formatação converte 'pct_acerto' em string
        df_final['pct_acerto'] = df_final['pct_acerto'].map(lambda x: round(x, 2))

        return df_final
    
    
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

 #---------------------------------------------------------------------
 #     FIM RELACAOTEMPORAL
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
        '''
        Registra novos atributos para o Doc e Token.
        
        '''
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
        if not Token.has_extension("tense"):
            Token.set_extension('tense', default='')
        if not Token.has_extension("pos"):
            Token.set_extension('pos', default='')
        
        #Regista atributos apenas de TIMEX3 ao Token
        if not Token.has_extension("tipo"):
            Token.set_extension('tipo', default='')
        if not Token.has_extension("value"):
            Token.set_extension('value', default='')

    def __call__(self, doc: Doc) -> Doc:
        '''
        Retorna Doc processado contendo informações do TimebankPT.
        
        '''
        #Se a sentença não existir no TimeBankPT, sai desse pipe e retorna o Doc original.
        #Mas os atributos devem ser registrados em __init__ para evitar erro ao serem chamados externamente
        dados_sentenca = self.__tb_dict.get(doc.text)

        if not dados_sentenca:
            return doc
        
        id_sentenca = dados_sentenca['isentenca']
        dct_doc = dados_sentenca['dct']
        nome_doc = dados_sentenca['doc'] 
        dct_tid = dados_sentenca['tid'] 
        dct_type = dados_sentenca['type'] 
        lista_event = dados_sentenca['lista_event']
        lista_timex3 = dados_sentenca['lista_timex3']
        
        
        ent_timebank = []
        ent_doc = []

        #Atribui valor para os novos atributos do Doc
        doc._.dct = dct_doc
        doc._.nome = nome_doc
        doc._.id_sentenca = id_sentenca
        doc._.dct_tid = dct_tid
        doc._.dct_type = dct_type
       
        with doc.retokenize() as retokenizer:

            #EVENT
            for tag, isentenca, eid, text, start, end, classe, aspecto, tense, pos in lista_event:

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
                        token._.tense = tense
                        token._.pos = pos

                #else:
                    #print('id:', isentenca, tag,  '  None span:', text)



            #TIMEX3
            for tag, isentenca, tid, text, start, end, tipo, value in lista_timex3:

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
