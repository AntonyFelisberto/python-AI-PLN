import re
import spacy
import string
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from spacy.training import Example
from sklearn.metrics import confusion_matrix, accuracy_score
from spacy.lang.pt.stop_words import STOP_WORDS


historico = []
pontuacao = string.punctuation
pln = spacy.load("pt_core_news_sm")
modelo = spacy.blank("pt")
textcat = modelo.add_pipe("textcat")

def pre_processamentos(texto):
    documento = pln(texto.lower())
    lista = []
    for token in documento:
        #lista.append(token.text)
        lista.append(token.lemma_)

    lista = [palavra for palavra in lista if palavra not in STOP_WORDS and palavra not in pontuacao]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    return lista

base_dados_teste = pd.read_csv("AI pln\\base_dados_tweeter\\TrainingDatasets\\Train500.csv",delimiter=";",encoding="UTF-8")
base_dados_teste["texto"] = base_dados_teste["texto"].apply(pre_processamentos)
print(sns.countplot(base_dados_teste["sentiment"],label="contagem"))
previsoes = []

for texto in base_dados_teste["texto"]:
    previsao = modelo_carregado(texto)
    previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
    if previsao["ALEGRIA"] > previsao["MEDO"]:
        previsoes_final.append("alegria")
    else:
        previsoes_final.append("alegria")

previsoes_final = np.array(previsoes_final)
respostas_reais = base_dados["emocao"].values
print(accuracy_score(respostas_reais, previsoes_final))
cm = confusion_matrix(respostas_reais,previsoes_final)
print(cm)