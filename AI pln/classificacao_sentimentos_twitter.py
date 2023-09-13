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
textcat.add_label("MEDO")
textcat.add_label("ALEGRIA")

exemplo_base_dados = [
    ["este trabalho é agradavel",{"POSITIVO":True,"NEGATIVO":False}],
    ["este lugar é assustador",{"POSITIVO":False,"NEGATIVO":True}]
]

def pre_processamentos(texto):
    lista_emoticons = {
        ":)":"emocaopositiva",
        ":d":"emocaopositiva",
        ":(":"emocaonegativa",
    }
    texto = texto.lower()
    texto = re.sub(r"@[A-Za-z0-9$-_@.&+]+"," ",texto)
    texto = re.sub(r"https?://[A-Za-z0-9./]+"," ",texto)
    texto = re.sub(r" +"," ",texto)

    for emocao in lista_emoticons:
        texto = texto.replace(emocao,lista_emoticons[emocao])

    documento = pln(texto)
    lista = []
    for token in documento:
        lista.append(token.lemma_)

    lista = [palavra for palavra in lista if palavra not in STOP_WORDS and palavra not in pontuacao]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    return lista

base_treinamento = pd.read_csv("AI pln\\base_dados_tweeter\\TrainingDatasets\\Train500.csv",delimiter=";",encoding="UTF-8")
print(sns.countplot(base_treinamento["sentiment"],label="contagem"))
base_treinamento.drop(["id","tweet_date","query_used"],axis=1,inplace=True)

base_teste = pd.read_csv("AI pln\\base_dados_tweeter\\TestDatasets\\Test.csv",delimiter=";")
print(sns.countplot(base_teste["sentiment"],label="contagem"))
base_teste.drop(["id","tweet_date","query_used"],axis=1,inplace=True)

text_test = base_treinamento["tweet_text"][1]
resultado = pre_processamentos(text_test)

base_treinamento["tweet_text"] = base_treinamento["tweet_text"].apply(pre_processamentos)
base_teste["tweet_text"] = base_treinamento["tweet_text"].apply(pre_processamentos)

base_dados_final = []
for text,emocao in zip(base_treinamento["tweet_text"],base_treinamento["sentiment"]):
    if emocao == 1:
        dic = ({"POSITIVO":False,"NEGATIVO":True})
    elif emocao == 0:
        dic = ({"POSITIVO":False,"NEGATIVO":True})
    base_dados_final.append([text,dic.copy()])

modelo.begin_training()
for epoca in range(1000):
    random.shuffle(base_dados_final)
    losses = {}
    for batch in spacy.util.minibatch(base_dados_final,256):
        textos = [modelo(texto) for texto,entities in batch]
        annotations = [{"cats":entities} for texto,entities in batch]
        examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(
            textos, annotations
        )]
        modelo.update(examples, losses=losses)
    if epoca % 5 == 0 :
        print(losses)
        historico.append(losses)

historico_loss = []
for i in historico:
    historico_loss.append(i.get("textcat"))

historico_loss = np.array(historico_loss)
print(historico_loss)

plt.plot(historico_loss)
plt.title("Progressão de erro")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.show()

modelo.to_disk("modelo_tweet")

modelo_carregado = spacy.load("modelo_tweet")
texto = base_teste["tweet_text"][21]
previsao = modelo_carregado(texto)
print(previsao.cats)

texto = "eu tenho medo dele"
texto_processado = pre_processamentos(texto)
previsao = modelo_carregado(texto)
print(previsao.cats)

texto = base_teste["tweet_text"][4000]
previsao = modelo_carregado(texto)
print(previsao.cats)

previsoes = []
for texto in base_treinamento["tweet_text"]:
    previsao = modelo_carregado(texto)
    previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
    if previsao["POSITIVO"] > previsao["NEGATIVO"]:
        previsoes_final.append(1)
    else:
        previsoes_final.append(0)

previsoes_final = np.array(previsoes_final)
respostas_reais = base_treinamento["sentiment"].values
print(accuracy_score(respostas_reais, previsoes_final))
cm = confusion_matrix(respostas_reais,previsoes_final)
print(cm)

print(sns.heatmap(cm,annot=True))

previsoes = []
for texto in base_teste["tweet_text"]:
    previsao = modelo_carregado(texto)
    previsoes.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
    if previsao["POSITIVO"] > previsao["NEGATIVO"]:
        previsoes_final.append(1)
    else:
        previsoes_final.append(0)

previsoes_final = np.array(previsoes_final)
respostas_reais = base_teste["sentiment"].values
print(accuracy_score(respostas_reais, previsoes_final))
cm = confusion_matrix(respostas_reais,previsoes_final)
print(cm)

print(sns.heatmap(cm,annot=True))
