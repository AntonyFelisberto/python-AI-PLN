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

base_dados = pd.read_csv("AI pln\\base_dados\\base_treinamento.txt",encoding="UTF-8")
exemplo_base_dados = [
    ["este trabalho é agradavel",{"ALEGRIA":True,"MEDO":False}],
    ["este lugar é assustador",{"ALEGRIA":False,"MEDO":True}]
]

print(base_dados.shape)
print(base_dados.head())
print(base_dados.tail())
sns.countplot(x=base_dados["emocao"],label="Conts")

def pre_processamentos(texto):
    documento = pln(texto.lower())
    lista = []
    for token in documento:
        #lista.append(token.text)
        lista.append(token.lemma_)

    lista = [palavra for palavra in lista if palavra not in STOP_WORDS and palavra not in pontuacao]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    return lista

teste = pre_processamentos("estou aprendendo processamento de linguagem")
print(teste)

base_dados["texto"] = base_dados["texto"].apply(pre_processamentos)

base_dados_final = []
for text,emocao in zip(base_dados["texto"],base_dados["emocao"]):
    if emocao == "alegria":
        dic = ({"ALEGRIA":True,"MEDO":False})
    elif emocao == "medo":
        dic = ({"ALEGRIA":False,"MEDO":True})
    base_dados_final.append([text,dic.copy()])

print(len(base_dados_final))
print(base_dados_final[0])
print(base_dados_final[0][1])
print(base_dados_final[0][0])

modelo.begin_training()
for epoca in range(1000):
    random.shuffle(base_dados_final)
    losses = {}
    for batch in spacy.util.minibatch(base_dados_final,30):
        textos = [modelo(texto) for texto,entities in batch]
        annotations = [{"cats":entities} for texto,entities in batch]
        examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(
            textos, annotations
        )]
        modelo.update(examples, losses=losses)
    if epoca % 100 == 0 :
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

modelo.to_disk("modelo")

modelo_carregado = spacy.load("modelo")
texto = "eu adoro a cor dos seus iris"
texto_processado = pre_processamentos(texto)
previsao = modelo_carregado(texto_processado)
print(previsao.cats)

texto = "eu tenho medo dele"
texto_processado = pre_processamentos(texto)
previsao = modelo_carregado(texto)
print(previsao.cats)


previsoes = []
for texto in base_dados["texto"]:
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

#TESTES COM OUTRA BASE DE DADOS
base_dados_teste = pd.read_csv("AI pln\\base_dados\\base_teste.txt",encoding="UTF-8")
base_dados_teste["texto"] = base_dados_teste["texto"].apply(pre_processamentos)
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
respostas_reais = base_dados_teste["emocao"].values
print(accuracy_score(respostas_reais, previsoes_final))
cm = confusion_matrix(respostas_reais,previsoes_final)
print(cm)