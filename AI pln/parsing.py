import spacy
import nltk
from spacy import displacy
from spacy.lang.pt.stop_words import STOP_WORDS

nltk.download("rslp")
pln = spacy.load('pt_core_news_sm')

documento = pln("reserve uma passagem saindo de guarulhos e chegando em curitiba")

origem,destino = documento[5],documento[9]

print(list(origem.ancestors))
print(list(destino.ancestors))

print(documento[0].is_ancestor(documento[2]))

documento = pln("reserva de uma mesa para o restaurante e de um taxi para o hotel")
tarefas = documento[3],documento[10]
locais = documento[6],documento[13]
print(tarefas,locais)

for local in locais:
    print("------",local)
    for objeto in local.ancestors:
        print(objeto)

for local in locais:
    for objeto in local.ancestors:
        if objeto in tarefas:
            print(f"reserva de {objeto} para o {local}")
            break

print(list(documento[6].children))

print(list(documento[3].ancestors))
print(list(documento[3].children))
#displacy.serve(documento,style="dep",options={"distance":90})

documento = pln("que locais podemos visitar em curitiba e para ficar em guarulhos ?")
lugares = documento[5],documento[10]    #pode fazer identificando os verbos e pronomes para deixar automatico
acoes = documento[3],documento[8]
print(lugares,acoes)

for local in lugares:
    for acao in local.ancestors:
        if acao in acoes:
            print(f"{local} para {acao}")
            break

displacy.serve(documento,style="dep")        