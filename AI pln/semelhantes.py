import spacy
import nltk
from spacy import displacy
from spacy.lang.pt.stop_words import STOP_WORDS

nltk.download("rslp")
pln = spacy.load('pt_core_news_sm')

p_um = pln("olá")
p_dois = pln("oi")
p_tres = pln("ou")

print(p_um.similarity(p_dois)) #quanto maior o valor maior a similaridade
print(p_dois.similarity(p_um))
print(p_um.similarity(p_tres))

texto = pln("quando sera lançado o novo filme")
texto_dois = pln("o novo filme vai sair mes que vem")
text_tres = pln("qual o nome daquele carro")

print(texto.similarity(texto_dois)) #quanto maior o valor maior a similaridade
print(texto_dois.similarity(texto))
print(texto.similarity(text_tres))

texto = pln("gato cachorro cavalo pessoa")
for texto_um in texto:
    for textos_dois in texto:
        similaridade = int(texto_um.similarity(textos_dois)*100)
        print(f"{texto_um} é {similaridade} similar a {textos_dois}")