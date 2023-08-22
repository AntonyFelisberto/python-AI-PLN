import spacy
import nltk
from spacy import displacy
from spacy.lang.pt.stop_words import STOP_WORDS

nltk.download("rslp")
pln = spacy.load('pt_core_news_sm')

texto = pln("estou aprendendo processamento de linguagem natural")
for texto_um in texto:
    print("tokenizacao: ",texto_um)

textos = "estou aprendendo processamento de linguagem natural"
print("sem tokenizacao: ",textos.split(" "))