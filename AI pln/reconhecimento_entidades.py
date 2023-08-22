import spacy
import nltk
from spacy import displacy

nltk.download("rslp")
pln = spacy.load('pt_core_news_sm')

stemmer = nltk.stem.RSLPStemmer()

texto = 'A IBM é uma empresa dos Estados Unidos voltada para a área de informática. Sua sede no Brasil fica em São Paulo e a receita em 2018 foi de aproximadamente 320 bilhões de reais'

documento = pln(texto)

displacy.render(documento,style='ent',jupyter=False)

for entidade in documento.ents:
    print(entidade.text,entidade.label_)

texto = 'Bill Gates nasceu em Seattle em 28/10/1955 e foi o criador da Microsoft'

documento = pln(texto)

for entidade in documento.ents:
    print(entidade.text,entidade.label_)

for entidade in documento.ents:
    if entidade.label_ == "PER":
        print(entidade.text)