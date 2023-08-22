import spacy
import nltk

nltk.download("rslp")
pln = spacy.load('pt_core_news_sm')

documento = pln("Aprendendo processamento de linguagem natural")

[token.lemma_ for token in documento]

stemmer = nltk.stem.RSLPStemmer()
print(stemmer.stem("aprender"))

for token in documento:
    print(token.text,token.lemma_,stemmer.stem(token.text))