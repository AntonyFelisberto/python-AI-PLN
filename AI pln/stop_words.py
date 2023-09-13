import spacy
import nltk
from spacy import displacy
from spacy.lang.pt.stop_words import STOP_WORDS

nltk.download("rslp")
pln = spacy.load('pt_core_news_sm')

print(STOP_WORDS)
print(len(STOP_WORDS))
print(pln.vocab["ir"].is_stop)
print(pln.vocab["caminhar"].is_stop)

documento = pln("A IBM é uma empresa dos Estados Unidos voltada para a área de informática. Sua sede no Brasil fica em São Paulo e a receita em 2018 foi de aproximadamente 320 bilhões de reais")
for token in documento:
    if not pln.vocab[token.text].is_stop:
        print(token.text)