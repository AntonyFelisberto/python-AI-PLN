import spacy

pln = spacy.load('pt_core_news_sm')

documento = pln("Aprendendo processamento de linguagem natural")

for token in documento:
    if token.pos_ == "PROPN":
        print(token.text)

for token in documento:
    print(token.text, token.lemma_,token.pos_,token.tag_,token.dep_,token.shape_,token.is_alpha, token.is_stop)