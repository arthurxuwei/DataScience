import spacy
nlp = spacy.load('en_core_web_sm')

txt = "some text read from one paper ..."
doc = nlp(txt)

for sent in doc.sents:
    print('sent: ')
    print(sent)
    print('#'*50)
