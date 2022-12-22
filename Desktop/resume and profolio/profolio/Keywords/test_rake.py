import spacy

nlp = spacy.load("en_core_web_trf")

# Process some text
doc = nlp("Apple is a technology company based in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.")

# Iterate over the entities in the document
for ent in doc.ents:
    print(ent.text, ent.label_)
