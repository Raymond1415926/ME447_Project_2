import spacy

# Load the pre-trained model
nlp = spacy.load('en_core_web_trf')
# Define the text to extract keywords from
with open("test_text.txt", "r", encoding="utf-8") as f:
    text = f.read()


# Process the text with the NER model
doc = nlp(text)

# Extract the keywords
keywords = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT', 'SKILL', 'TECH']]

kw_set = set(keywords) #delete duplicates
keywords = list(kw_set)


test_result = "Result for small spacy model: \n \n"
for keyword in keywords:
    test_result += keyword + "\n"

test_result += "\n"

# Print the keywords
with open("test_results.txt", "a", encoding="utf-8") as f:
    f.write(test_result)
