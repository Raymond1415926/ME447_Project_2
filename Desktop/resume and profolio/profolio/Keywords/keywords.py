import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


# Define a function to extract keywords from a job description
def extract_keywords(text, classifier, vectorizer):
    # Pre-process the text
    text = text.lower()  # lowercase all words
    text = nltk.regexp_tokenize(text, r'\w+')  # remove punctuation and split into words
    text = [word for word in text if word not in stopwords.words('english')]  # remove stopwords
    lemmatizer = WordNetLemmatizer()  # create a lemmatizer
    text = [lemmatizer.lemmatize(word) for word in text]  # lemmatize words

    # Vectorize the text
    X = vectorizer.transform([text])

    # Predict the relevance of each word
    y_pred = classifier.predict(X)

    # Extract the most relevant words
    top_words = [word for word, label in zip(text, y_pred) if label == 1]

    # Map synonyms to keywords
    synonym_mapping = {
        "finite element analysis": "FEA",
        "FEA": "finite element analysis"
    }
    keywords = []
    for word in top_words:
        if word in synonym_mapping:
            keywords.append(synonym_mapping[word])
        else:
            keywords.append(word)

    return keywords


# Load the trained classifier and vectorizer
# classifier = SVC.load("product_designer_classifier.pkl")
# vectorizer = TfidfVectorizer.load("product_designer_vectorizer.pkl")
#
# # Test the function with a sample job description
# text = "We are seeking a Product Designer with experience in design for manufacturability and Finite Element Analysis (FEA) to join our team. The ideal candidate will have a strong background in design and manufacturing, as well as skills in SolidWorks and Abaqus. Knowledge of DG&T and CMA-ES is a plus. Experience in PCB design is also desired."
# keywords = extract_keywords(text, classifier, vectorizer)
# print(keywords)
