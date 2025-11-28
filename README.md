# 📘 NLP Text Preprocessing Pipeline 
*A complete end-to-end text cleaning & normalization workflow using Python, pandas, and NLTK*  

---

## 🧠 1. Project Overview

This project demonstrates a full **Natural Language Processing (NLP) preprocessing pipeline**, applied to a dataset of hotel reviews from TripAdvisor.

The goal is to transform raw text into a clean, structured, machine-ready format suitable for:

- Sentiment analysis  
- Topic modeling  
- Keyword extraction  
- N-gram analysis  
- Feature engineering  
- Machine learning model training

This project helped me deepen my understanding of pandas, lambda functions, tokenization, stemming, lemmatization, and regex cleaning used in real NLP systems.

---

## 🎯 2. Motivation

Real-world NLP requires text to be:

✔ normalized  
✔ consistent  
✔ punctuation-free  
✔ tokenized  
✔ stripped of noise  
✔ linguistically processed

This pipeline mimics what companies do before sending text into ML models or vectorizers.

---

## 🔧 3. Technology Stack

| Component | Description |
|----------|-------------|
| **Python 3.12** | Primary language |
| **pandas** | Data manipulation |
| **NLTK** | Tokenization, stopwords, stemming, lemmatization, n-grams |
| **regex (re)** | Text pattern cleaning |
| **Jupyter / Codespaces** | Development environment |

---

## 📂 4. Dataset
tripadvisor_hotel_reviews.csv

## 🚀 5. Preprocessing Pipeline Diagram
Raw Text
↓
Lowercase
↓
Stopword Removal (keep "not")
↓
Regex Replace: "*" → "star"
↓
Punctuation Removal
↓
Tokenization
↓
Stemming
↓
Lemmatization
↓
Flatten Tokens
↓
N-gram Extraction


---

## 🔬 6. Detailed Step-by-Step Processing


**6.1 Lowercasing

```python
data['review_lowercase'] = data['Review'].str.lower()

### **6.2 Stopword Removal (keep “not”)**
en_stopwords = stopwords.words('english')
en_stopwords.remove('not')

data['review_no_stopwords'] = data['review_lowercase'].apply(
    lambda x: ' '.join([w for w in x.split() if w not in en_stopwords])
)

### **6.3 Replace “*” with “star”**
data['review_no_stopwords_no_punct'] = data.apply(
    lambda x: re.sub(r"[*]", "star", x['review_no_stopwords']),
    axis=1
)

### **6.4 Remove punctuation**
data['review_no_stopwords_no_punct'] = data.apply(
    lambda x: re.sub(r"([^\w\s])", "", x['review_no_stopwords_no_punct']),
    axis=1
)

### **6.5 Tokenization**
data['tokenized'] = data.apply(
    lambda x: word_tokenize(x['review_no_stopwords_no_punct']),
    axis=1
)

### **6.6 Stemming**
ps = PorterStemmer()
data['stemmed'] = data['tokenized'].apply(
    lambda tokens: [ps.stem(t) for t in tokens]
)

### **6.7 Lemmatization**
lemmatizer = WordNetLemmatizer()
data['lemmatized'] = data['tokenized'].apply(
    lambda tokens: [lemmatizer.lemmatize(t) for t in tokens]
)

### **6.8 Flatten tokens**
tokens_clean = sum(data['lemmatized'], [])

### **6.9 N-gram Extraction**
Unigrams
unigrams = pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()
print(unigrams.head(20))

Bigrams
bigrams = pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()
print(bigrams.head(20))





