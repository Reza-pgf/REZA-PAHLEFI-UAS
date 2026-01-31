import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. MEMBACA DATASET ASLI
df = pd.read_csv('file/dataset_sentimen (2).csv', sep=';')
df = df.dropna(subset=['review_text', 'sentiment'])

# 2. PROSES PREPROCESSING
factory = StemmerFactory()
stemmer = factory.create_stemmer()
indo_stopwords = set(stopwords.words('indonesian'))

def preprocess(text):
    # Case Folding & Hapus Simbol
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenizing
    tokens = word_tokenize(text)
    # Stopword Removal & Stemming
    tokens = [w for w in tokens if w not in indo_stopwords]
    stemmed = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)

print("Sedang memproses preprocessing data...")
df['clean_text'] = df['review_text'].apply(preprocess)

# --- MENYIMPAN SEMUA HASIL BERSIH ---
df.to_csv('file/dataset_hasil_bersih.csv', index=False, sep=';')

# --- MEMISAHKAN DAN MENYIMPAN PER SENTIMEN ---
# 0.0 = Negatif, 1.0 = Netral, 2.0 = Positif
df[df['sentiment'] == 0.0].to_csv('file/hasil_negatif.csv', index=False, sep=';')
df[df['sentiment'] == 1.0].to_csv('file/hasil_netral.csv', index=False, sep=';')
df[df['sentiment'] == 2.0].to_csv('file/hasil_positif.csv', index=False, sep=';')

print("File CSV untuk semua kategori telah dibuat di folder 'file'.")

# 3. TRAINING & EVALUASI (Syarat UAS)
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)
print(f"Jumlah Data Latih: {len(X_train)}")
print(f"Jumlah Data Uji: {len(X_test)}")
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# 4. TAMPILKAN HASIL DI TERMINAL
print("\n=== EVALUASI MODEL ===")
print(classification_report(y_test, y_pred))
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

# 5. TAMPILKAN GRAFIK
ConfusionMatrixDisplay.from_estimator(model, X_test_vec, y_test, cmap=plt.cm.Blues)
plt.show()