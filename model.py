import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Langkah 1: Persiapan Data dan Analisis Data
data = pd.read_excel('dataset/Diagnosa.xlsx')

# Langkah 2: Preprocessing Teks dengan Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    return stemmer.stem(text)

data['Gejala_bersih'] = data['Gejala'].apply(preprocess_text)

# Membangun matriks TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Gejala_bersih'])

# Langkah 3: Implementasi Model dan Prediksi untuk Data Baru
y_penyakit = data['Penyakit']
y_risiko = data['Risiko']
y_penanganan = data['Penanganan']

# Membuat model Random Forest untuk setiap label
rf_classifier_penyakit = RandomForestClassifier()
rf_classifier_penyakit.fit(tfidf_matrix, y_penyakit)

rf_classifier_risiko = RandomForestClassifier()
rf_classifier_risiko.fit(tfidf_matrix, y_risiko)

rf_classifier_penanganan = RandomForestClassifier()
rf_classifier_penanganan.fit(tfidf_matrix, y_penanganan)

# Simpan model dan vektorizer ke dalam file .pkl
joblib.dump({
    'tfidf_vectorizer': tfidf_vectorizer,
    'rf_classifier_penyakit': rf_classifier_penyakit,
    'rf_classifier_risiko': rf_classifier_risiko,
    'rf_classifier_penanganan': rf_classifier_penanganan
}, 'model/model_and_vectorizer2   .pkl')
