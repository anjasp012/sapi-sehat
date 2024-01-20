import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Langkah 1: Persiapan Data dan Analisis Data
data = pd.read_excel('dataset/sapi.xlsx')
print(data.head())
print(data.describe())

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

# Prediksi untuk data baru
def predict_new_data(new_data):
    new_data_transformed = tfidf_vectorizer.transform([new_data])
    
    # Prediksi untuk setiap label
    prediction_penyakit = rf_classifier_penyakit.predict(new_data_transformed)
    prediction_risiko = rf_classifier_risiko.predict(new_data_transformed)
    prediction_penanganan = rf_classifier_penanganan.predict(new_data_transformed)
    
    return {
        'Penyakit': prediction_penyakit[0],
        'Risiko': prediction_risiko[0],
        'Penanganan': prediction_penanganan[0]
    }

# Contoh prediksi untuk data baru
new_data_to_predict = "Bulu kusam, Gatal, Kelainan kulit, Perkulitan"
prediction_result = predict_new_data(new_data_to_predict)
print("Prediksi untuk data baru:", prediction_result)

# Menghitung akurasi untuk keseluruhan prediksi
def evaluate_total_accuracy(true_labels, predicted_labels):
    correct_predictions = 0
    total_predictions = len(true_labels)
    for i in range(total_predictions):
        if true_labels[i] == predicted_labels[i]:
            correct_predictions += 1
    return (correct_predictions / total_predictions) * 100

# Menggabungkan prediksi dari semua label
all_predicted_labels = {
    'Penyakit': rf_classifier_penyakit.predict(tfidf_matrix),
    'Risiko': rf_classifier_risiko.predict(tfidf_matrix),
    'Penanganan': rf_classifier_penanganan.predict(tfidf_matrix)
}

# Menghitung akurasi keseluruhan
true_labels = {
    'Penyakit': y_penyakit,
    'Risiko': y_risiko,
    'Penanganan': y_penanganan
}

overall_accuracy = evaluate_total_accuracy(true_labels['Penyakit'], all_predicted_labels['Penyakit'])
print("Akurasi keseluruhan:", overall_accuracy, "%")
