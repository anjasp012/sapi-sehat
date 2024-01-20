from flask import Flask, render_template, request, redirect, url_for, flash, session
import joblib
from sklearn.metrics import accuracy_score
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from flask_mysqldb import MySQL

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Konfigurasi MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'sapi_sehat'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

loaded_data = joblib.load('model/model_and_vectorizer.pkl')

tfidf_vectorizer = loaded_data['tfidf_vectorizer']
rf_classifier_penyakit = loaded_data['rf_classifier_penyakit']
rf_classifier_risiko = loaded_data['rf_classifier_risiko']
rf_classifier_penanganan = loaded_data['rf_classifier_penanganan']

# Fungsi untuk memprediksi gejala baru
def predict_symptoms(symptoms):
    new_data_transformed = tfidf_vectorizer.transform([symptoms])

    # Prediksi untuk setiap label
    prediction_penyakit = rf_classifier_penyakit.predict(new_data_transformed)
    prediction_risiko = rf_classifier_risiko.predict(new_data_transformed)
    prediction_penanganan = rf_classifier_penanganan.predict(new_data_transformed)

    return {
        'penyakit': prediction_penyakit[0],
        'risiko': prediction_risiko[0],
        'penanganan': prediction_penanganan[0]
    }

# Fungsi untuk menghitung akurasi dari hasil klasifikasi
def calculate_accuracy():
    # Load data
    data = pd.read_excel('dataset/sapi_indo.xlsx')

    # Preprocessing dengan Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def preprocess_text(text):
        return stemmer.stem(text)

    data['Gejala_bersih'] = data['gejala'].apply(preprocess_text)
    tfidf_matrix = tfidf_vectorizer.transform(data['Gejala_bersih'])

    # Menggabungkan prediksi dari semua label
    all_predicted_labels = {
        'penyakit': rf_classifier_penyakit.predict(tfidf_matrix),
        'risiko': rf_classifier_risiko.predict(tfidf_matrix),
        'penanganan': rf_classifier_penanganan.predict(tfidf_matrix)
    }

    # Menghitung akurasi keseluruhan
    true_labels = {
        'penyakit': data['penyakit'],
        'risiko': data['risiko'],
        'penanganan': data['penanganan']
    }

    accuracy_penyakit = accuracy_score(true_labels['penyakit'], all_predicted_labels['penyakit'])
    accuracy_risiko = accuracy_score(true_labels['risiko'], all_predicted_labels['risiko'])
    accuracy_penanganan = accuracy_score(true_labels['penanganan'], all_predicted_labels['penanganan'])

    # Hitung akurasi keseluruhan dari semua label
    overall_accuracy = (accuracy_penyakit + accuracy_risiko + accuracy_penanganan) / 3

    return round(overall_accuracy * 100, 2)

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk halaman diagnosa
@app.route('/diagnosa')
def diagnosa():
    return render_template('diagnosa.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        prediction_result = predict_symptoms(symptoms)
        accuracy_result = calculate_accuracy()
        return render_template('diagnosa.html', result=prediction_result, accuracy=accuracy_result)

if __name__ == '__main__':
    app.run(debug=True)
