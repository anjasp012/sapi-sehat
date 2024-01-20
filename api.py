from flask import Flask, render_template, jsonify, request
import joblib
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Memuat model dan vektorizer
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
        'Penyakit': prediction_penyakit[0],
        'Risiko': prediction_risiko[0],
        'Penanganan': prediction_penanganan[0]
    }

# Fungsi untuk menghitung akurasi dari hasil klasifikasi
def calculate_accuracy():
    # Load dataset
    data = pd.read_excel('dataset/sapi.xlsx')

    # Preprocessing dengan Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def preprocess_text(text):
        return stemmer.stem(text)

    data['Gejala_bersih'] = data['Gejala'].apply(preprocess_text)
    tfidf_matrix = tfidf_vectorizer.transform(data['Gejala_bersih'])

    # Menggabungkan prediksi dari efsemua label
    all_predicted_labels = {
        'Penyakit': rf_classifier_penyakit.predict(tfidf_matrix),
        'Risiko': rf_classifier_risiko.predict(tfidf_matrix),
        'Penanganan': rf_classifier_penanganan.predict(tfidf_matrix)
    }

    # Menghitung akurasi keseluruhan
    true_labels = {
        'Penyakit': data['Penyakit'],
        'Risiko': data['Risiko'],
        'Penanganan': data['Penanganan']
    }

    accuracy_penyakit = accuracy_score(true_labels['Penyakit'], all_predicted_labels['Penyakit'])
    accuracy_risiko = accuracy_score(true_labels['Risiko'], all_predicted_labels['Risiko'])
    accuracy_penanganan = accuracy_score(true_labels['Penanganan'], all_predicted_labels['Penanganan'])

    # Hitung akurasi keseluruhan dari semua label
    overall_accuracy = (accuracy_penyakit + accuracy_risiko + accuracy_penanganan) / 3

    return round(overall_accuracy * 100, 2)

@app.route('/', methods=['GET'])
def welcome():
    # return 'Selamat Datang di API Diagnosa Sapi'
    return render_template('api.html')

@app.route('/diagnosa-sapi', methods=['POST'])
def predict_symptoms_api():
    if request.method == 'POST':
        symptoms = request.json['gejala']
        prediction_result = predict_symptoms(symptoms)
        accuracy_result = calculate_accuracy()
        return jsonify({'result': prediction_result, 'accuracy': accuracy_result})

if __name__ == '__main__':
    app.run(debug=True)
