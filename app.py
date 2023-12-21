from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

app = Flask(__name__)

# Memuat model hanya sekali saat aplikasi Flask dijalankan
model = load_model('model.h5')

# Memuat dataset
file_path = 'dataset_pathway - data.csv'  # Sesuaikan dengan path dataset Anda
df = pd.read_csv(file_path)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
feature_vec = vectorizer.fit_transform(df['Teks'].values)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['Label'].values)

# Membuat pemetaan jurusan ke karir yang sesuai
def map_jurusan_to_karir(df):
    mapping = {}
    for index, row in df.iterrows():
        label = row['Label']
        karir = row['Karir yang Sesuai'].split(', ')
        if label not in mapping:
            mapping[label] = karir
        else:
            mapping[label].extend(karir)
    return mapping

mapping_jurusan_to_karir = map_jurusan_to_karir(df)

# Endpoint untuk melakukan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari permintaan (request)
        data = request.get_json(force=True)

        # Validasi input
        if 'text' not in data:
            return jsonify({"error": "Field 'text' diperlukan"}), 400

        user_input = data['text']

        # Transformasi narasi pengguna menjadi representasi vektor TF-IDF
        user_tfidf = vectorizer.transform([user_input])

        # Prediksi jurusan berdasarkan narasi pengguna
        predicted_probabilities = model.predict(user_tfidf.toarray())
        predicted_label = tf.argmax(predicted_probabilities, axis=1).numpy()[0]
        predicted_label_string = label_encoder.inverse_transform([predicted_label])[0]

        # Identifikasi kata kunci dalam kalimat input
        input_keywords = [word.lower() for word in user_input.split()]

        # Ambil similarity antara narasi pengguna dengan semua data
        user_similarity = cosine_similarity(user_tfidf, feature_vec)

        # Dapatkan similarity dengan semua data kecuali data pengguna
        all_similarity_except_user = cosine_similarity(feature_vec, feature_vec)

        # Sesuaikan similarity berdasarkan kata kunci dalam kalimat input
        adjusted_similarity = all_similarity_except_user[0].copy()
        for i, word in enumerate(df['Teks'].values):
            for keyword in input_keywords:
                if keyword in word.lower():
                    adjusted_similarity[i] *= 1.5

        index_similar_texts = adjusted_similarity.argsort()[::-1]

        # Dapatkan rekomendasi jurusan
        recommended_labels = []
        similarities = []

        for idx in index_similar_texts:
            if len(recommended_labels) >= 5:
                break

            if idx != 0:
                label = labels[idx]
                label_string = label_encoder.inverse_transform([label])[0]
                if label_string not in recommended_labels:
                    recommended_labels.append(label_string)
                    similarities.append(all_similarity_except_user[0][idx])

        # Dapatkan rekomendasi karir
        rekomendasi_karir = []
        if predicted_label_string in mapping_jurusan_to_karir:
            rekomendasi_karir = mapping_jurusan_to_karir[predicted_label_string][:10]

        # Format output
        output = {
            "Rekomendasi Jurusan": predicted_label_string,
            # "recommended_labels": recommended_labels[:3],
            # "similarities": similarities[:3],
            "Rekomendasi Karir": rekomendasi_karir
        }

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
