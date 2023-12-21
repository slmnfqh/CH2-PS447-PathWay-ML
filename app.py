from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model and other necessary objects
model = tf.keras.models.load_model('model.h5')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
# label_karir_mapping = joblib.load('label_encoder.pkl')  # Load label_karir_mapping

# Function to make predictions and get career recommendations
def make_predictions(user_input):
    # Transform the input text using the provided TF-IDF Vectorizer
    user_tfidf = tfidf_vectorizer.transform([user_input])

    # Make predictions using the loaded model
    predicted_probabilities = model.predict(user_tfidf.toarray())
    predicted_label = tf.argmax(predicted_probabilities, axis=1).numpy()[0]
    predicted_label_string = label_encoder.inverse_transform([predicted_label])[0]

    # Get career recommendations
    # rekomendasi_karir_predicted = label_karir_mapping.get(predicted_label_string, [])[:10]

    return predicted_label_string
    # , rekomendasi_karir_predicted
# API endpoint for making predictions and getting career recommendations
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        user_input = data['user_input']

        # Make predictions and get career recommendations
        predicted_label_string = make_predictions(user_input)
        # , rekomendasi_karir

        # Return the predictions and career recommendations as JSON
        return jsonify({'Prediksi Jurusan': predicted_label_string })
    # 'career_recommendations': rekomendasi_karir

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


