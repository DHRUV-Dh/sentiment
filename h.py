from flask import Flask, request, jsonify, render_template
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load the model and tokenizer
        model = load_model("/Users/apple/Desktop/Sentiment-Analysis/Models/sentiment_analysis_lstm_model.h5")
        tokenizer = pickle.load(open("/Users/apple/Desktop/Sentiment-Analysis/Models/tokenizer.pickle", "rb"))
        
        # Preprocess the text input
        text_input = request.json["text"]
        text_input = preprocess_text(text_input)
        
        # Tokenize the text input
        sequences = tokenizer.texts_to_sequences([text_input])
        
        # Padding sequences to ensure uniform length
        max_sequence_length = 118  # Update with your model's expected sequence length
        padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length)
        
        # Make predictions
        prediction = model.predict(padded_sequence)
        
        # Convert prediction to a single value
        predicted_class = "Positive" if prediction[0][0] > 0.5 else "Negative"
        
        response_json = {"prediction": predicted_class}
        return jsonify(response_json)
    
    except Exception as e:
        return jsonify({"error": str(e)})


def preprocess_text(text):
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    return review


if __name__ == "__main__":
    app.run(port=5000, debug=True)
