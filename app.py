from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F

app = Flask(__name__)

# ========= LOAD BERT MODEL ===========
MODEL_DIR = "bert_emotion_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

emotion_labels = ['Calm', 'Energetic', 'Happy', 'Relaxed', 'Sad', 'Stressed']

# ========= LOAD DATASETS ==========
music = pd.read_csv("real_spotify_music_dataset.csv")
podcast = pd.read_csv("synthetic_podcast_dataset.csv")

music["emotion"] = music["emotion"].astype(str).str.strip().str.capitalize()
podcast["mood_tag"] = podcast["mood_tag"].astype(str).str.strip().str.capitalize()

# ========= EMOTION PREDICTION =======
def bert_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        prob = F.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1).item()

    return emotion_labels[pred]

# ========= RECOMMENDATIONS ===========
def recommend_music(emotion):
    df = music[music["emotion"] == emotion]
    return df.sample(min(5, len(df))).to_dict(orient="records") if not df.empty else []

def recommend_podcast(emotion):
    df = podcast[podcast["mood_tag"] == emotion]
    return df.sample(min(5, len(df))).to_dict(orient="records") if not df.empty else []

# ========= SERVE FRONTEND ===========
@app.route("/")
def frontend():
    return send_from_directory(".", "index.html")

# ========= API ===========
@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.json.get("text", "")
    emotion = bert_predict(user_text)

    return jsonify({
        "emotion": emotion,
        "music": recommend_music(emotion),
        "podcasts": recommend_podcast(emotion)
    })

# ========= RUN ===========
if __name__ == "__main__":
    app.run(debug=True)
