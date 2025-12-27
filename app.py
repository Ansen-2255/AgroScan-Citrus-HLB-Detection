import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import timm
import google.generativeai as genai

# ------------------ CONFIG ------------------
load_dotenv()

# Gemini (chatbot)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    system_instruction=(
        "You are AgroScan AI, a helpful agricultural expert. "
        "You specialize in identifying plant diseases and suggesting solutions."
    )
)

# Flask
app = Flask(__name__)
CORS(app)

# ------------------ ML MODEL ------------------
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 3)

    model.load_state_dict(
        torch.load("citrus_vit_model.pth", map_location=device)
    )

    model.to(device)
    model.eval()
    print("✅ Model loaded")
    return model, device


def predict_image(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_names = ['Canker', 'Greening', 'Healthy']
    return class_names[predicted.item()], confidence.item()


# Load model once (important)
MODEL, DEVICE = load_model()

# ------------------ ROUTES ------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # ✅ THIS IS THE KEY FIX
    image = Image.open(file.stream).convert("RGB")

    result, confidence = predict_image(image, MODEL, DEVICE)

    print("Predicted:", result, "Confidence:", confidence)

    return jsonify({
        "prediction": result,
        "confidence": f"{confidence*100:.2f}%"
    })


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"reply": "Please type a message."})

    try:
        response = chat_model.generate_content(user_msg)
        return jsonify({"reply": response.text})
    except Exception as e:
        print("Gemini error:", e)
        return jsonify({"reply": "AI is currently unavailable."})


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
