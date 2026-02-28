from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# ✅ Homepage route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.json["message"]
    data = cv.transform([message])
    result = model.predict(data)[0]
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)