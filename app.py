from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = pickle.load(open("fish_model.pkl", "rb"))

# Load dataset to restore the original species names
df = pd.read_csv("Fish.csv")  # Ensure this CSV is in the same directory

# Fit LabelEncoder with species names
label_encoder = LabelEncoder()
df["Species"] = label_encoder.fit_transform(df["Species"])

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("app_rf.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    input_features = [float(request.form[key]) for key in ["Weight", "length1", "length2", "length3", "height", "width"]]
    
    # Predict species (numeric value)
    predicted_label = model.predict([input_features])[0]

    # Convert numeric prediction to species name
    predicted_species = label_encoder.inverse_transform([predicted_label])[0]

    return f"Predicted Fish Species: {predicted_species}"

if __name__ == "__main__":
    app.run(debug=True)
