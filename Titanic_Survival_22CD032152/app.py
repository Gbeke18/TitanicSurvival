from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "titanic_survival_model.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        pclass = int(request.form["pclass"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])
        sex = request.form["sex"]
        embarked = request.form["embarked"]

        # Encode Sex (matches training)
        sex_male = 1 if sex == "male" else 0

        # Encode Embarked (drop_first=True logic)
        embarked_q = 1 if embarked == "Q" else 0
        embarked_s = 1 if embarked == "S" else 0
        # Embarked_C -> both 0

        # Final feature order MUST match training
        input_data = np.array([[
            pclass,
            sibsp,
            parch,
            fare,
            sex_male,
            embarked_q,
            embarked_s
        ]])


        result = model.predict(input_data)[0]
        prediction = "🟢 Survived" if result == 1 else "🔴 Did Not Survive"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
