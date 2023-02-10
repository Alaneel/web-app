import numpy as np
import os
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("./module2.pkl", "rb"))


@app.route("/")
def home(): 
    return render_template("template.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template(
        "template.html", prediction_text="Predicted Price: {}".format(output)
    )


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 8081)),host='0.0.0.0',debug=True)