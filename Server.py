from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import sklearn


load_clf = pickle.load(open('heart_dis_pred.pkl1', 'rb'))


app = Flask(__name__)
CORS(app)

# print("version", sklearn.__version__)

df = pd.read_csv("heart-disease-problem.csv")
X = df.drop("target", axis=1)
data = []
colms = []
print(X)

for col in X.columns:
    X[col].mean()
    colms.append(col)
    data.append({"col": col,
                 "mean":  X[col].mean()})


@app.route("/send",  methods=["get"])
def send():
    return jsonify({"data": data})


@app.route("/accept", methods=["post"])
def accept():
    data = request.get_json()

    for i in range(len(data)):
        if(data[i] == ""):
            data[i] = df[colms[i]].mean()

    data = [[float(i) for i in data]]
    pred = load_clf.predict(data)
    pred = pred.tolist()
    return jsonify({"data": pred})


@app.route("/csv", methods=["post", "get"])
def csv_check():

    inp_col = []

    try:
        f = request.files["myFile"]

        dfx = pd.read_csv(f)

        if(dfx.isnull().sum().sum() > 0):
            dfx = dfx.fillna(df.mean())
        for col in dfx.columns:
            inp_col.append(col)
    except:
        return jsonify({"data": "wrong file type"})

    if(inp_col == colms):
        try:
            pred = load_clf.predict(dfx)
            pred = pred.tolist()
            return jsonify({"data": pred})

        except:

            return jsonify({"data": "eror occurs model cannot predict"})
    else:
        return jsonify({"data": "enter a valid csv file"})


if __name__ == "main":
    app.run(debug=True)


# flask run
# set FLASK_APP=Server.py
