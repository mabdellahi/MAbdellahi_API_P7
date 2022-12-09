from joblib import load
from flask import Flask, jsonify, request, render_template
import json
import pandas as pd
import pickle
import traceback
# from prediction_api import *
import shap
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# API definition
app = Flask(__name__)


# Charger les données
df = pd.read_csv("df_test_output.csv")
print('df columns :', df.columns)


# Charger le modèle

model = pickle.load(open("RandomForestClassifier.pkl", "rb"))


@app.route("/predict/<id_client>")
def predict(id_client):
    data = pd.read_csv("df_test_output.csv")

    print('id_client', id_client)
    ID = int(id_client)

    X = data[data['SK_ID_CURR'] == ID]

    features_dropped = ['Unnamed: 0','SK_ID_CURR','TARGET']
    relevant_features = [col for col in data.columns if col not in features_dropped]

    X = X[relevant_features]

    print('X shape', X.shape)
    print('X columns :', X.columns)
    prediction = model.predict(X)  # y_pred
    proba = model.predict_proba(X)  # y_proba
    dict_result = {"prediction": int(prediction), "proba": float(proba[0][0])}
    print('Résultat de Prédiction : \n', dict_result)
    return jsonify(dict_result)
# Lancer l'application
if __name__ == "__main__":
    app.run(debug=True)
    


