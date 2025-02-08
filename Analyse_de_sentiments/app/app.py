# -*- coding: utf-8 -*-

import pickle
import numpy as np
import sys
from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import string
import tempfile
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import re
import os
import tempfile

app = Flask(__name__)
model = joblib.load("analyse_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def nettoyer_texte(texte):
    texte = texte.lower()  # Convertir en minuscules
    texte = re.sub(r"http\S+|www\S+|https\S+", "", texte, flags=re.MULTILINE)  # Supprimer les URL
    texte = re.sub(r"\@\w+|\#", "", texte)  # Supprimer les mentions et hashtags
    texte = re.sub(r"[^\w\s]", "", texte)  # Supprimer la ponctuation
    texte = re.sub(r"\d+", "", texte)  # Supprimer les chiffres
    return texte.strip()


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        print("Erreur : Aucun fichier fourni.", file=sys.stderr)
        return jsonify({"error": "Aucun fichier fourni."}), 400
    
    file = request.files['file']
    try:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            print("Erreur : La colonne 'text' est requise dans le fichier CSV.", file=sys.stderr)
            return jsonify({"error": "La colonne 'text' est requise dans le fichier CSV."}), 400
        
        
        # Retirer la colonne cible si elle existe (par exemple "target", "label")
        if "cible" in df.columns:
            target_column = df["cible"]
            df = df.drop(columns=["cible"])
        else:
            target_column = None
            
        #-----------------------------------------------------------------#
        # Pré-traitement
        
        df["text_clean"] = df["text"].apply(nettoyer_texte)
        nltk.download('stopwords')
        stop_words = set(stopwords.words('french'))
        df["text_clean"] = df["text_clean"].apply(lambda x: " ".join([mot for mot in x.split() if mot not in stop_words]))
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()
        df["text_clean"] = df["text_clean"].apply(lambda x: " ".join([lemmatizer.lemmatize(mot) for mot in x.split()]))
        X_tfidf = vectorizer.transform(df["text_clean"])
        # Prédictions
        predictions = model.predict(X_tfidf)
        df["sentiment"] = predictions
        # Ajouter la colonne cible (si elle existait auparavant)
        if target_column is not None:
           df["cible"] = target_column
        
        # Sauvegarde des résultats dans un fichier CSV temporaire
        output_path = os.path.join(tempfile.gettempdir(), "predictions.csv")
        df.to_csv(output_path, index=False)
       # Envoyer le fichier généré comme pièce jointe
        return send_file(output_path, as_attachment=True, download_name="predictions.csv", mimetype='text/csv')
 
    except Exception as e:
        return jsonify({"error": f"Erreur lors du traitement du fichier : {str(e)}"}), 500


from flask import render_template
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/download")
def download():
    # Chemin vers le fichier temporaire généré précédemment
    file_path = os.path.join(tempfile.gettempdir(), "predictions.csv")
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name="predictions.csv", mimetype='text/csv')
    else:
        return jsonify({"error": "Le fichier n'est pas disponible."}), 404

if __name__ == "__main__":
    print("Lancement du serveur Flask...")
    app.run(host="0.0.0.0", port=5000, debug=False)




