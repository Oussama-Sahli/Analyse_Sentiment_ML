# -*- coding: utf-8 -*-



import pandas as pd

# Spécifier les noms des colonnes
columns = ['text','cible']

# Charger le dataset avec les noms de colonnes
df_pos = pd.read_csv("data/data_positif.csv", encoding='utf-8')
df_neg = pd.read_csv("data/data_negatif.csv", encoding='utf-8')



df = pd.concat([df_pos, df_neg], ignore_index=True)

# Afficher les premières lignes pour vérifier
print(df.head())




    

# ------------------ Visualiser la répartition des sentiments dans les données -- #

#---------  Répartition des Sentiments --- #

import matplotlib.pyplot as plt
import seaborn as sns


df['cible'] = df['cible'].replace("positif", 1)
df['cible'] = df['cible'].replace("negatif", 0)



# Compter le nombre d'occurrences pour chaque classe dans 'Cible'
class_counts = df['cible'].value_counts()

# Visualiser la proportion des sentiments
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')

# Ajuster les labels pour chaque classe
labels = {0: 'Négatif', 1: 'Positif'}  

# Appliquer les labels sur l'axe des X
plt.title('Répartition des sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Nombre de tweets')
plt.xticks(ticks=[0, 1], labels=[labels[0], labels[1]])

plt.show()


#--------------  Analyse de la Longueur des Tweet ----------- #

# Ajouter une colonne pour la longueur des tweets
df['longueur_tweet'] = df['text'].apply(len)

# Visualiser la longueur des tweets
plt.figure(figsize=(10, 6))
sns.histplot(df['longueur_tweet'], kde=True, bins=30, color='blue')
plt.title('Répartition de la longueur des tweets')
plt.xlabel('Longueur des tweets (en caractères)')
plt.ylabel('Nombre de tweets')
plt.show()




#--------------  Analyse de la Fréquence des Mots (Nuage de Mots) ----------- #

from wordcloud import WordCloud

# Fusionner tous les tweets en un seul texte
all_text = " ".join(df['text'])

# Générer le nuage de mots
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Afficher le nuage de mots
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



#--------------  Analyse de la longueur des tweets par sentiment ----------- #


plt.figure(figsize=(10, 6))
sns.histplot(df[df['cible'] == 1]['longueur_tweet'], kde=True,
             bins=30, color='green', label='Positif')
sns.histplot(df[df['cible'] == 0]['longueur_tweet'], kde=True, bins=30,
             color='red', label='Négatif')

plt.title('Distribution de la longueur des tweets par sentiment')
plt.xlabel('Longueur des tweets')
plt.ylabel('Nombre de tweets')
plt.legend()
plt.show()


#-------------- Analyse des mots les plus fréquents (Stopwords exclus)----------- #


from collections import Counter
import nltk
from nltk.corpus import stopwords
import string

# Télécharger les stopwords français
nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

# Fonction pour nettoyer et tokeniser les tweets
def nettoyer_texte(texte):
    tokens = texte.lower().split()
    tokens = [mot.strip(string.punctuation) for mot in tokens if mot not in stop_words]
    return tokens

# Appliquer le nettoyage et compter la fréquence des mots
df['tokens'] = df['text'].apply(nettoyer_texte)
all_words = [word for tokens in df['tokens'] for word in tokens]
word_freq = Counter(all_words)

# Sélectionner les 20 mots les plus fréquents
common_words = word_freq.most_common(20)
words, counts = zip(*common_words)

# Afficher sous forme de barplot
plt.figure(figsize=(12, 6))
sns.barplot(x=list(words), y=list(counts), palette='coolwarm')
plt.xticks(rotation=45)
plt.title('Mots les plus fréquents (stopwords exclus)')
plt.xlabel('Mots')
plt.ylabel('Fréquence')
plt.show()

#-------------- Analyse des bigrammes et trigrammes les plus fréquents----------- #

from itertools import islice
from nltk.util import ngrams

# Générer des bigrammes et trigrammes
def get_ngrams(text_series, n=2):
    all_ngrams = []
    for text in text_series:
        tokens = nettoyer_texte(text)
        all_ngrams.extend(ngrams(tokens, n))
    return Counter(all_ngrams)

# Afficher les bigrammes et trigrammes les plus fréquents
bigram_freq = get_ngrams(df['text'], n=2).most_common(15)
trigram_freq = get_ngrams(df['text'], n=3).most_common(15)

# Fonction d'affichage
def plot_ngrams(ngrams_freq, title):
    ngrams, counts = zip(*ngrams_freq)
    ngrams = [' '.join(ngram) for ngram in ngrams]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(ngrams), y=list(counts), palette='viridis')
    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel('N-grammes')
    plt.ylabel('Fréquence')
    plt.show()

plot_ngrams(bigram_freq, 'Bigrammes les plus fréquents')
plot_ngrams(trigram_freq, 'Trigrammes les plus fréquents')


#--------------Comparaison des mots les plus fréquents par sentiment ----------- #

pos_text = df[df['cible'] == 1]['text'].str.cat(sep=" ")
neg_text = df[df['cible'] == 0]['text'].str.cat(sep=" ")

wordcloud_pos = WordCloud(width=800, height=400, background_color='white', 
                          colormap='Greens').generate(pos_text)
wordcloud_neg = WordCloud(width=800, height=400, background_color='white', 
                          colormap='Reds').generate(neg_text)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].imshow(wordcloud_pos, interpolation='bilinear')
ax[0].axis('off')
ax[0].set_title('Nuage de mots - Positif')

ax[1].imshow(wordcloud_neg, interpolation='bilinear')
ax[1].axis('off')
ax[1].set_title('Nuage de mots - Négatif')

plt.show()




#-------------- prétraitement des données ----------- #
df=df[['text','cible']]
# Supprimer les lignes avec des valeurs manquantes
df.dropna(inplace=True)

# Supprimer les doublons
df.drop_duplicates(inplace=True)
#-------------- Normalisation du texte ----------- #
import re

def nettoyer_texte(texte):
    texte = texte.lower()  # Convertir en minuscules
    texte = re.sub(r"http\S+|www\S+|https\S+", "", texte, flags=re.MULTILINE)  # Supprimer les URL
    texte = re.sub(r"\@\w+|\#", "", texte)  # Supprimer les mentions et hashtags
    texte = re.sub(r"[^\w\s]", "", texte)  # Supprimer la ponctuation
    texte = re.sub(r"\d+", "", texte)  # Supprimer les chiffres
    return texte.strip()

df["text_clean"] = df["text"].apply(nettoyer_texte)

#-------------- Suppression des stopwords ----------- #
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

df["text_clean"] = df["text_clean"].apply(lambda x: " ".join([mot for mot in x.split() 
                                                              if mot not in stop_words]))
#-------------- Lemmatisation ou Stemmatisation ----------- #
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

df["text_clean"] = df["text_clean"].apply(lambda x: " ".join([lemmatizer.lemmatize(mot) 
                                                              for mot in x.split()]))

#-------------- Vectorisation des textes avec TF-IDF ----------- #
# alternatives : Bag of Words (CountVectorizer)  /  Word Embeddings (Word2Vec, FastText, BERT) 
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Télécharger les stopwords français
nltk.download('stopwords')
french_stopwords = stopwords.words('french')

# Initialiser le vectoriseur TF-IDF avec les stopwords français
vectorizer = TfidfVectorizer(max_features=5000, stop_words=french_stopwords)
# Transformer les textes en vecteurs TF-IDF
X_tfidf = vectorizer.fit_transform(df["text_clean"])
# Convertir en DataFrame pour visualisation
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
# Afficher les premières lignes
print(tfidf_df.head())



#--------------------- Validation croisée avec SVM ---------------------------- #


y=df['cible']


from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np

# Initialiser le modèle SVM
svm_model = SVC(kernel='linear')

# Appliquer la validation croisée
cv_scores = cross_val_score(svm_model, X_tfidf, y, cv=5)  # 5-fold cross-validation

# Afficher les résultats de la validation croisée
print(f"Scores de validation croisée (SVM) : {cv_scores}")
print(f"Précision moyenne : {np.mean(cv_scores):.2f}")





#--------------------- Validation croisée avec Random Forest---------------------------- #

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialiser le modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Appliquer la validation croisée
cv_scores_rf = cross_val_score(rf_model, X_tfidf, y, cv=5)

# Afficher les résultats de la validation croisée
print(f"Scores de validation croisée (Random Forest) : {cv_scores_rf}")
print(f"Précision moyenne : {np.mean(cv_scores_rf):.2f}")



#--------------------- Validation croisée avec MLP (perceptron multicouche)  ---------------------------- #
from sklearn.neural_network import MLPClassifier

# Initialiser le modèle MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=42)

# Appliquer la validation croisée
cv_scores_mlp = cross_val_score(mlp_model, X_tfidf, y, cv=5)

# Afficher les résultats de la validation croisée
print(f"Scores de validation croisée (MLP) : {cv_scores_mlp}")
print(f"Précision moyenne : {np.mean(cv_scores_mlp):.2f}")




#--------------  Entraînement du modèle SVM pour l'analyse de sentiments----------- #
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1️⃣ Diviser les données (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['cible'], test_size=0.2, random_state=42)

# 2️⃣ Initialiser et entraîner le modèle SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# 3️⃣ Prédictions sur les données de test
y_pred = svm_model.predict(X_test)

# 4️⃣ Évaluation du modèle
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
print("\nRapport de classification :\n", classification_report(y_test, y_pred))
print("\nExactitude : {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

#--------------------- Random Forest ---------------------------- #
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# Créer le modèle Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Entraîner le modèle
rf_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred_rf = rf_model.predict(X_test)

# Afficher le rapport de classification
print("Rapport de classification (Random Forest) :")
print(classification_report(y_test, y_pred_rf))

# Afficher l'exactitude
print("\nExactitude (Random Forest) : {:.2f}%".format(accuracy_score(y_test, y_pred_rf) * 100))


#--------------------- MLP ---------------------------- #
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Créer le modèle MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# Entraîner le modèle
mlp_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred_mlp = mlp_model.predict(X_test)

# Afficher le rapport de classification
print("Rapport de classification (MLP) :")
print(classification_report(y_test, y_pred_mlp))

# Afficher l'exactitude
print("\nExactitude (MLP) : {:.2f}%".format(accuracy_score(y_test, y_pred_mlp) * 100))





import joblib

joblib.dump(mlp_model, "app/analyse_sentiment_model.pkl")
joblib.dump(vectorizer, "app/tfidf_vectorizer.pkl")




#--------------------- optimiser le MLP avec GridSearchCV  ---------------------------- #
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Définir les hyperparamètres à tester
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (200,)],  # Nombre de neurones dans chaque couche cachée
    'activation': ['tanh', 'relu'],  # Fonction d'activation
    'solver': ['adam', 'sgd'],  # Optimiseur
    'alpha': [0.0001, 0.001, 0.01],  # Pénalité de régularisation
    'learning_rate': ['constant', 'adaptive'],  # Stratégie d'ajustement du taux d'apprentissage
}

# Initialiser le modèle MLP
mlp_model = MLPClassifier(max_iter=1000, random_state=42)
# GridSearchCV avec validation croisée à 5 plis
grid_search_mlp = GridSearchCV(mlp_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# Entraîner GridSearchCV
grid_search_mlp.fit(X_tfidf, y)
# Afficher les meilleurs paramètres trouvés
print("Meilleurs paramètres : ", grid_search_mlp.best_params_)
# Afficher le meilleur score de validation croisée
print("Meilleur score de validation croisée : {:.2f}".format(grid_search_mlp.best_score_))
# Entraîner le modèle final avec les meilleurs paramètres
best_mlp_model = grid_search_mlp.best_estimator_
# Effectuer des prédictions avec le modèle optimisé
y_pred_mlp_optimized = best_mlp_model.predict(X_test)
# Évaluer les performances sur les données de test
print("\nRapport de classification (MLP optimisé) :")
print(classification_report(y_test, y_pred_mlp_optimized))
# Afficher l'exactitude finale
print("\nExactitude (MLP optimisé) : {:.2f}%".format(accuracy_score(y_test, y_pred_mlp_optimized) * 100))








