Projet : Analyse de Sentiment avec Machine Learning et Déploiement AWS/Docker

Objectif
Créer un modèle d’analyse de sentiment (positif, négatif) en utilisant Random Forest, SVM et MLP, et l'optimiser via Grid Search et validation croisée. Le modèle sera déployé sur AWS EC2 avec Docker.

Technologies Utilisées
📌 Data Science & Machine Learning

Python (Pandas, NumPy, Scikit-learn)
TF-IDF pour la vectorisation du texte
Modèles : Random Forest, SVM, Multi-Layer Perceptron (MLP)
Optimisation : Grid Search + Validation croisée

📌 Déploiement

Docker (pour containeriser l’application)
AWS EC2 (pour héberger le modèle)
Flask (pour l’API)

1️⃣ Prétraitement des Données
Chargement des données (CSV)
Nettoyage du texte :
Suppression des stopwords
Tokenization
Lemmatisation
TF-IDF Vectorization pour transformer les textes en données numériques

2️⃣ Entraînement des Modèles
On teste trois modèles :
✔ Random Forest (puissant et robuste)
✔ SVM (bon sur petits datasets)
✔ MLP (Multi-Layer Perceptron) (réseau de neurones simple)

Validation croisée (K-fold) pour éviter le surapprentissage et voir si le modèle généralise bien.

Optimisation avec Grid Search pour trouver les meilleurs hyperparamètres.

3️⃣ Évaluation des Modèles
Métriques : précision, rappel, F1-score
Matrice de confusion
Courbe ROC/AUC

4️⃣ Déploiement sur AWS EC2 avec Docker
🔹 Étapes du Déploiement
✅ Créer une API Flask/FastAPI qui prend un texte en entrée et renvoie la prédiction du modèle
✅ Créer un Dockerfile pour containeriser l’API
✅ Pousser l’image Docker sur AWS EC2
✅ Déploiement  pour une API performante

🚀 Conclusion
Ce projet combine Machine Learning + Déploiement Cloud, parfait pour une application réelle d’analyse de sentiment.
