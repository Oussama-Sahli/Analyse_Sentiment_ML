# Utiliser une image légère de Python
FROM python:3.9-slim  

# Définir le répertoire de travail
WORKDIR /app  

# Copier tous les fichiers de l'application dans le conteneur
COPY . /app
 

# Installer les dépendances
RUN pip install --no-cache-dir joblib flask pandas numpy nltk scikit-learn  

# Télécharger les ressources de NLTK
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Exposer le port 5000 (celui utilisé par Flask)
EXPOSE 5000  

# Lancer l'application Flask
CMD ["python", "app.py"]  
