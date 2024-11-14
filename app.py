import streamlit as st
import pandas as pd
from data import load_data
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import DataPreprocessor
from models import ModelTrainer
from function import apply_pca, pca_new_data
#from models import create_models

st.set_page_config(
    page_title="Rock Ferrand",
    layout="wide"
)
# Charger les données
data = load_data("Invistico_Airline.csv")

# Créer les modèles
#knn, nb, lr = create_models(X_processed, y)

# Interface Streamlit
st.title('Prédiction de la Satisfaction des Clients')

# Formulaire pour entrer de nouvelles données
st.sidebar.header('Entrez les informations du vol')
age = st.sidebar.slider('Âge', 18, 100, 30)
flight_distance = st.sidebar.slider('Distance du vol (en km)', 0, 5000, 1000)
wifi_service = st.sidebar.slider('Service Wi-Fi en vol (1-5)', 1, 5, 4)
arrival_delay = st.sidebar.slider('Délai d\'arrivée (en minutes)', 0, 120, 30)

# Créer un DataFrame pour les nouvelles données
new_data = pd.DataFrame({
    'Age': [age],
    'FlightDistance': [flight_distance],
    'InflightWifiService': [wifi_service],
    'ArrivalDelay': [arrival_delay]
})

st.write("### Apperçu des données")
st.write("Nombre de lignes et colonnes:", data.shape)
st.dataframe(data.head())

st.write("### Information sur les données")
st.write(data.describe())



col1, col2, col3 = st.columns(3)

with col1:
    show_outliers = st.button('Outliers')

with col2:
    show_corr = st.button('Correlations')

with col3:
    show_scatter = st.button('Scatter Plot')

if show_outliers :
    st.write('### Les valeurs aberrantes')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data.select_dtypes(include=['float64', 'int64']))
    plt.xticks(rotation=45)
    st.pyplot(plt)

elif show_corr:
    st.write('### Identificatoins des correlations ')
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title("Matrice de Corrélation")
    st.pyplot(plt)

else:
    missing_values = data.isnull().sum()
    st.write("Tableau des valeurs manquantes par colonne :")
    st.write(missing_values.to_frame().T)





# Créer une instance de DataPreprocessor avec les données et la colonne cible
preprocessor = DataPreprocessor(data, target_column='satisfaction')

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = preprocessor.split_data()

pipeline = preprocessor.create_pipeline(X_train)

# Appliquer le pipeline d'entraînement sur les données d'entraînement
X_train_processed = pipeline.fit_transform(X_train)

X_test_processed = pipeline.transform(X_test)


st.title("Visualisation de la réduction de dimension avec PCA")

col4, col5 = st.columns(2)

with col4:
    show_2d = st.button('Afficher en 2D')
with col5:
    show_3d = st.button('Afficher en 3D')

if show_2d :
    st.write("### Visualisation en 2D")
    X_train_pca, X_test_pca, pca = apply_pca(X_train_processed, X_test_processed, y_train, n_components=2)


    st.write('### Modélisation et Prédiction')
    model_trainer = ModelTrainer(X_train_pca, X_test_pca, y_train, y_test, is_regression=False)
    st.write(model_trainer.evaluate_models())

    # # Prédiction avec KNN
    # st.write("### Prédiction avec KNN")
    # new_data_pca = pca_new_data(new_data, n_components=2)
    # knn_prediction = model_trainer.predict('KNN', new_data_pca)
    # st.write(f"Résultat de la prédiction KNN: {knn_prediction}")

    # # Prédiction avec Naive Bayes
    # st.write("### Prédiction avec Naive Bayes")
    # nb_prediction = model_trainer.predict('Naive Bayes', new_data_pca)
    # st.write(f"Résultat de la prédiction Naive Bayes: {nb_prediction}")
elif show_3d:
    st.write("### Visualisation en 3D")
    X_train_pca, X_test_pca, pca = apply_pca(X_train_processed, X_test_processed, y_train, n_components=3)

    st.write('### Modélisation et Prédiction')
    model_trainer = ModelTrainer(X_train_pca, X_test_pca, y_train, y_test, is_regression=False)
    st.write(model_trainer.evaluate_models())
    new_data_pca = pca_new_data(new_data, n_components=3)

    # # Prédiction avec KNN
    # st.write("### Prédiction avec KNN")
    # knn_prediction = model_trainer.predict('KNN', new_data_pca)
    # st.write(f"Résultat de la prédiction KNN: {knn_prediction}")

    # # Prédiction avec Naive Bayes
    # st.write("### Prédiction avec Naive Bayes")
    # nb_prediction = model_trainer.predict('Naive Bayes', new_data_pca)
    # st.write(f"Résultat de la prédiction Naive Bayes: {nb_prediction}")
