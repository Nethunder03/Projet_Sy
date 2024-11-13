import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, is_regression=False):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.is_regression = is_regression
        self.models = {}

    def train_linear_regression(self):
        """Entraîne une régression linéaire pour la prédiction de notes de satisfaction (si applicable)."""
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)

        # Affichage des résultats
        st.markdown("### Résultats de la Régression Linéaire")
        st.write(f"Mean Squared Error (MSE): **{mse:.4f}**")

        self.models['Linear Regression'] = {
            'model': model,
            'predictions': predictions,
            'mse': mse
        }

    def train_knn(self):
        """Recherche des meilleurs hyperparamètres pour le modèle KNN et entraînement."""
        param_grid = {'n_neighbors': [3, 5, 7, 9]}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='accuracy', cv=5)
        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_
        predictions = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions, output_dict=True)

        # Affichage des meilleurs hyperparamètres et résultats
        st.markdown("### Résultats de KNN avec Recherche d'Hyperparamètres")
        st.write(f"Meilleur nombre de voisins (n_neighbors): **{grid_search.best_params_['n_neighbors']}**")
        st.write(f"Précision: **{accuracy:.2%}**")
        st.write("Rapport de Classification:")
        st.table(report)

        self.models['KNN'] = {
            'model': best_model,
            'predictions': predictions,
            'accuracy': accuracy,
            'report': report
        }

    def train_naive_bayes(self):
        """Entraîne le modèle Naïve Bayes pour la classification des classes de satisfaction."""
        # Naïve Bayes n'a pas beaucoup d'hyperparamètres à ajuster, donc on peut directement l'entraîner
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions, output_dict=True)

        # Affichage des résultats pour Naïve Bayes
        st.markdown("### Résultats de Naïve Bayes")
        st.write(f"Précision: **{accuracy:.2%}**")
        st.write("Rapport de Classification:")
        st.table(report)

        self.models['Naive Bayes'] = {
            'model': model,
            'predictions': predictions,
            'accuracy': accuracy,
            'report': report
        }

    def evaluate_models(self):
        """Rapporte les performances de chaque modèle entraîné."""
        if self.is_regression:
            st.markdown("# Évaluation de la Régression Linéaire")
            self.train_linear_regression()
        else:
            st.markdown("# Évaluation des Modèles de Classification")
            self.train_knn()
            self.train_naive_bayes()


    def predict(self, model_name, new_data):
        model = self.models.get(model_name, {}).get('model')
        if model:
            prediction = model.predict(new_data)
            return prediction
        else:
            return None