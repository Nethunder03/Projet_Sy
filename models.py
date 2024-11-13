from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

class ModelTrainer:
    """
    Modélisation et Prédiction

    • Diviser le dataset en ensemble d’entraînement et de test.
    • Régression Linéaire : Utiliser la régression pour prédire une note de satisfaction (si applicable).
    • KNN : Appliquer KNN pour classer la satisfaction (satisfait/non-satisfait).
    • Naïve Bayes : Appliquer Naïve Bayes pour comparer les performances et évaluer la probabilité
      d’appartenance aux classes.
    • Livrable : Rapport de comparaison des performances de chaque modèle (précision, rappel, F1-score).
    """

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
        print("Régression Linéaire - MSE:", mse)

        self.models['Linear Regression'] = {
            'model': model,
            'predictions': predictions,
            'mse': mse
        }

    def train_knn(self, n_neighbors=5):
        """Entraîne le modèle KNN pour la classification binaire (satisfait/non-satisfait)."""
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        print("KNN - Accuracy:", accuracy)
        print("KNN - Classification Report:\n", report)

        self.models['KNN'] = {
            'model': model,
            'predictions': predictions,
            'accuracy': accuracy,
            'report': report
        }

    def train_naive_bayes(self):
        """Entraîne le modèle Naïve Bayes pour la classification des classes de satisfaction."""
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        print("Naïve Bayes - Accuracy:", accuracy)
        print("Naïve Bayes - Classification Report:\n", report)

        self.models['Naive Bayes'] = {
            'model': model,
            'predictions': predictions,
            'accuracy': accuracy,
            'report': report
        }

    def evaluate_models(self):
        """Rapporte les performances de chaque modèle entraîné."""
        if self.is_regression:
            print("Évaluation de la Régression Linéaire :")
            self.train_linear_regression()
        else:
            print("Évaluation des Modèles de Classification :")
            self.train_knn()
            self.train_naive_bayes()
