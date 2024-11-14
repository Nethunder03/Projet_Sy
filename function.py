import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D  # Nécessaire pour le 3D

def apply_pca(X, y, target, n_components=2, feature_names=None):
    """
    Applique l'ACP pour réduire les dimensions du jeu de données et visualiser les résultats.

    Parameters:
    -----------
    X : np.ndarray
        Les données à transformer.
    y : np.ndarray
        Les données de test à transformer.
    target : pd.Series
        Colonne cible pour colorer les points lors de la visualisation.
    n_components : int, optional
        Nombre de dimensions souhaitées (2 ou 3 pour la visualisation).
    feature_names : list, optional
        Liste des noms de colonnes pour les données.

    Returns:
    --------
    X_train_pca : np.ndarray
        Données projetées dans l'espace de l'ACP.
    pca : PCA
        Modèle d'ACP ajusté, utile pour interpréter l'importance des composantes principales.
    """
    # Initialiser l'ACP
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X)
    X_test_pca = pca.transform(y)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(target)

    # Visualisation en 2D
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_encoded, cmap='coolwarm', alpha=0.7)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title("Projection ACP en 2D")
        plt.colorbar(label="Cible")
        st.pyplot(plt)  # Afficher le graphique dans Streamlit
        plt.clf()  # Clear figure after display to avoid overlap on next plot

    # Visualisation en 3D
    elif n_components == 3:
        
        df_pca = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2', 'PC3'])
        df_pca['satisfaction'] = target 

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Encoder les labels de satisfaction en entiers pour le coloriage
        le = LabelEncoder()
        satisfaction_encoded = le.fit_transform(df_pca['satisfaction'])
        scatter = ax.scatter(df_pca['PC1'], 
                            df_pca['PC2'], 
                            df_pca['PC3'],
                            c=satisfaction_encoded,
                            cmap='coolwarm',
                            s=80, 
                            alpha=0.6)

        # Ajouter des étiquettes aux axes avec la variance expliquée
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)")
        
        ax.set_title('Visualisation 3D des composantes principales')

        # Afficher dans Streamlit
        st.pyplot(fig)


    # Interprétation des composantes principales
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]

    feature_importance = pd.DataFrame(
        np.abs(pca.components_),
        columns=feature_names,
        index=[f'PC{i+1}' for i in range(n_components)]
    ).T
    feature_importance['Most Influential Component'] = feature_importance.idxmax(axis=1)
    st.write("Importance des variables sur chaque composante principale :")
    st.write(feature_importance)

    return X_train_pca, X_test_pca, pca


def pca_new_data(X, n_components=2):
    max_components = min(X.shape[1], n_components)
    pca = PCA(n_components=max_components)
    X_pca = pca.fit_transform(X)
    return X_pca