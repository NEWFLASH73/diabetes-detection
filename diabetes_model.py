# diabetes_model.py
"""
Modèle de détection de diabète - Classification Binaire
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.df = None
        self.accuracy = None
        self.feature_names = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age'
        ]
        self.target_name = 'outcome'
        
    def load_data(self):
        """Charger le dataset Pima Indians Diabetes"""
        print("📊 Chargement des données de diabète...")
        
        # URL du dataset
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        
        # Noms des colonnes
        columns = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome'
        ]
        
        try:
            self.df = pd.read_csv(url, names=columns)
            print(f"✅ Données chargées: {len(self.df)} patients")
            return self.df
        except Exception as e:
            print(f"❌ Erreur de chargement: {e}")
            # Créer des données factices pour le test
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Créer des données factices si le téléchargement échoue"""
        print("📝 Création de données factices pour le test...")
        np.random.seed(42)
        
        n_samples = 768
        self.df = pd.DataFrame({
            'pregnancies': np.random.randint(0, 15, n_samples),
            'glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
            'blood_pressure': np.random.normal(70, 12, n_samples).clip(0, 122),
            'skin_thickness': np.random.normal(20, 10, n_samples).clip(0, 99),
            'insulin': np.random.normal(80, 100, n_samples).clip(0, 846),
            'bmi': np.random.normal(32, 8, n_samples).clip(0, 67),
            'diabetes_pedigree': np.random.exponential(0.5, n_samples).clip(0, 2.5),
            'age': np.random.randint(21, 81, n_samples),
            'outcome': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
        })
        
        print(f"✅ Données factices créées: {len(self.df)} patients")
        return self.df
    
    def explore_data(self):
        """Explorer et analyser les données"""
        print("\n🔍 Exploration des données:")
        
        # Informations de base
        print(f"Shape: {self.df.shape}")
        print(f"Colonnes: {list(self.df.columns)}")
        
        # Statistiques descriptives
        print("\n📊 Statistiques descriptives:")
        print(self.df.describe())
        
        # Vérifier les valeurs manquantes
        print("\n🔎 Valeurs manquantes:")
        print(self.df.isnull().sum())
        
        # Distribution de la variable cible
        print("\n🎯 Distribution de la variable cible (outcome):")
        outcome_counts = self.df['outcome'].value_counts()
        print(outcome_counts)
        print(f"Pourcentage de diabétiques: {outcome_counts[1]/len(self.df)*100:.1f}%")
        
        return self.df
    
    def preprocess_data(self):
        """Prétraiter les données"""
        print("\n🔧 Prétraitement des données...")
        
        # Remplacer les zéros improbables par NaN
        columns_to_clean = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
        self.df[columns_to_clean] = self.df[columns_to_clean].replace(0, np.nan)
        
        # Remplacer les NaN par la médiane
        for column in columns_to_clean:
            self.df[column].fillna(self.df[column].median(), inplace=True)
        
        print("✅ Données prétraitées")
        return self.df
    
    def visualize_data(self):
        """Créer des visualisations des données"""
        print("\n🎨 Création des visualisations...")
        
        os.makedirs('diabetes_plots', exist_ok=True)
        
        # 1. Distribution de la variable cible
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 2, 1)
        self.df['outcome'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('Distribution Diabète vs Non-Diabète')
        plt.xlabel('Diabétique (1=Oui, 0=Non)')
        plt.ylabel('Nombre de patients')
        plt.xticks(rotation=0)
        
        # 2. Distribution du glucose
        plt.subplot(2, 2, 2)
        plt.hist([self.df[self.df['outcome'] == 0]['glucose'], 
                 self.df[self.df['outcome'] == 1]['glucose']],
                 alpha=0.7, label=['Non-Diabétique', 'Diabétique'], bins=20)
        plt.xlabel('Glucose')
        plt.ylabel('Fréquence')
        plt.title('Distribution du Glucose')
        plt.legend()
        
        # 3. Distribution du BMI
        plt.subplot(2, 2, 3)
        plt.hist([self.df[self.df['outcome'] == 0]['bmi'], 
                 self.df[self.df['outcome'] == 1]['bmi']],
                 alpha=0.7, label=['Non-Diabétique', 'Diabétique'], bins=20)
        plt.xlabel('BMI')
        plt.ylabel('Fréquence')
        plt.title('Distribution du BMI')
        plt.legend()
        
        # 4. Matrice de corrélation
        plt.subplot(2, 2, 4)
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True)
        plt.title('Matrice de Corrélation')
        
        plt.tight_layout()
        plt.savefig('diabetes_plots/data_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Graphiques sauvegardés dans 'diabetes_plots/'")
    
    def train_model(self, model_type='random_forest', test_size=0.2, random_state=42):
        """Entraîner le modèle de classification"""
        print(f"\n🤖 Entraînement du modèle ({model_type})...")
        
        # Préparer les données
        X = self.df[self.feature_names]
        y = self.df[self.target_name]
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalisation des features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"📊 Données d'entraînement: {X_train.shape[0]} patients")
        print(f"📊 Données de test: {X_test.shape[0]} patients")
        
        # Sélection du modèle
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=random_state)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=random_state)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        
        # Entraînement
        self.model.fit(X_train_scaled, y_train)
        
        # Prédictions et évaluation
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        self.accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"✅ Modèle entraîné avec succès!")
        print(f"📈 Accuracy: {self.accuracy:.4f}")
        print(f"📊 AUC Score: {auc_score:.4f}")
        
        # Rapport de classification
        print("\n📋 Rapport de classification:")
        print(classification_report(y_test, y_pred, target_names=['Non-Diabétique', 'Diabétique']))
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Diabétique', 'Diabétique'],
                   yticklabels=['Non-Diabétique', 'Diabétique'])
        plt.title('Matrice de Confusion - Détection Diabète')
        plt.xlabel('Prédiction')
        plt.ylabel('Vérité')
        plt.savefig('diabetes_plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Courbe ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux Faux Positifs')
        plt.ylabel('Taux Vrais Positifs')
        plt.title('Courbe ROC - Détection Diabète')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('diabetes_plots/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.accuracy, auc_score
    
    def predict_diabetes(self, features_dict):
        """Prédire le risque de diabète pour un nouveau patient"""
        if self.model is None:
            raise ValueError("Le modèle n'est pas encore entraîné!")
        
        # Convertir le dictionnaire en array
        features = np.array([[features_dict[feature] for feature in self.feature_names]])
        
        # Normaliser les features
        features_scaled = self.scaler.transform(features)
        
        # Prédiction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        result = {
            'diabetes_risk': bool(prediction),
            'confidence': probability[prediction],
            'probability_no_diabetes': probability[0],
            'probability_diabetes': probability[1],
            'risk_level': 'Élevé' if probability[1] > 0.7 else 'Modéré' if probability[1] > 0.3 else 'Faible'
        }
        
        return result
    
    def get_feature_importance(self):
        """Obtenir l'importance des features"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Visualisation
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance_df, x='importance', y='feature')
            plt.title('Importance des Caractéristiques')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('diabetes_plots/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_importance_df
        return None
    
    def save_model(self, filename='diabetes_model.joblib'):
        """Sauvegarder le modèle entraîné"""
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'accuracy': self.accuracy
        }
        
        joblib.dump(model_data, filename)
        print(f"💾 Modèle sauvegardé sous: {filename}")
    
    def load_model(self, filename='diabetes_model.joblib'):
        """Charger un modèle sauvegardé"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.accuracy = model_data['accuracy']
        print(f"📂 Modèle chargé depuis: {filename}")

def main():
    """Fonction principale pour tester le modèle"""
    print("🏥 DÉTECTEUR DE DIABÈTE - CLASSIFICATION BINAIRE")
    print("=" * 60)
    
    # Initialiser et entraîner le modèle
    predictor = DiabetesPredictor()
    predictor.load_data()
    predictor.explore_data()
    predictor.preprocess_data()
    predictor.visualize_data()
    predictor.train_model()
    
    # Importance des features
    feature_importance = predictor.get_feature_importance()
    if feature_importance is not None:
        print("\n📊 Importance des caractéristiques:")
        print(feature_importance)
    
    # Sauvegarder le modèle
    predictor.save_model()
    
    # Test de prédiction
    print("\n🎯 TEST DE PRÉDICTION")
    test_patient = {
        'pregnancies': 2,
        'glucose': 138,
        'blood_pressure': 62,
        'skin_thickness': 35,
        'insulin': 0,
        'bmi': 33.6,
        'diabetes_pedigree': 0.127,
        'age': 47
    }
    
    result = predictor.predict_diabetes(test_patient)
    
    print(f"Caractéristiques du patient: {test_patient}")
    print(f"Risque de diabète: {'OUI' if result['diabetes_risk'] else 'NON'}")
    print(f"Niveau de risque: {result['risk_level']}")
    print(f"Probabilité diabète: {result['probability_diabetes']:.2%}")
    print(f"Confiance: {result['confidence']:.2%}")

if __name__ == "__main__":
    main()