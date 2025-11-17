# app.py
"""
Application Web de Détection de Diabète - Classification Binaire
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import joblib

# Ajouter le chemin actuel pour importer diabetes_model
sys.path.append(os.path.dirname(__file__))

# Configuration de la page
st.set_page_config(
    page_title="Détection Diabète",
    page_icon="🏥",
    layout="wide"
)

# Titre principal
st.title("🏥 Détection de Diabète - Outil de Diagnostic Assisté")
st.markdown("""
Cette application utilise l'intelligence artificielle pour évaluer le risque de diabète 
basé sur les caractéristiques médicales d'un patient. **Classification binaire avec Machine Learning.**
""")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à:", [
    "🔍 Diagnostic Patient", 
    "📊 Analyse des Données",
    "🤖 Performance du Modèle",
    "ℹ️ À Propos"
])

# Fonction pour charger le prédicteur
def load_predictor():
    try:
        from diabetes_model import DiabetesPredictor
        predictor = DiabetesPredictor()
        predictor.load_data()
        predictor.preprocess_data()
        
        # Essayer de charger un modèle existant
        if os.path.exists('diabetes_model.joblib'):
            try:
                predictor.load_model('diabetes_model.joblib')
                st.sidebar.success("✅ Modèle médical chargé!")
            except:
                st.sidebar.warning("⚠️ Entraînement d'un nouveau modèle...")
                predictor.train_model()
                predictor.save_model()
        else:
            with st.spinner("Entraînement du modèle médical en cours..."):
                predictor.train_model()
                predictor.save_model()
            st.sidebar.success("✅ Nouveau modèle entraîné!")
        
        return predictor
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None

# Charger le prédicteur
predictor = load_predictor()

if page == "🔍 Diagnostic Patient":
    st.header("🔍 Diagnostic du Risque de Diabète")
    
    if predictor is None:
        st.error("❌ Le système de diagnostic n'est pas disponible.")
    else:
        # Deux colonnes pour les inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Informations Démographiques")
            age = st.slider("Âge du patient", 20, 80, 45)
            pregnancies = st.slider("Nombre de grossesses", 0, 15, 2)
            
        with col2:
            st.subheader("📊 Mesures Médicales")
            glucose = st.slider("Glucose (mg/dL)", 50, 200, 120)
            blood_pressure = st.slider("Pression artérielle (mmHg)", 40, 120, 70)
            bmi = st.slider("IMC (Body Mass Index)", 15.0, 50.0, 25.0, 0.1)
        
        # Autres mesures dans une nouvelle ligne
        col3, col4 = st.columns(2)
        
        with col3:
            skin_thickness = st.slider("Épaisseur peau triceps (mm)", 5, 60, 20)
            insulin = st.slider("Insuline (mu U/ml)", 0, 300, 80)
        
        with col4:
            diabetes_pedigree = st.slider("Fonction pedigree diabète", 0.0, 2.5, 0.5, 0.01)
        
        # Résumé des caractéristiques
        st.subheader("📋 Profil Médical du Patient")
        patient_data = {
            "Âge": f"{age} ans",
            "Grossesses": pregnancies,
            "Glucose": f"{glucose} mg/dL",
            "Pression artérielle": f"{blood_pressure} mmHg",
            "IMC": f"{bmi:.1f}",
            "Épaisseur peau": f"{skin_thickness} mm",
            "Insuline": f"{insulin} mu U/ml",
            "Pedigree diabète": f"{diabetes_pedigree:.2f}"
        }
        
        # Afficher les données patient
        cols = st.columns(4)
        for idx, (key, value) in enumerate(patient_data.items()):
            cols[idx % 4].metric(key, value)
        
        # Bouton de diagnostic
        if st.button("🎯 Analyser le Risque de Diabète", type="primary"):
            with st.spinner("Analyse médicale en cours..."):
                try:
                    # Préparer les données pour la prédiction
                    features = {
                        'pregnancies': pregnancies,
                        'glucose': glucose,
                        'blood_pressure': blood_pressure,
                        'skin_thickness': skin_thickness,
                        'insulin': insulin,
                        'bmi': bmi,
                        'diabetes_pedigree': diabetes_pedigree,
                        'age': age
                    }
                    
                    # Faire la prédiction
                    result = predictor.predict_diabetes(features)
                    
                    # Afficher les résultats
                    st.markdown("---")
                    
                    if result['diabetes_risk']:
                        st.error(f"## ⚠️ RISQUE ÉLEVÉ DE DIABÈTE DÉTECTÉ")
                    else:
                        st.success(f"## ✅ RISQUE FAIBLE DE DIABÈTE")
                    
                    # Jauge de risque
                    risk_percentage = result['probability_diabetes'] * 100
                    st.metric("Probabilité de diabète", f"{risk_percentage:.1f}%")
                    st.progress(int(risk_percentage))
                    
                    # Niveau de risque
                    risk_color = "red" if result['risk_level'] == 'Élevé' else "orange" if result['risk_level'] == 'Modéré' else "green"
                    st.markdown(f"**Niveau de risque:** :{risk_color}[{result['risk_level']}]")
                    
                    # Graphique des probabilités
                    st.subheader("📊 Analyse Probabiliste")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Pie chart
                    labels = ['Non-Diabétique', 'Diabétique']
                    sizes = [result['probability_no_diabetes'] * 100, result['probability_diabetes'] * 100]
                    colors = ['#4ECDC4', '#FF6B6B']
                    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Distribution des Risques')
                    
                    # Bar chart
                    ax2.bar(labels, [sizes[0], sizes[1]], color=colors, alpha=0.8)
                    ax2.set_ylabel('Probabilité (%)')
                    ax2.set_title('Probabilités de Diagnostic')
                    ax2.set_ylim(0, 100)
                    
                    # Ajouter les valeurs sur les barres
                    for i, v in enumerate([sizes[0], sizes[1]]):
                        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
                    
                    # Recommandations
                    with st.expander("💡 Recommandations Médicales"):
                        if result['risk_level'] == 'Élevé':
                            st.warning("""
                            **Recommandations pour risque élevé:**
                            - Consultation médicale urgente
                            - Test de glycémie approfondi
                            - Régime alimentaire strict
                            - Activité physique régulière
                            - Surveillance continue
                            """)
                        elif result['risk_level'] == 'Modéré':
                            st.info("""
                            **Recommandations pour risque modéré:**
                            - Consultation médicale recommandée
                            - Surveillance de la glycémie
                            - Adaptation du régime alimentaire
                            - Exercice physique régulier
                            - Contrôles périodiques
                            """)
                        else:
                            st.success("""
                            **Recommandations pour risque faible:**
                            - Maintenir un mode de vie sain
                            - Contrôles annuels de routine
                            - Alimentation équilibrée
                            - Activité physique modérée
                            """)
                
                except Exception as e:
                    st.error(f"❌ Erreur lors du diagnostic: {e}")

elif page == "📊 Analyse des Données":
    st.header("📊 Analyse du Dataset Diabète")
    
    if predictor is None:
        st.error("❌ Le système d'analyse n'est pas disponible.")
    else:
        # Charger et explorer les données
        df = predictor.explore_data()
        
        # Statistiques générales
        st.subheader("📈 Statistiques Descriptives")
        st.dataframe(df.describe())
        
        # Distribution de la variable cible
        st.subheader("🎯 Distribution des Diagnostics")
        col1, col2 = st.columns(2)
        
        with col1:
            outcome_counts = df['outcome'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#4ECDC4', '#FF6B6B']
            bars = ax.bar(['Non-Diabétique', 'Diabétique'], outcome_counts.values, color=colors)
            ax.set_title('Répartition des Patients')
            ax.set_ylabel('Nombre de Patients')
            
            # Ajouter les comptes sur les barres
            for bar, count in zip(bars, outcome_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
        
        with col2:
            st.metric("Patients diabétiques", f"{outcome_counts[1]}", 
                     f"{outcome_counts[1]/len(df)*100:.1f}%")
            st.metric("Patients non-diabétiques", f"{outcome_counts[0]}",
                     f"{outcome_counts[0]/len(df)*100:.1f}%")
        
        # Visualisations des features importantes
        st.subheader("📊 Analyse des Caractéristiques")
        
        # Glucose vs Outcome - CORRIGÉ
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='outcome', y='glucose', ax=ax, 
                   hue='outcome', palette=['#4ECDC4', '#FF6B6B'], legend=False)
        ax.set_title('Distribution du Glucose par Diagnostic')
        ax.set_xlabel('Diabétique (1=Oui, 0=Non)')
        ax.set_ylabel('Glucose (mg/dL)')
        st.pyplot(fig)
        
        # BMI vs Outcome - CORRIGÉ
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='outcome', y='bmi', ax=ax,
                   hue='outcome', palette=['#4ECDC4', '#FF6B6B'], legend=False)
        ax.set_title('Distribution de l\'IMC par Diagnostic')
        ax.set_xlabel('Diabétique (1=Oui, 0=Non)')
        ax.set_ylabel('IMC')
        st.pyplot(fig)

elif page == "🤖 Performance du Modèle":
    st.header("🤖 Performance du Modèle Médical")
    
    if predictor is None:
        st.error("❌ Les données de performance ne sont pas disponibles.")
    else:
        st.markdown("""
        ### Évaluation des Performances du Modèle
        Le modèle utilise un **Random Forest Classifier** pour la détection de diabète.
        """)
        
        # Métriques de performance
        if predictor.accuracy is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{predictor.accuracy:.1%}")
            
            with col2:
                st.metric("Patients analysés", f"{len(predictor.df)}")
            
            with col3:
                st.metric("Caractéristiques", "8")
        
        # Importance des features
        st.subheader("📊 Importance des Caractéristiques")
        
        feature_importance = predictor.get_feature_importance()
        if feature_importance is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax, palette='viridis')
            ax.set_title('Importance des Caractéristiques Médicales')
            ax.set_xlabel('Importance Relative')
            st.pyplot(fig)
            
            # Tableau d'importance
            st.dataframe(feature_importance)
        
        # Réentraînement du modèle
        st.subheader("🔄 Réentraînement du Modèle")
        
        with st.expander("Options avancées d'entraînement"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Type de modèle",
                    ['random_forest', 'logistic_regression', 'svm'],
                    format_func=lambda x: {
                        'random_forest': 'Forêt Aléatoire',
                        'logistic_regression': 'Régression Logistique', 
                        'svm': 'SVM'
                    }[x]
                )
                
                test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20) / 100
            
            with col2:
                random_state = st.number_input("Seed aléatoire", 0, 100, 42)
            
            if st.button("🔄 Réentraîner le Modèle", type="secondary"):
                with st.spinner("Entraînement en cours..."):
                    try:
                        from diabetes_model import DiabetesPredictor
                        new_predictor = DiabetesPredictor()
                        new_predictor.load_data()
                        new_predictor.preprocess_data()
                        accuracy, auc = new_predictor.train_model(
                            model_type=model_type,
                            test_size=test_size,
                            random_state=random_state
                        )
                        
                        new_predictor.save_model()
                        predictor = new_predictor
                        
                        st.success(f"✅ Modèle réentraîné! Accuracy: {accuracy:.2%}, AUC: {auc:.2%}")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'entraînement: {e}")

elif page == "ℹ️ À Propos":
    st.header("ℹ️ À Propos de cette Application")
    
    st.markdown("""
    ## 🏥 Détection de Diabète - Système de Diagnostic Assisté
    
    **Description:**
    Cette application utilise l'apprentissage automatique pour évaluer le risque de diabète 
    chez les patients basé sur huit caractéristiques médicales.
    
    **Caractéristiques analysées:**
    1. **Grossesses**: Nombre de fois enceinte
    2. **Glucose**: Concentration en glucose plasmatique
    3. **Pression artérielle**: Pression artérielle diastolique (mm Hg)
    4. **Épaisseur peau triceps**: Épaisseur du pli cutané du triceps (mm)
    5. **Insuline**: Insuline sérique (mu U/ml)
    6. **IMC**: Indice de masse corporelle (poids en kg/(taille en m)²)
    7. **Fonction pedigree diabète**: Score de risque génétique
    8. **Âge**: Âge du patient (années)
    
    **Dataset:**
    - Pima Indians Diabetes Dataset
    - 768 patients
    - 268 cas de diabète (34.9%)
    - 500 cas non-diabétiques (65.1%)
    
    **Algorithme utilisé:**
    - Random Forest Classifier (par défaut)
    - Régression Logistique (optionnel)
    - SVM (optionnel)
    
    **⚠️ Avertissement Médical:**
    Cet outil est destiné à des fins éducatives et de démonstration uniquement.
    Il ne remplace pas un diagnostic médical professionnel.
    
    **Développé avec:**
    - Python 🐍 & Scikit-learn 🤖
    - Streamlit 🌐
    - Matplotlib 📊 & Seaborn
    """)
    
    st.warning("""
    **Avertissement Important:**
    Ce système est un prototype de démonstration. Les résultats ne constituent pas 
    un diagnostic médical. Consultez toujours un professionnel de santé qualifié 
    pour des problèmes de santé réels.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "🏥 Application médicale éducative | "
    "Classification binaire - Machine Learning | "
    "**À des fins de démonstration uniquement**"
)