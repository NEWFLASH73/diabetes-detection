 @echo off
echo 🏥 Déploiement du Systeme de Detection de Diabète sur GitHub...
echo.

:: Vérifier Git
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git n'est pas installé!
    echo 📥 Téléchargez Git depuis: https://git-scm.com
    pause
    exit /b 1
)

:: Initialiser Git
echo 🔧 Initialisation Git...
git init

:: Configurer Git
echo 📝 Configuration Git...
git config user.email "newflash73@example.com"
git config user.name "NEWFLASH73"

:: Vérifier et corriger le remote
echo 🔗 Configuration du remote...
git remote remove origin 2>nul
git remote add origin https://github.com/NEWFLASH73/diabetes-detection.git

:: Ajouter les fichiers
echo 📁 Ajout des fichiers...
git add .

:: Commit
echo 💾 Création du commit...
git commit -m "feat: Initial commit - Diabetes Detection System with Streamlit

- Machine Learning binary classification
- Medical diagnostic interface
- Real-time risk assessment
- Multiple algorithm support
- Comprehensive data analysis
- Professional medical design"

:: Pousser sur GitHub
echo 🚀 Poussée vers GitHub...
git branch -M main
git push -u origin main

if errorlevel 1 (
    echo.
    echo ❌ Erreur lors du déploiement.
    echo.
    echo 🔧 Solutions possibles:
    echo 1. Vérifiez que le repository existe sur GitHub
    echo 2. Vérifiez vos identifiants GitHub
    echo 3. Essayez: git push -u origin main --force
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Déploiement réussi!
echo 🌐 Votre projet est disponible sur:
echo    https://github.com/NEWFLASH73/diabetes-detection
echo.
echo 🚀 Prochaines étapes:
echo 1. Ajouter une description sur GitHub
echo 2. Configurer GitHub Pages si besoin
echo 3. Partager le lien
echo.
pause