@echo off
echo 🏥 Installation du Systeme de Detection de Diabete...
echo.

:: Verifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installe
    echo 📥 Telechargez Python depuis: https://python.org
    pause
    exit /b 1
)

:: Mettre a jour pip
echo 📦 Mise a jour de pip...
python -m pip install --upgrade pip

:: Installer les dependances
echo 📥 Installation des bibliotheques...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ❌ Erreur d'installation. Essayez:
    echo py -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ✅ Installation terminee!
echo 🚀 Pour lancer l'application:
echo    streamlit run app.py
echo.
pause