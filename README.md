# Immobilienpreisvorhersage mittels Machine Learning: Eine Analyse der Leistungsfähigkeit verschiedener Regressionsmodelle

Dieses Projekt zielt darauf ab, Immobilienpreise in Milwaukee mithilfe von Machine-Learning-Modellen vorherzusagen. Es werden verschiedene Algorithmen wie **Lasso Regression**, **Ridge Regression**, **Gradient Boosting** und **Random Forest** verwendet, um die Vorhersagegenauigkeit zu evaluieren. Der Datensatz umfasst 5831 Beobachtungen mit 20 Variablen, und die Modelle wurden durch **GridSearchCV** optimiert und anhand von Metriken wie **MSE** und **R²** bewertet.

## Inhaltsverzeichnis
1. [Ziele des Projekts](#ziele-des-projekts)
2. [Daten und Methodik](#daten-und-methodik)
3. [Installation](#installation)

## Ziele des Projekts
- **Vorhersage von Immobilienpreisen**: Entwicklung und Vergleich von Machine-Learning-Modellen zur genauen Prognose von Immobilienwerten.
- **Identifikation von Einflussfaktoren**: Analyse der wichtigsten Variablen, die den Immobilienpreis beeinflussen.
- **Bereitstellung einer interaktiven Anwendung**: Entwicklung einer Dash-Anwendung zur Visualisierung der Vorhersagen.

## Daten und Methodik
### Datensatz
- **Quelle**: „Property Sales“ von Milwaukee (2023).
- **Umfang**: 5831 Beobachtungen mit 20 Variablen (numerisch und kategorial).
- **Datenvorbereitung**:
  - Entfernung irrelevanter Variablen (z. B. `Extwall`, `CondoProject`).
  - Imputation fehlender Werte (Durchschnitt für numerische Variablen).
  - One-Hot-Encoding für kategoriale Variablen (z. B. `style`, `proptype`).
  - Logarithmische Transformation der Zielvariable `Sale_price` zur Normalisierung.

### Methodik
- **Explorative Datenanalyse (EDA)**:
  - Untersuchung der Datenverteilungen, Identifikation von Ausreißern und Korrelationen.
  - Erstellung von Histogrammen, Boxplots und Korrelationsmatrizen.
- **Modellierung**:
  - Verwendung von **Lasso Regression**, **Ridge Regression**, **Gradient Boosting** und **Random Forest**.
  - Hyperparameter-Optimierung mittels **GridSearchCV**.
- **Evaluation**:
  - Bewertung der Modelle anhand von **MSE** und **R²**.
  - 5-fache Kreuzvalidierung zur Sicherstellung der Robustheit.
 
## Installation
1. Klone das Repository:
   ```bash
   git clone https://github.com/Chris2610/houseprice.git

2. Erstellen einer Environment mit Conda. Hierzu Datei requirements-py3.11-ads-ml.txt verwenden.
    ```bash
    conda create -n ads-ml python=3.11 -y
    conda activate ads-ml
    pip install -r requirements-py3.11-ads-ml.txt
    pip install -e .

3. Verwenden des Datensatz im Repository in den Variablen **eda_file_path** und **model_file_path**
  
4. Ausführen des Notebooks in einer IDE wie z.B. Visual Studio Code
