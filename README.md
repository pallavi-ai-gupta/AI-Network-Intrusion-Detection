# AI-Powered Network Intrusion Detection System

## Overview
A BTech final-year project that uses machine learning to classify
network traffic records as Normal or Attack.

## Technologies
- Python
- Flask
- Pandas
- Scikit-learn
- HTML, CSS and JavaScript

## Features
- User registration and login
- Random Forest and Decision Tree training
- Model performance comparison
- Traffic prediction from CSV files
- Confusion matrix and performance reports
- Downloadable prediction results

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the libraries:

```bash
python -m pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the application:

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Dataset format
The training CSV must contain feature columns and a target column
named `label`. The example dataset contains synthetic records.

## Limitations
This prototype analyzes CSV records. It does not capture live network
packets, block attacks, or prove that a predicted attack is real.
Evaluate the model on appropriate real-world datasets before making
security claims.