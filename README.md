# AI-Driven Student Performance Prediction System

A  machine-learning project that predicts a student's final grade (0–100%) from study habits, attendance, previous performance, wellbeing, and selected study-context information.

## Features

- Final grade prediction, grade band, and pass status
- 14 data-aligned student questions
- Personalized study recommendations
- Result chart comparing predicted grade, previous grade, attendance, and study time
- Visible evaluation metrics: R², MAE, and RMSE

## Run

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\train.py
python app.py
```

Open `http://127.0.0.1:5000`.

## Model evaluation

The included model was trained on 2,40,000 records and evaluated on 60,000 held-out records. Run `python src\train.py` after changing the dataset to generate a fresh model and metrics file.

## Project files

- `dataset/student_performance_full.csv` — supplied training data
- `src/train.py` — reproducible regression training script
- `models/final_grade_model.json` — generated model
- `models/metrics.json` — evaluation results
