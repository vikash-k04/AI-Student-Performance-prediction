from pathlib import Path
import json
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent
NUMERIC = ["age", "study_hours", "attendance", "sleep_hours", "previous_grade", "assignments_completed", "practice_tests_taken", "group_study_hours", "motivation_level", "mental_health_score", "screen_time", "social_media_hours","time_management_score",
"notes_quality_score"]
LABELS = {"age":"Age", "study_hours":"Study Hours per Day", "attendance":"Attendance (%)", "sleep_hours":"Sleep Hours per Night", "previous_grade":"Previous Grade (%)", "assignments_completed":"Assignments Completed", "practice_tests_taken":"Practice Tests Taken", "group_study_hours":"Group Study Hours", "motivation_level":"Motivation Level ", "mental_health_score":"Mental Health Score ", "screen_time":"Screen Time (hours/day)", "social_media_hours":"Social Media (hours/day)","time_management_score": "Time Management Score","notes_quality_score": "Notes Quality Score"}
LIMITS = {"age":(15,21), "study_hours":(0,12), "attendance":(40,100), "sleep_hours":(3,10), "previous_grade":(20,100), "assignments_completed":(0,10), "practice_tests_taken":(0,10), "group_study_hours":(0,6), "motivation_level":(0,10), "mental_health_score":(0,10), "screen_time":(0,12), "social_media_hours":(0,8),"time_management_score": (0, 10),"notes_quality_score": (0, 10)}
app = Flask(__name__); model = metrics = None

def assets():
    global model, metrics
    if model is None:
        model = json.loads((BASE / "models" / "final_grade_model.json").read_text(encoding="utf-8"))
        metrics = json.loads((BASE / "models" / "metrics.json").read_text(encoding="utf-8"))
    return model, metrics

def grade(score): return "A" if score >= 80 else "B" if score >= 70 else "C" if score >= 60 else "D" if score >= 50 else "F"
def tips(v):
    result=[]
    if v["attendance"] < 75: result.append("Improve attendance to at least 75%.")
    if v["study_hours"] < 3: result.append("Set aside a consistent daily study session.")
    if v["practice_tests_taken"] < 3: result.append("Use more practice tests to improve exam readiness.")
    if v["sleep_hours"] < 7: result.append("Aim for 7–8 hours of sleep for better concentration.")
    return result or ["Keep your balanced study routine and revise weak topics regularly."]

@app.get("/")
def home():
    current_model, current_metrics = assets()
    return render_template("index.html", labels=LABELS, limits=LIMITS, choices=current_model["category_values"], metrics=current_metrics)

@app.post("/predict")
def predict():
    current_model, current_metrics = assets()
    try:
        values={}
        for name in NUMERIC:
            value=float(request.form.get(name, ""))
            minimum, maximum = LIMITS[name]
            if not minimum <= value <= maximum: raise ValueError(f"{LABELS[name]} must be between {minimum} and {maximum}.")
            values[name]=value
        for field, allowed in current_model["category_values"].items():
            value=request.form.get(field, "")
            if value not in allowed: raise ValueError(f"Select a valid {field.replace('_', ' ')}.")
            values[field]=value
        if values["group_study_hours"] > values["study_hours"]: raise ValueError("Group study hours cannot exceed total study hours.")
        if values["social_media_hours"] > values["screen_time"]: raise ValueError("Social-media time cannot exceed total screen time.")
        if values["study_hours"] + values["sleep_hours"] + values["screen_time"] > 24: raise ValueError("Study, sleep, and screen time together cannot exceed 24 hours per day.")
        vector=[1.0]+[values[name] for name in NUMERIC]
        vector += [1.0 if values[name.split('=',1)[0]] == name.split('=',1)[1] else 0.0 for name in current_model["feature_names"][len(NUMERIC):]]
        score=round(min(100,max(0,sum(a*b for a,b in zip(vector,current_model["coefficients"])))),1)
        return render_template("result.html", score=score, grade=grade(score), status="Pass" if score >= 50 else "Needs improvement", tips=tips(values), values=values, labels=LABELS, metrics=current_metrics)
    except (ValueError, TypeError) as error:
        return render_template("index.html", labels=LABELS, limits=LIMITS, choices=current_model["category_values"], metrics=current_metrics, error=str(error) or "Please complete every field with a valid value."), 400

if __name__ == "__main__": app.run(debug=True)
