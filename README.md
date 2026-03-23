# student-dropout-prediction
ML pipeline predicting student dropout, enrollment, or graduation using stacking ensemble (78.5% accuracy). Includes SHAP explainability, fairness audit, cross-institutional validation (UPV Spain), and Streamlit advisor dashboard. ITEC 6240, York University 2026.

# Student Dropout Prediction — ML Pipeline

**Predicting Student Dropout and Academic Success Using Machine Learning**  
*A Multi-Strategy Approach with Cross-Institutional Validation and Fairness Analysis*

**ITEC 6240 — Machine Learning and Its Applications | York University, Winter 2026**  

---

## 🎯 Project Overview

Nearly 1 in 3 students who begin a bachelor's degree never finish it. Universities collect vast amounts of student data — but most of it goes unused until it's too late.

This project builds a machine learning pipeline that predicts whether a student will **drop out**, remain **enrolled** (uncertain), or **graduate** — using 36 features collected at enrollment and after two semesters. We go beyond standard model comparison by adding subgroup-stratified SHAP analysis, the first formal fairness audit on this dataset, and cross-institutional validation against 20,427 students from a Spanish university.

**Live demo:** https://student-dropout-prediction-itec6240.streamlit.app/

---

## 📊 Results at a Glance

| Model | Accuracy | Macro F1 | Error Rate |
|---|---|---|---|
| **Stacking Ensemble ★ Best** | **78.5%** | **0.732** | **21.5%** |
| Random Forest | 77.2% | 0.718 | 22.8% |
| XGBoost | 76.7% | 0.706 | 23.3% |
| Logistic Regression | 73.6% | 0.697 | 26.4% |

**Top predictors (SHAP):** 2nd semester courses approved (0.105) · 1st semester approved (0.050) · 2nd semester grade (0.046) · Tuition fees up to date (0.030)

**Key finding:** For students behind on tuition, tuition payment status becomes the **#1 predictor (SHAP = 0.184)**, overtaking all academic features.

**Fairness audit:** All 4 demographic groups (gender, scholarship status, age, tuition) fail the four-fifths disparate impact rule. Most critical: the model misses **53% of scholarship-holder dropouts** (TPR 47.4% vs 74.3% for non-holders).

---

## 🗂️ Repository Structure

```
student-dropout-prediction/
│
├── app.py                          # Streamlit advisor dashboard (v4)
├── data.csv                        # UCI dataset (4,424 students, 36 features)
├── lms_thresholds.json             # LMS behavioral thresholds from UPV analysis
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
└── student_dropout_prediction_pipeline_final.py   # Full ML pipeline (Sections 1–10)
```

---

## 🚀 Running Locally

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn streamlit plotly
```

### Run the Streamlit App
```bash
# Clone the repo
git clone https://github.com/hammadmrza/student-dropout-prediction.git
cd student-dropout-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```
The app trains models on first launch (~30 seconds), then caches them for instant reuse.

### Run the Full ML Pipeline
```bash
python student_dropout_prediction_pipeline_final.py
```
Outputs console results + 12 plots saved to a `plots/` directory.

**Note:** The cross-institutional UPV section (Section 9) requires the UPV 2022 dataset (`dataset_2022_hash.csv`). Download from: https://doi.org/10.3390/data10100162 and place in the same directory.

---

## 📋 Pipeline Sections

| Section | Description |
|---|---|
| 1 | Data loading and exploration |
| 2 | Preprocessing (split → scale → SMOTE) |
| 3 | Individual model training (RF, LR, XGBoost) |
| 4 | Stacking ensemble |
| 5 | Evaluation plots (confusion matrices, model comparison, per-class F1) |
| 6 | SHAP global explainability |
| 7 | Subgroup-stratified SHAP (tuition + scholarship subgroups) |
| 8 | Fairness audit (disparate impact + equalized odds) |
| 9 | Cross-institutional validation (UPV dataset) |
| 10 | Summary |

---

## 🖥️ Streamlit App Features

**Mode 1 — Individual Prediction**
- Enter student data → color-coded risk label (🔴 Dropout / 🟡 Enrolled / 🟢 Graduate)
- SHAP waterfall chart showing top 10 contributing features
- Context-aware intervention suggestions (financial aid referral, academic tutoring, etc.)
- 🍁 Canadian grade conversion guide built-in (% ÷ 5 for course grades, % × 2 for admission)

**Mode 2 — Batch CSV Upload**
- Upload a cohort CSV → risk distribution pie chart → ranked high-risk table → downloadable predictions CSV

**Mode 3 — LMS Behavioral Assessment (Optional)**
- Enter monthly LMS login days, assignment submissions, campus Wi-Fi days, platform minutes
- Compared against thresholds from 20,427 UPV students → traffic-light risk indicator
- Independent of the main prediction model — supplementary behavioral check

---

## 📐 Methodology

### Data
- **Primary:** UCI "Predict Students' Dropout and Academic Success" (Realinho et al., 2022) — 4,424 students, 36 features, CC BY 4.0
- **Cross-validation:** UPV Longitudinal Dataset (Igualde-Sáez et al., 2025) — 20,427 students, 28 features + LMS behavioral data

### Preprocessing (order matters)
1. Label encode target (Dropout=0, Enrolled=1, Graduate=2)
2. Stratified 80/20 train/test split (random seed 42)
3. StandardScaler — fit on training data only (prevents leakage)
4. SMOTE — applied to training set only, after split (prevents leakage)

### Models
- **Logistic Regression** — interpretable linear baseline
- **Random Forest (200 trees)** — bagging ensemble, used for SHAP and fairness analysis
- **XGBoost (200 estimators)** — gradient boosting comparison
- **Stacking Ensemble** — RF + LR + XGBoost → Logistic Regression meta-learner (5-fold CV)

### Explainability
SHAP TreeExplainer on Random Forest, computed on a 200-sample test subset (seeded). Subgroup-stratified SHAP computed on the full test set for tuition and scholarship subgroups.

### Fairness
Disparate Impact Ratio (four-fifths rule) + Equalized Odds (TPR/FPR) across 4 protected attributes: gender, scholarship status, age, tuition status.

---

## 📈 Cross-Institutional Validation

The same methodology (RF + SMOTE + SHAP) was applied to the UPV dataset:

| Metric | Value |
|---|---|
| Accuracy | 93.1% |
| Macro F1 | 0.747 |
| Students | 20,427 |
| Dropout rate | 7.3% |
| Wi-Fi campus days SHAP rank | #8 (highest non-academic feature) |

**Key finding:** Academic performance features generalize across institutions (Portugal and Spain). Tuition diverges because only 2.5% of UPV students had unpaid tuition — low variance = low discriminative power.

---

## ⚖️ Fairness Audit Results

| Protected Attribute | Disparate Impact | Status | Key Finding |
|---|---|---|---|
| Scholarship holders | 0.183 | ❌ FAIL | Model catches only 47% of holder dropouts vs 74% for non-holders |
| Gender (M vs F) | 0.571 | ❌ FAIL | Detection near-equal (72.4% vs 72.7%), males falsely flagged more |
| Age (< 22 vs 30+) | 0.347 | ❌ FAIL | Older students flagged more aggressively |
| Tuition (current vs behind) | 0.222 | ❌ FAIL | 96% detection but 44% false positive rate for overdue students |

> ⚠️ **Deployment note:** Never rely solely on model output. Supplement with direct student contact, especially for scholarship holders.

---

## 📚 References

1. Liu, Z. et al. (2025). Student Dropout Prediction Using Ensemble Learning with SHAP. *JSSPA*, 2(3), 111–132.
2. Kim, S. et al. (2023). Student Dropout Prediction for University with High Precision and Recall. *Applied Sciences*, 13(10), 6275.
3. OECD (2025). Education at a Glance 2025: OECD Indicators.
4. Quimiz-Moreira, M. et al. (2025). Factors, Prediction, Explainability, and Simulating University Dropout Through ML. *Computation*, 13(8), 198.
5. Realinho, V. et al. (2022). Predicting Student Dropout and Academic Success. *Data*, 7(11), 146.
6. Villar, A. & de Andrade, C. (2024). Supervised ML for Predicting Student Dropout. *Discover AI*, 4(2).
7. Igualde-Sáez, A. et al. (2025). University Student Dropout: A Longitudinal Dataset. *Data*, 10(10), 162.

---

## 🔗 Links

Code: https://github.com/hammadmrza/student-dropout-prediction
Live App: https://student-dropout-prediction-itec6240.streamlit.app/
UCI Dataset: https://archive.ics.uci.edu/dataset/697
UPV Dataset: https://doi.org/10.3390/data10100162

---

*ITEC 6240 — Machine Learning and Its Applications | York University, Winter 2026*
