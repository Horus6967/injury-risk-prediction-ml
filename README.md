# Injury Risk Prediction using Machine Learning

## Overview
This project develops a machine learning model to predict athlete injury risk using longitudinal training workload data. The model leverages features such as training intensity, rest patterns, and session frequency to identify patterns associated with injury occurrence.

The goal of this project is to address the challenge of early injury detection, which is critical in sports science, by prioritizing recall to ensure that potential injuries are not missed.

---

## Results
- Random Forest AUC: 0.745  
- Logistic Regression AUC: 0.616  
- Best Threshold: 0.296  
- Precision (injury class): 0.034  
- Recall (injury class): 0.327  

The model achieves significantly improved recall compared to baseline approaches, highlighting its effectiveness in identifying injury cases despite severe class imbalance.

---

## Methods

### Data Processing
- Cleaned and structured time-series training data
- Removed irrelevant or redundant features
- Handled missing values

### Feature Engineering
- Weekly training load features
- Rest and recovery metrics
- Intensity zone aggregations (Z1–Z5)
- Session-based metrics (intervals, strength training)

### Class Imbalance Handling
- Applied SMOTE (Synthetic Minority Oversampling Technique)
- Balanced training data from highly skewed distribution

### Models
- Random Forest Classifier (primary model)
- Logistic Regression (baseline comparison)

### Evaluation
- ROC-AUC for overall performance
- Precision, Recall, F1-score
- Confusion Matrix
- Threshold tuning for optimal recall

### Model Interpretability
- SHAP (SHapley Additive Explanations)
- Feature importance ranking
- Beeswarm and interaction plots

---

## Key Visualizations

### Confusion Matrix
<img width="515" height="455" alt="image" src="https://github.com/user-attachments/assets/d1256b49-8d58-4cbf-87b7-ba48c7de42be" />


### ROC Curve
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/42c87925-460f-4338-85a4-77de0d62b7bd" />



### SHAP Feature Importance
<img width="868" height="497" alt="image" src="https://github.com/user-attachments/assets/894d9a07-59be-47a0-b4b4-20ece864adc5" />


---

## Dataset

This project uses the publicly available running injury dataset:

Ramskov, D., et al. Running-related injuries and their association with changes in training load.

The dataset contains longitudinal training data for runners, including:
- Training sessions
- Rest days
- Mileage across intensity zones (Z1–Z5)
- Strength training frequency
- Interval sessions

The processed dataset used in this project is:
week_approach_maskedID_timeseries.csv

### How to Access
Download the dataset here:
https://zenodo.org/record/4553123

### Setup Instructions
1. Download the dataset
2. Extract the files
3. Place `week_approach_maskedID_timeseries.csv` in the root directory of this repository

### Notes
- Data is anonymized (masked athlete IDs)
- This project uses engineered time-series features derived from weekly training load
- No personal or identifiable data is included

---

## Repository Structure
- injury_model.py → full machine learning pipeline
- Running.ipynb → notebook version of the workflow
- Confusion Matrix.png → evaluation visualization
- ROC curve injury prediction.png → ROC curve
- SHAP Value.png → feature importance visualization
- README.md → project documentation

---

## How to Run

### Install dependencies
```
pip install numpy pandas scikit-learn matplotlib seaborn shap imbalanced-learn
```

### Run the model
```
python injury_model.py
```

---

## Key Insights
- Training load and recovery balance are strong predictors of injury risk  
- Strength training frequency and rest days play significant roles  
- High-intensity workload spikes contribute to increased injury probability  
- Model performance is limited by class imbalance, emphasizing recall-focused optimization  

---

## Future Work
- Incorporate temporal deep learning models (LSTM, Transformers)
- Improve precision while maintaining high recall
- Expand dataset with more athletes and longer time horizons
- Deploy as a real-time injury risk monitoring system

---

## Author
Shourya Kukkala

---

## References
Ramskov, D., Nielsen, R. O., Rasmussen, S., & Sørensen, H. (2021). Running-related injuries and their association with changes in training load.
