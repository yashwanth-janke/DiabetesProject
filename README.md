# Diabetes 30-Day Readmission Prediction: Research-Grade ML Pipeline

## Overview
This repository contains a comprehensive, research-grade machine learning pipeline for predicting 30-day hospital readmissions in diabetes patients. The system implements a novel meta-learning approach called "Prediction2Feature" that uses base model predictions as features for enhanced neural network learning.

## Features

### 🔬 Research-Grade Implementation
- **Modular Architecture**: Clean, maintainable code structure
- **Multiple Feature Selection Methods**: Statistical, model-based, wrapper, SHAP, and hybrid approaches
- **Comprehensive Evaluation**: Multiple train-test splits, k-fold CV, statistical significance tests
- **Advanced Meta-Learning**: Stacking, blending, and weighted ensemble approaches
- **Full Explainability Suite**: SHAP, LIME, and counterfactual analysis

### 📊 Dataset Handling
- **Robust Preprocessing**: Handles missing values, categorical encoding, class imbalance
- **Domain-Specific Features**: ICD-9 code grouping, medication encoding, HbA1c processing
- **Multiple Imbalance Techniques**: SMOTE, SMOTE-ENN, class weights, cost-sensitive learning

### 🤖 Advanced Modeling
- **Base Models**: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost, ANN
- **Meta-Learning**: Neural network meta-learner trained on base model predictions + original features
- **Hyperparameter Tuning**: Grid search optimization for all models

## Quick Start

### Installation
```bash
git clone <repository-url>
cd diabetes_readmission_prediction
pip install -r requirements.txt