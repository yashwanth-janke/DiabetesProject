import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import logging

class BaseModels:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.trained_models = {}
        self.logger = logging.getLogger(__name__)
    
    def initialize_models(self):
        """Initialize all base models with hyperparameters"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=5
            ),
            'svm': SVC(
                random_state=42,
                class_weight='balanced',
                probability=True  # Important for meta-learning
            ),
            'xgboost': XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=1  # Will adjust based on class imbalance
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=500
            )
        }
        
        return self.models
    
    def train_model(self, model_name, X_train, y_train, hyperparameter_tuning=True):
        """Train individual model with optional hyperparameter tuning"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hyperparameter_tuning:
            model = self._tune_hyperparameters(model_name, model, X_train, y_train)
        
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        
        self.logger.info(f"{model_name} trained successfully")
        return model
    
    def _tune_hyperparameters(self, model_name, model, X_train, y_train):
        """Hyperparameter tuning for each model"""
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'decision_tree': {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name],
                cv=3,
                scoring='f1',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        return model
    
    def train_all_models(self, X_train, y_train):
        """Train all base models"""
        self.initialize_models()
        predictions = {}
        
        for model_name in self.models.keys():
            try:
                model = self.train_model(model_name, X_train, y_train)
                predictions[model_name] = model.predict_proba(X_train)[:, 1]  # Probability of positive class
                self.logger.info(f"Successfully trained {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        return predictions
    
    def get_predictions(self, X_test, model_names=None):
        """Get predictions from trained models"""
        if model_names is None:
            model_names = list(self.trained_models.keys())
        
        predictions = {}
        for model_name in model_names:
            if model_name in self.trained_models:
                model = self.trained_models[model_name]
                predictions[model_name] = model.predict_proba(X_test)[:, 1]
            else:
                self.logger.warning(f"Model {model_name} not found in trained models")
        
        return predictions