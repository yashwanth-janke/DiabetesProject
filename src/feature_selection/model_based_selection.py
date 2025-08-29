import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
import logging

class ModelBasedFeatureSelection:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_importance = {}
    
    def random_forest_selection(self, X, y, n_estimators=100, max_features=20):
        """Random Forest feature importance selection"""
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importance
        if hasattr(X, 'columns'):
            feature_names = X.columns
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        importance_scores = dict(zip(feature_names, rf.feature_importances_))
        
        # Select top features
        selector = SelectFromModel(rf, max_features=max_features, prefit=True)
        X_selected = selector.transform(X)
        
        selected_features = [name for name, selected in zip(feature_names, selector.get_support()) if selected]
        
        self.feature_importance['random_forest'] = importance_scores
        self.logger.info(f"Random Forest selection: {len(selected_features)} features selected")
        
        return X_selected, selected_features, importance_scores
    
    def xgboost_selection(self, X, y, max_features=20):
        """XGBoost feature importance selection"""
        xgb = XGBClassifier(
            random_state=42,
            eval_metric='logloss'
        )
        xgb.fit(X, y)
        
        if hasattr(X, 'columns'):
            feature_names = X.columns
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        importance_scores = dict(zip(feature_names, xgb.feature_importances_))
        
        # Select top features
        selector = SelectFromModel(xgb, max_features=max_features, prefit=True)
        X_selected = selector.transform(X)
        
        selected_features = [name for name, selected in zip(feature_names, selector.get_support()) if selected]
        
        self.feature_importance['xgboost'] = importance_scores
        self.logger.info(f"XGBoost selection: {len(selected_features)} features selected")
        
        return X_selected, selected_features, importance_scores
    
    def logistic_regression_selection(self, X, y, max_features=20):
        """L1 regularized logistic regression for feature selection"""
        lr = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        lr.fit(X, y)
        
        if hasattr(X, 'columns'):
            feature_names = X.columns
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        importance_scores = dict(zip(feature_names, np.abs(lr.coef_[0])))
        
        # Select top features
        selector = SelectFromModel(lr, max_features=max_features, prefit=True)
        X_selected = selector.transform(X)
        
        selected_features = [name for name, selected in zip(feature_names, selector.get_support()) if selected]
        
        self.feature_importance['logistic_regression'] = importance_scores
        self.logger.info(f"Logistic Regression selection: {len(selected_features)} features selected")
        
        return X_selected, selected_features, importance_scores
    
    def ensemble_model_selection(self, X, y, max_features=20):
        """Combine multiple model-based selections"""
        # Get importance from different models
        rf_X, rf_features, rf_scores = self.random_forest_selection(X, y, max_features=50)
        xgb_X, xgb_features, xgb_scores = self.xgboost_selection(X, y, max_features=50)
        lr_X, lr_features, lr_scores = self.logistic_regression_selection(X, y, max_features=50)
        
        # Combine scores (normalize first)
        all_features = set(rf_features + xgb_features + lr_features)
        combined_scores = {}
        
        for feature in all_features:
            score = 0
            count = 0
            
            if feature in rf_scores:
                score += rf_scores[feature] / max(rf_scores.values())
                count += 1
            
            if feature in xgb_scores:
                score += xgb_scores[feature] / max(xgb_scores.values()) 
                count += 1
                
            if feature in lr_scores:
                score += lr_scores[feature] / max(lr_scores.values())
                count += 1
            
            combined_scores[feature] = score / count if count > 0 else 0
        
        # Select top features
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat[0] for feat in sorted_features[:max_features]]
        
        if hasattr(X, 'columns'):
            X_selected = X[selected_features]
        else:
            feature_indices = [list(X.columns if hasattr(X, 'columns') else range(X.shape[1])).index(feat) 
                             for feat in selected_features if feat in (X.columns if hasattr(X, 'columns') else range(X.shape[1]))]
            X_selected = X[:, feature_indices]
        
        self.logger.info(f"Ensemble model selection: {len(selected_features)} features selected")
        
        return X_selected, selected_features, dict(sorted_features[:max_features])