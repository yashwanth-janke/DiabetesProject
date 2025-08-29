import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile
import logging

class StatisticalFeatureSelection:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.selected_features = None
        self.scores = None
    
    def chi_square_selection(self, X, y, k=20):
        """Chi-square test for categorical features"""
        selector = SelectKBest(score_func=chi2, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get feature names if X is DataFrame
        if hasattr(X, 'columns'):
            selected_features = X.columns[selector.get_support()].tolist()
        else:
            selected_features = [f'feature_{i}' for i in range(X_selected.shape[1])]
        
        scores = selector.scores_
        
        self.logger.info(f"Chi-square selection: {len(selected_features)} features selected")
        return X_selected, selected_features, scores
    
    def f_test_selection(self, X, y, k=20):
        """F-test for numerical features"""
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        if hasattr(X, 'columns'):
            selected_features = X.columns[selector.get_support()].tolist()
        else:
            selected_features = [f'feature_{i}' for i in range(X_selected.shape[1])]
        
        scores = selector.scores_
        
        self.logger.info(f"F-test selection: {len(selected_features)} features selected")
        return X_selected, selected_features, scores
    
    def mutual_info_selection(self, X, y, k=20):
        """Mutual information for feature selection"""
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        if hasattr(X, 'columns'):
            selected_features = X.columns[selector.get_support()].tolist()
        else:
            selected_features = [f'feature_{i}' for i in range(X_selected.shape[1])]
        
        scores = selector.scores_
        
        self.logger.info(f"Mutual info selection: {len(selected_features)} features selected")
        return X_selected, selected_features, scores
    
    def combined_statistical_selection(self, X, y, k=20):
        """Combine multiple statistical tests"""
        # Separate numerical and categorical features
        if hasattr(X, 'dtypes'):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        else:
            # Assume all numeric if no dtype info
            numeric_cols = list(range(X.shape[1]))
            categorical_cols = []
        
        all_scores = {}
        
        # F-test for numerical features
        if len(numeric_cols) > 0:
            _, _, f_scores = self.f_test_selection(X[numeric_cols], y, k=min(k, len(numeric_cols)))
            for i, col in enumerate(numeric_cols):
                all_scores[col] = f_scores[i]
        
        # Chi-square for categorical features  
        if len(categorical_cols) > 0:
            X_cat_encoded = pd.get_dummies(X[categorical_cols])
            _, _, chi_scores = self.chi_square_selection(X_cat_encoded, y, k=min(k, len(X_cat_encoded.columns)))
            for i, col in enumerate(X_cat_encoded.columns):
                all_scores[col] = chi_scores[i]
        
        # Mutual information for all features
        _, _, mi_scores = self.mutual_info_selection(X, y, k=k)
        if hasattr(X, 'columns'):
            for i, col in enumerate(X.columns):
                all_scores[f'{col}_mi'] = mi_scores[i]
        
        # Rank features by combined scores
        sorted_features = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat[0] for feat in sorted_features[:k]]
        
        self.selected_features = selected_features
        self.scores = dict(sorted_features[:k])
        
        return X[selected_features], selected_features, self.scores