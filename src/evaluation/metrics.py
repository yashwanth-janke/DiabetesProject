import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    brier_score_loss, precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import logging

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results = {}
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        # AUC metrics
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value (precision)
        
        # Brier Score for probability calibration
        metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        
        # Additional metrics
        metrics['true_positives'] = tp
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv_folds=5, scoring_metrics=None):
        """Perform cross-validation with multiple metrics"""
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
            cv_results[f'{metric}_scores'] = scores.tolist()
        
        return cv_results
    
    def evaluate_multiple_splits(self, model, X, y, test_sizes=[0.3, 0.2, 0.1]):
        """Evaluate model on multiple train-test splits"""
        from sklearn.model_selection import train_test_split
        
        split_results = {}
        
        for test_size in test_sizes:
            train_size = 1 - test_size
            split_key = f"train_{int(train_size*100)}_test_{int(test_size*100)}"
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            split_results[split_key] = metrics
            
            self.logger.info(f"Evaluation completed for {split_key} split")
        
        return split_results
    
    def compare_models(self, models_dict, X, y, cv_folds=5):
        """Compare multiple models using cross-validation"""
        comparison_results = {}
        
        for model_name, model in models_dict.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            try:
                # Cross-validation results
                cv_results = self.cross_validate_model(model, X, y, cv_folds)
                
                # Multiple split results
                split_results = self.evaluate_multiple_splits(model, X, y)
                
                comparison_results[model_name] = {
                    'cross_validation': cv_results,
                    'train_test_splits': split_results
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}
        
        return comparison_results
    
    def statistical_significance_test(self, model1_scores, model2_scores):
        """Perform statistical significance test between two models"""
        from scipy import stats
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_p_value = stats.wilcoxon(model1_scores, model2_scores)
        
        return {
            'paired_t_test': {'statistic': t_stat, 'p_value': p_value},
            'wilcoxon_test': {'statistic': w_stat, 'p_value': w_p_value}
        }
    
    def create_results_summary(self, results_dict):
        """Create a comprehensive results summary"""
        summary_data = []
        
        for model_name, results in results_dict.items():
            if 'error' in results:
                continue
                
            cv_results = results.get('cross_validation', {})
            
            summary_row = {
                'Model': model_name,
                'CV_Accuracy_Mean': cv_results.get('accuracy_mean', 0),
                'CV_Accuracy_Std': cv_results.get('accuracy_std', 0),
                'CV_Precision_Mean': cv_results.get('precision_mean', 0),
                'CV_Precision_Std': cv_results.get('precision_std', 0),
                'CV_Recall_Mean': cv_results.get('recall_mean', 0),
                'CV_Recall_Std': cv_results.get('recall_std', 0),
                'CV_F1_Mean': cv_results.get('f1_mean', 0),
                'CV_F1_Std': cv_results.get('f1_std', 0),
                'CV_AUC_Mean': cv_results.get('roc_auc_mean', 0),
                'CV_AUC_Std': cv_results.get('roc_auc_std', 0)
            }
            
            # Add train-test split results
            split_results = results.get('train_test_splits', {})
            for split_name, metrics in split_results.items():
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        summary_row[f'{split_name}_{metric_name}'] = value
            
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)