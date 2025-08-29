import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import logging

class SHAPAnalyzer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, model, X_background, model_type='tree'):
        """Create SHAP explainer based on model type"""
        if model_type == 'tree':
            # For tree-based models (RF, XGBoost, etc.)
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            # For linear models
            self.explainer = shap.LinearExplainer(model, X_background)
        elif model_type == 'kernel':
            # For any model (slower but universal)
            self.explainer = shap.KernelExplainer(
                model.predict_proba, 
                X_background.sample(min(100, len(X_background)))
            )
        elif model_type == 'deep':
            # For neural networks
            self.explainer = shap.DeepExplainer(model, X_background)
        else:
            # Default to Kernel explainer
            self.explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1], 
                X_background.sample(min(100, len(X_background)))
            )
        
        self.logger.info(f"SHAP explainer created with type: {model_type}")
        return self.explainer
    
    def calculate_shap_values(self, X_explain, max_samples=None):
        """Calculate SHAP values for explanation dataset"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        # Limit samples for computational efficiency
        if max_samples and len(X_explain) > max_samples:
            X_sample = X_explain.sample(max_samples, random_state=42)
        else:
            X_sample = X_explain
        
        try:
            self.shap_values = self.explainer.shap_values(X_sample)
            
            # For binary classification, take positive class SHAP values
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Positive class
            
            self.logger.info(f"SHAP values calculated for {len(X_sample)} samples")
            return self.shap_values
            
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {e}")
            raise
    
    def plot_feature_importance(self, feature_names, save_path=None):
        """Plot SHAP feature importance"""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            features=feature_names,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def plot_summary_plot(self, X_explain, feature_names, save_path=None):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            self.shap_values,
            X_explain,
            feature_names=feature_names,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def plot_waterfall(self, instance_index, X_explain, feature_names, save_path=None):
        """Create waterfall plot for individual prediction"""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_index],
                base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value,
                data=X_explain.iloc[instance_index].values,
                feature_names=feature_names
            ),
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def get_top_features(self, n_features=10):
        """Get top N most important features based on mean absolute SHAP values"""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        mean_shap_values = np.mean(np.abs(self.shap_values), axis=0)
        top_indices = np.argsort(mean_shap_values)[-n_features:][::-1]
        
        return top_indices, mean_shap_values[top_indices]
    
    def feature_interaction_analysis(self, X_explain, feature_names):
        """Analyze feature interactions using SHAP"""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")
        
        # Calculate interaction values (computationally expensive)
        try:
            interaction_values = self.explainer.shap_interaction_values(X_explain.sample(min(500, len(X_explain))))
            
            # Plot interaction summary
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                interaction_values,
                X_explain.sample(min(500, len(X_explain))),
                feature_names=feature_names,
                show=False
            )
            plt.tight_layout()
            plt.show()
            
            return interaction_values
            
        except Exception as e:
            self.logger.warning(f"Interaction analysis failed: {e}")
            return None
    
    def generate_explanations_report(self, X_explain, feature_names, model_name, save_dir=None):
        """Generate comprehensive explanations report"""
        report = {
            'model_name': model_name,
            'n_samples': len(X_explain),
            'n_features': len(feature_names)
        }
        
        # Global feature importance
        top_indices, top_values = self.get_top_features(n_features=min(20, len(feature_names)))
        report['top_features'] = {
            'names': [feature_names[i] for i in top_indices],
            'importance_scores': top_values.tolist()
        }
        
        # Mean SHAP values per feature
        mean_shap = np.mean(self.shap_values, axis=0)
        report['mean_shap_values'] = dict(zip(feature_names, mean_shap.tolist()))
        
        # Feature statistics
        report['feature_stats'] = {
            'positive_impact_features': [feature_names[i] for i in np.where(mean_shap > 0)[0]],
            'negative_impact_features': [feature_names[i] for i in np.where(mean_shap < 0)[0]]
        }
        
        if save_dir:
            import json
            import os
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f'{model_name}_shap_report.json'), 'w') as f:
                json.dump(report, f, indent=2)
        
        return report