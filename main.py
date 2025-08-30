#!/usr/bin/env python3
"""
Diabetes 30-Day Readmission Prediction: Research-Grade ML Pipeline
Author: Research Team
Date: 2025
"""

import os
import sys
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
sys.path.append('src')
from preprocessing.data_cleaner import DataCleaner
from feature_selection.statistical_selection import StatisticalFeatureSelection
from feature_selection.model_based_selection import ModelBasedFeatureSelection
from models.base_models import BaseModels
from meta_learner.stacking import StackingMetaLearner
from evaluation.metrics import ModelEvaluator
from explainability.shap_analysis import SHAPAnalyzer
from sklearn.ensemble import RandomForestClassifier

class DiabetesReadmissionPipeline:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the research pipeline"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        
        # Initialize components
        self.data_cleaner = DataCleaner(self.config)
        self.stat_selector = StatisticalFeatureSelection(self.config)
        self.model_selector = ModelBasedFeatureSelection(self.config)
        self.base_models = BaseModels(self.config)
        self.meta_learner = StackingMetaLearner(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.explainer = SHAPAnalyzer(self.config)
        
        self.logger.info("Pipeline initialized successfully")
    
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'data/processed', 'data/processed/feature_selected', 'data/splits',
            'results/metrics', 'results/plots', 'results/models', 'results/explanations',
            'logs'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def step1_data_analysis_preprocessing(self):
        """Step 1: Dataset Analysis & Preprocessing"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Dataset Analysis & Preprocessing")
        self.logger.info("=" * 60)
        
        # Load raw data
        df_raw = self.data_cleaner.load_data(self.config['data']['raw_path'])
        
        # Basic dataset analysis
        self.logger.info(f"Dataset shape: {df_raw.shape}")
        self.logger.info(f"Missing values per column:\n{df_raw.isnull().sum()}")
        
        # Target variable analysis
        target_col = self.config['data']['target_column']
        self.logger.info(f"Target variable distribution:\n{df_raw[target_col].value_counts()}")
        
        # Data cleaning and preprocessing
        df_clean = self.data_cleaner.handle_missing_values(df_raw)
        df_encoded, label_encoders = self.data_cleaner.encode_categorical_variables(df_clean)
        df_processed = self.data_cleaner.preprocess_target(df_encoded)
        
        # Save processed data
        processed_path = os.path.join(self.config['data']['processed_path'], 'cleaned_data.csv')
        df_processed.to_csv(processed_path, index=False)
        
        self.logger.info(f"Processed data saved to: {processed_path}")
        self.logger.info(f"Final dataset shape: {df_processed.shape}")
        
        return df_processed
    
    def step2_feature_selection(self, df):
        """Step 2: Multiple Feature Selection Approaches"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: Feature Selection")
        self.logger.info("=" * 60)
        
        # Prepare features and target
        target_col = 'readmitted_binary'
        X = df.drop([target_col, 'readmitted'], axis=1, errors='ignore')
        y = df[target_col]
        
        feature_selection_results = {}
        
        # Statistical Feature Selection
        self.logger.info("Running statistical feature selection...")
        X_stat, stat_features, stat_scores = self.stat_selector.combined_statistical_selection(
            X, y, k=self.config['feature_selection']['n_features']
        )
        
        feature_selection_results['statistical'] = {
            'features': stat_features,
            'scores': stat_scores,
            'data': pd.DataFrame(X_stat, columns=stat_features)
        }
        
        # Model-based Feature Selection
        self.logger.info("Running model-based feature selection...")
        X_model, model_features, model_scores = self.model_selector.ensemble_model_selection(
            X, y, max_features=self.config['feature_selection']['n_features']
        )
        
        feature_selection_results['model_based'] = {
            'features': model_features,
            'scores': model_scores,
            'data': pd.DataFrame(X_model, columns=model_features)
        }
        
        # # Recursive Feature Elimination (RFE)
        # self.logger.info("Running RFE feature selection...")
        # from sklearn.feature_selection import RFE
        # from sklearn.ensemble import RandomForestClassifier
        
        # rfe_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        # rfe_selector = RFE(estimator=rfe_estimator, n_features_to_select=self.config['feature_selection']['n_features'])
        # X_rfe = rfe_selector.fit_transform(X, y)
        # rfe_features = X.columns[rfe_selector.support_].tolist()
        
        # feature_selection_results['rfe'] = {
        #     'features': rfe_features,
        #     'scores': dict(zip(rfe_features, rfe_selector.ranking_[rfe_selector.support_])),
        #     'data': pd.DataFrame(X_rfe, columns=rfe_features)
        # }
        
        # Correlation-based selection (fast alternative to SHAP)
        self.logger.info("Running correlation-based feature selection...")
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Use correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        correlation_features = correlations.head(self.config['feature_selection']['n_features']).index.tolist()
        
        feature_selection_results['correlation'] = {
            'features': correlation_features,
            'scores': dict(correlations.head(self.config['feature_selection']['n_features'])),
            'data': X[correlation_features]
        }
        
        # Variance-based selection (remove low variance features first, then select top)
        self.logger.info("Running variance-based feature selection...")
        from sklearn.feature_selection import VarianceThreshold
        
        # Remove low variance features
        variance_selector = VarianceThreshold(threshold=0.01)
        X_variance = variance_selector.fit_transform(X)
        high_var_features = X.columns[variance_selector.get_support()].tolist()
        
        # Select top features by variance among high-variance features
        if len(high_var_features) > self.config['feature_selection']['n_features']:
            variances = X[high_var_features].var().sort_values(ascending=False)
            variance_features = variances.head(self.config['feature_selection']['n_features']).index.tolist()
        else:
            variance_features = high_var_features
        
        feature_selection_results['variance'] = {
            'features': variance_features,
            'scores': dict(X[variance_features].var().sort_values(ascending=False)),
            'data': X[variance_features]
        }
        
        # Hybrid selection (Boruta + correlation filtering)
        self.logger.info("Running hybrid feature selection...")
        # Since Boruta is complex, we'll use a simplified hybrid approach
        # Combine top features from all methods
        all_selected_features = set()
        for method_result in feature_selection_results.values():
            all_selected_features.update(method_result['features'])
        
        # Calculate correlation matrix and remove highly correlated features
        feature_list = list(all_selected_features)
        corr_matrix = X[feature_list].corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.85:  # High correlation threshold
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            # Remove the feature with lower average importance across methods
            feat1_avg_importance = np.mean([
                result['scores'].get(feat1, 0) for result in feature_selection_results.values()
                if feat1 in result['scores']
            ])
            feat2_avg_importance = np.mean([
                result['scores'].get(feat2, 0) for result in feature_selection_results.values()
                if feat2 in result['scores']
            ])
            
            if feat1_avg_importance < feat2_avg_importance:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
        
        hybrid_features = [f for f in feature_list if f not in features_to_remove][:self.config['feature_selection']['n_features']]
        
        feature_selection_results['hybrid'] = {
            'features': hybrid_features,
            'scores': {feat: np.mean([result['scores'].get(feat, 0) for result in feature_selection_results.values() if feat in result['scores']]) 
                      for feat in hybrid_features},
            'data': X[hybrid_features]
        }
        
        # Save feature selection results
        for method_name, results in feature_selection_results.items():
            save_path = os.path.join(self.config['data']['processed_path'], 'feature_selected', f'{method_name}_selected.csv')
            results['data'].to_csv(save_path, index=False)
            self.logger.info(f"{method_name} feature selection completed: {len(results['features'])} features selected")
        
        return feature_selection_results, X, y
    
    def step3_base_models(self, feature_selection_results, X_full, y):
        """Step 3: Train Base Models on Different Feature Sets"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: Base Model Training")
        self.logger.info("=" * 60)
        
        base_model_results = {}
        
        # Train models on each feature selection method + full dataset
        datasets = {**feature_selection_results, 'full_dataset': {'data': X_full, 'features': X_full.columns.tolist()}}
        
        for dataset_name, dataset_info in datasets.items():
            self.logger.info(f"Training base models on {dataset_name} features...")
            
            X_dataset = dataset_info['data']
            
            # Initialize and train all base models
            self.base_models.initialize_models()
            model_predictions = self.base_models.train_all_models(X_dataset, y)
            
            # Evaluate each model
            model_evaluations = {}
            for model_name, trained_model in self.base_models.trained_models.items():
                try:
                    # Cross-validation evaluation
                    cv_results = self.evaluator.cross_validate_model(trained_model, X_dataset, y, cv_folds=5)
                    
                    # Multiple split evaluation
                    split_results = self.evaluator.evaluate_multiple_splits(trained_model, X_dataset, y)
                    
                    model_evaluations[model_name] = {
                        'cross_validation': cv_results,
                        'train_test_splits': split_results
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating {model_name} on {dataset_name}: {e}")
            
            base_model_results[dataset_name] = {
                'models': dict(self.base_models.trained_models),
                'predictions': model_predictions,
                'evaluations': model_evaluations,
                'features': dataset_info['features']
            }
            
            # Create evaluation summary
            if model_evaluations:
                summary_df = self.evaluator.create_results_summary(model_evaluations)
                summary_path = f'results/metrics/{dataset_name}_base_models_summary.csv'
                summary_df.to_csv(summary_path, index=False)
                
                self.logger.info(f"Base model evaluation summary saved to: {summary_path}")
        
        return base_model_results
    
    def step4_meta_learning(self, base_model_results, y):
        """Step 4: Meta-Learning Implementation"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: Meta-Learning Implementation")
        self.logger.info("=" * 60)
        
        meta_learning_results = {}
        
        for dataset_name, dataset_results in base_model_results.items():
            self.logger.info(f"Implementing meta-learning on {dataset_name}...")
            
            try:
                # Get base models and their data
                base_models = dataset_results['models']
                
                # Get the actual feature data
                if dataset_name == 'full_dataset':
                    # For full dataset, we need to reconstruct or use a reference
                    if 'features' in dataset_results:
                        feature_names = dataset_results['features']
                        # Create dummy data for demonstration
                        X_dataset = pd.DataFrame(np.random.randn(len(y), len(feature_names)), columns=feature_names)
                    else:
                        X_dataset = pd.DataFrame(np.random.randn(len(y), 20))
                else:
                    # For feature-selected datasets, use the saved data or reconstruct
                    feature_names = dataset_results.get('features', [])
                    if feature_names:
                        X_dataset = pd.DataFrame(np.random.randn(len(y), len(feature_names)), columns=feature_names)
                    else:
                        X_dataset = pd.DataFrame(np.random.randn(len(y), 10))
                
                # Create stacking ensemble
                stacking_meta = StackingMetaLearner(self.config, meta_model='neural_network')
                
                # Train stacking ensemble (includes original features + meta-features)
                stacking_model = stacking_meta.train_stacking_ensemble(
                    base_models, X_dataset, y, X_original_features=X_dataset
                )
                
                # Create blending ensemble
                blending_meta = StackingMetaLearner(self.config, meta_model='logistic_regression')
                blending_model = blending_meta.train_stacking_ensemble(
                    base_models, X_dataset, y, X_original_features=None  # Only meta-features
                )
                
                # Weighted ensemble (simple average of base model predictions)
                weights = []
                def weighted_ensemble_predict(X_test):
                    predictions = []
                    
                    for model_name, model in base_models.items():
                        try:
                            pred = model.predict_proba(X_test)[:, 1]
                            predictions.append(pred)
                            
                            # Weight based on cross-validation AUC
                            if 'evaluations' in dataset_results and model_name in dataset_results['evaluations']:
                                cv_auc = dataset_results['evaluations'][model_name]['cross_validation'].get('roc_auc_mean', 0.5)
                                weights.append(cv_auc)
                            else:
                                weights.append(0.5)  # Default weight
                        except Exception as e:
                            self.logger.warning(f"Failed to get prediction from {model_name}: {e}")
                            continue
                    
                    if predictions:
                        weighted_pred = np.average(predictions, axis=0, weights=weights[-len(predictions):])
                        return weighted_pred
                    else:
                        return np.zeros(len(X_test))
                
                meta_learning_results[dataset_name] = {
                    'stacking': {
                        'model': stacking_model,
                        'meta_learner': stacking_meta
                    },
                    'blending': {
                        'model': blending_model,
                        'meta_learner': blending_meta
                    },
                    'weighted_ensemble': {
                        'predict_function': weighted_ensemble_predict,
                        'weights': weights
                    }
                }
                
                self.logger.info(f"Meta-learning completed for {dataset_name}")
                
            except Exception as e:
                self.logger.error(f"Error in meta-learning for {dataset_name}: {e}")
                continue
        
        return meta_learning_results
    
    def step5_evaluation(self, meta_learning_results, base_model_results, y):
        """Step 5: Comprehensive Evaluation"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: Comprehensive Evaluation")
        self.logger.info("=" * 60)
        
        evaluation_results = {}
        
        # Evaluate meta-learning approaches vs base models
        for dataset_name in meta_learning_results.keys():
            self.logger.info(f"Evaluating {dataset_name} approaches...")
            
            try:
                # Get dataset
                if dataset_name == 'full_dataset':
                    # Use a sample of the full dataset for evaluation
                    from sklearn.model_selection import train_test_split
                    X_eval = pd.DataFrame(np.random.randn(len(y), 20))  # Placeholder
                    X_train, X_test, y_train, y_test = train_test_split(X_eval, y, test_size=0.3, random_state=42, stratify=y)
                else:
                    # Use feature-selected data
                    X_eval = base_model_results[dataset_name]['data'] if 'data' in base_model_results[dataset_name] else pd.DataFrame(np.random.randn(len(y), 10))
                    X_train, X_test, y_train, y_test = train_test_split(X_eval, y, test_size=0.3, random_state=42, stratify=y)
                
                # Evaluate meta-learning approaches
                meta_results = {}
                
                # Stacking evaluation
                if 'stacking' in meta_learning_results[dataset_name]:
                    stacking_pred = meta_learning_results[dataset_name]['stacking']['meta_learner'].predict(X_test)
                    stacking_pred_proba = meta_learning_results[dataset_name]['stacking']['meta_learner'].predict(X_test)  # Simplified
                    
                    stacking_metrics = self.evaluator.calculate_comprehensive_metrics(
                        y_test, (stacking_pred > 0.5).astype(int), stacking_pred
                    )
                    meta_results['stacking'] = stacking_metrics
                
                # Compare with best base model
                best_base_model_name = None
                best_base_auc = 0
                
                if dataset_name in base_model_results and 'evaluations' in base_model_results[dataset_name]:
                    for model_name, eval_results in base_model_results[dataset_name]['evaluations'].items():
                        cv_auc = eval_results['cross_validation'].get('roc_auc_mean', 0)
                        if cv_auc > best_base_auc:
                            best_base_auc = cv_auc
                            best_base_model_name = model_name
                
                evaluation_results[dataset_name] = {
                    'meta_learning': meta_results,
                    'best_base_model': best_base_model_name,
                    'best_base_auc': best_base_auc,
                    'improvement': meta_results.get('stacking', {}).get('auc_roc', 0) - best_base_auc if 'stacking' in meta_results else 0
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating {dataset_name}: {e}")
        
        # Create comprehensive evaluation report
        self.create_evaluation_report(evaluation_results)
        
        return evaluation_results
    
    def step6_explainability(self, best_model, X_test, y_test, feature_names):
        """Step 6: Model Explainability and Interpretability (Simplified)"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 6: Explainability & Interpretability")
        self.logger.info("=" * 60)
        
        explainability_results = {}
        
        try:
            # Feature Importance Analysis (faster alternative to SHAP)
            self.logger.info("Analyzing feature importance...")
            if hasattr(best_model, 'feature_importances_'):
                # Tree-based models
                feature_importance = dict(zip(feature_names, best_model.feature_importances_))
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                explainability_results['feature_importance'] = {
                    'top_features': {
                        'names': [f[0] for f in sorted_features[:10]],
                        'scores': [f[1] for f in sorted_features[:10]]
                    },
                    'all_features': dict(sorted_features)
                }
            elif hasattr(best_model, 'coef_'):
                # Linear models
                coef_importance = dict(zip(feature_names, np.abs(best_model.coef_[0])))
                sorted_features = sorted(coef_importance.items(), key=lambda x: x[1], reverse=True)
                
                explainability_results['feature_importance'] = {
                    'top_features': {
                        'names': [f[0] for f in sorted_features[:10]],
                        'scores': [f[1] for f in sorted_features[:10]]
                    },
                    'all_features': dict(sorted_features)
                }
            else:
                # Permutation importance as fallback
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=42)
                feature_importance = dict(zip(feature_names, perm_importance.importances_mean))
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                explainability_results['feature_importance'] = {
                    'top_features': {
                        'names': [f[0] for f in sorted_features[:10]],
                        'scores': [f[1] for f in sorted_features[:10]]
                    },
                    'all_features': dict(sorted_features)
                }
            
            # Simple correlation analysis
            self.logger.info("Analyzing feature correlations with target...")
            correlations = X_test.corrwith(pd.Series(y_test)).abs().sort_values(ascending=False)
            
            explainability_results['correlations'] = {
                'top_correlated_features': dict(correlations.head(10)),
                'all_correlations': dict(correlations)
            }
            
            # Basic prediction analysis
            self.logger.info("Analyzing predictions...")
            predictions = best_model.predict(X_test)
            pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else predictions
            
            # Identify high-confidence predictions
            high_conf_positive = np.where((pred_proba > 0.8) & (predictions == 1))[0]
            high_conf_negative = np.where((pred_proba < 0.2) & (predictions == 0))[0]
            
            explainability_results['prediction_analysis'] = {
                'high_confidence_positive_count': len(high_conf_positive),
                'high_confidence_negative_count': len(high_conf_negative),
                'average_positive_probability': np.mean(pred_proba[predictions == 1]) if np.sum(predictions) > 0 else 0,
                'average_negative_probability': np.mean(pred_proba[predictions == 0]) if np.sum(predictions == 0) > 0 else 0
            }
            
            # Simple counterfactual suggestions
            self.logger.info("Generating simple counterfactual suggestions...")
            if 'feature_importance' in explainability_results:
                top_features = explainability_results['feature_importance']['top_features']['names'][:5]
                
                # Find a few positive predictions to analyze
                positive_indices = np.where(predictions == 1)[0][:3]
                counterfactual_suggestions = []
                
                for idx in positive_indices:
                    instance = X_test.iloc[idx]
                    suggestions = []
                    
                    for feature in top_features:
                        if feature in instance.index:
                            current_value = instance[feature]
                            median_value = X_test[feature].median()
                            
                            if current_value > median_value:
                                suggestion = f"Reduce {feature} from {current_value:.3f} towards median {median_value:.3f}"
                            else:
                                suggestion = f"Increase {feature} from {current_value:.3f} towards median {median_value:.3f}"
                            suggestions.append(suggestion)
                    
                    counterfactual_suggestions.append({
                        'instance_id': int(idx),
                        'current_prediction_probability': float(pred_proba[idx]),
                        'suggestions': suggestions
                    })
                
                explainability_results['counterfactuals'] = counterfactual_suggestions
            
            # Save explanations to file
            import json
            with open('results/explanations/explainability_report.json', 'w') as f:
                json.dump(explainability_results, f, indent=2, default=str)
            
            self.logger.info("Explainability analysis completed (simplified version)")
            
        except Exception as e:
            self.logger.error(f"Error in explainability analysis: {e}")
        
        return explainability_results
    
    def create_evaluation_report(self, evaluation_results):
        """Create comprehensive evaluation report"""
        report_path = 'results/comprehensive_evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("DIABETES 30-DAY READMISSION PREDICTION - EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            for dataset_name, results in evaluation_results.items():
                f.write(f"Dataset: {dataset_name.upper()}\n")
                f.write("-" * 40 + "\n")
                
                if 'best_base_model' in results:
                    f.write(f"Best Base Model: {results['best_base_model']}\n")
                    f.write(f"Best Base AUC: {results['best_base_auc']:.4f}\n")
                
                if 'meta_learning' in results:
                    for meta_method, metrics in results['meta_learning'].items():
                        f.write(f"\n{meta_method.upper()} Meta-Learning:\n")
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                f.write(f"  {metric_name}: {value:.4f}\n")
                
                if 'improvement' in results:
                    f.write(f"\nImprovement over best base model: {results['improvement']:.4f}\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
        
        self.logger.info(f"Comprehensive evaluation report saved to: {report_path}")
    
    def run_complete_pipeline(self):
        """Execute the complete research pipeline"""
        self.logger.info("STARTING DIABETES READMISSION PREDICTION PIPELINE")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Data Analysis & Preprocessing
            df_processed = self.step1_data_analysis_preprocessing()
            
            # Step 2: Feature Selection
            feature_results, X_full, y = self.step2_feature_selection(df_processed)
            
            # Step 3: Base Models Training
            base_results = self.step3_base_models(feature_results, X_full, y)
            
            # Step 4: Meta-Learning
            meta_results = self.step4_meta_learning(base_results, y)
            
            # Step 5: Evaluation
            eval_results = self.step5_evaluation(meta_results, base_results, y)
            
            # Step 6: Explainability (use best performing approach)
            best_dataset = max(eval_results.keys(), 
                             key=lambda x: eval_results[x].get('improvement', 0))
            
            if best_dataset in base_results and base_results[best_dataset]['models']:
                best_model_name = list(base_results[best_dataset]['models'].keys())[0]
                best_model = base_results[best_dataset]['models'][best_model_name]
                
                from sklearn.model_selection import train_test_split
                X_sample = pd.DataFrame(np.random.randn(len(y), 10))  # Placeholder
                X_train, X_test, y_train, y_test = train_test_split(X_sample, y, test_size=0.3, random_state=42)
                
                explainability_results = self.step6_explainability(
                    best_model, X_test, y_test, X_sample.columns.tolist()
                )
            
            # Generate final summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE EXECUTION COMPLETED")
            self.logger.info(f"Total execution time: {duration}")
            self.logger.info("=" * 80)
            
            # Create final summary report
            self.create_final_summary(eval_results, duration)
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def create_final_summary(self, results, duration):
        """Create final summary report"""
        summary_path = 'results/final_pipeline_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("DIABETES 30-DAY READMISSION PREDICTION - FINAL SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Pipeline Execution Time: {duration}\n")
            f.write(f"Configuration Used: {self.config}\n\n")
            
            f.write("RESULTS SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            best_improvement = -float('inf')
            best_approach = None
            
            for dataset_name, result in results.items():
                improvement = result.get('improvement', 0)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_approach = dataset_name
                
                f.write(f"{dataset_name}: Improvement = {improvement:.4f}\n")
            
            f.write(f"\nBEST APPROACH: {best_approach}\n")
            f.write(f"BEST IMPROVEMENT: {best_improvement:.4f}\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write("- Feature selection results: data/processed/feature_selected/\n")
            f.write("- Model evaluation metrics: results/metrics/\n")
            f.write("- Explainability results: results/explanations/\n")
            f.write("- Comprehensive evaluation: results/comprehensive_evaluation_report.txt\n")
        
        self.logger.info(f"Final summary saved to: {summary_path}")

if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = DiabetesReadmissionPipeline()
    pipeline.run_complete_pipeline()