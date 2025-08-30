#!/usr/bin/env python3
"""
Optimized Diabetes 30-Day Readmission Prediction Pipeline
Fast training version with reduced complexity
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

# Data processing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold

# Feature selection
from sklearn.feature_selection import SelectKBest, chi2, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Base models (fast training)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# TensorFlow for meta-learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class OptimizedDiabetesReadmissionPipeline:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize optimized pipeline"""
        self.config = self.load_config(config_path) if os.path.exists(config_path) else self.default_config()
        self.setup_logging()
        
        # Pipeline components
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selectors = {}
        self.base_models = {}
        self.meta_model = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Optimized pipeline initialized")
    
    def default_config(self):
        """Default configuration"""
        return {
            'data': {
                'raw_path': 'data/raw/diabetic_data.csv',
                'processed_path': 'data/processed/'
            },
            'preprocessing': {
                'missing_threshold': 0.5,
                'test_sizes': [0.2, 0.25, 0.3]  # 3 train-test splits
            },
            'feature_selection': {
                'n_features': 20,
                'methods': ['statistical_chi2', 'statistical_f', 'rf_importance', 'combined']
            }
        }
    
    def load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return self.default_config()
    
    def setup_logging(self):
        """Setup logging"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/optimized_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path):
        """Load diabetes dataset"""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Data loaded: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df):
        """Efficient missing value handling"""
        df_clean = df.copy()
        
        # Drop columns with high missingness
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        high_missing_cols = missing_pct[missing_pct > self.config['preprocessing']['missing_threshold']].index
        
        if len(high_missing_cols) > 0:
            self.logger.info(f"Dropping high-missingness columns: {list(high_missing_cols)}")
            df_clean = df_clean.drop(columns=high_missing_cols)
        
        # Fast imputation for remaining missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['int64', 'float64']:
                    # Median for numerical
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    # Mode for categorical
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        self.logger.info(f"Missing values handled. Final shape: {df_clean.shape}")
        return df_clean
    
    def encode_categorical_variables(self, df):
        """Fast categorical encoding"""
        df_encoded = df.copy()
        label_encoders = {}
        
        # Simple ICD-9 grouping (fast version)
        icd9_mapping = {
            'circulatory': ['390', '391', '392', '393', '394', '395', '396', '397', '398', 
                           '401', '402', '403', '404', '405', '410', '411', '412', '413', '414', '415', '416', '417', '418', '419',
                           '420', '421', '422', '423', '424', '425', '426', '427', '428', '429',
                           '430', '431', '432', '433', '434', '435', '436', '437', '438', '439',
                           '440', '441', '442', '443', '444', '445', '446', '447', '448', '449',
                           '450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '785'],
            'respiratory': ['460', '470', '480', '490', '500', '510', '786'],
            'digestive': ['520', '530', '540', '550', '560', '570', '787'],
            'diabetes': ['250'],
            'injury': ['800', '810', '820', '830', '840', '850', '860', '870', '880', '890'],
            'other': []
        }
        
        # Fast ICD-9 categorization for diagnosis columns
        for diag_col in ['diag_1', 'diag_2', 'diag_3']:
            if diag_col in df_encoded.columns:
                df_encoded[f'{diag_col}_group'] = df_encoded[diag_col].astype(str).str[:3].apply(
                    lambda x: self._fast_icd9_categorize(x, icd9_mapping)
                )
        
        # Simple medication encoding (binary: changed or not)
        medication_cols = [col for col in df_encoded.columns if col in [
            'metformin', 'insulin', 'glyburide', 'glipizide', 'glimepiride'
        ]]
        
        for col in medication_cols:
            if col in df_encoded.columns:
                df_encoded[f'{col}_changed'] = (df_encoded[col].isin(['Up', 'Down', 'Steady'])).astype(int)
        
        # HbA1c encoding
        if 'A1Cresult' in df_encoded.columns:
            hba1c_mapping = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
            df_encoded['A1Cresult_encoded'] = df_encoded['A1Cresult'].map(hba1c_mapping).fillna(0)
        
        # Label encode remaining categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop(['readmitted'], errors='ignore')
        
        for col in categorical_cols:
            if col not in label_encoders:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le
        
        return df_encoded, label_encoders
    
    def _fast_icd9_categorize(self, code, icd9_mapping):
        """Fast ICD-9 categorization"""
        if pd.isna(code) or code == 'nan':
            return 'missing'
        
        code_prefix = str(code)[:3]
        
        for category, prefixes in icd9_mapping.items():
            if any(code_prefix.startswith(prefix) for prefix in prefixes):
                return category
        
        return 'other'
    
    def preprocess_target(self, df):
        """Process target variable"""
        if 'readmitted' in df.columns:
            # Binary encoding: <30 days = 1, others = 0
            target_mapping = {'<30': 1, '>30': 0, 'NO': 0}
            df['readmitted_binary'] = df['readmitted'].map(target_mapping)
            return df
        else:
            raise ValueError("Target column 'readmitted' not found")

class ClassImbalanceHandler:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_imbalance(self, y):
        """Analyze class imbalance in the dataset"""
        class_counts = y.value_counts().sort_index()
        total_samples = len(y)
        
        imbalance_info = {
            'class_counts': class_counts.to_dict(),
            'class_percentages': (class_counts / total_samples * 100).to_dict(),
            'total_samples': total_samples
        }
        
        if len(class_counts) == 2:
            majority_count = class_counts.max()
            minority_count = class_counts.min()
            imbalance_info['imbalance_ratio'] = majority_count / minority_count
            imbalance_info['minority_percentage'] = minority_count / total_samples * 100
        
        return imbalance_info
    
    def calculate_class_weights(self, y):
        """Calculate optimal class weights for imbalanced dataset"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        weight_dict = dict(zip(classes, class_weights))
        
        self.logger.info(f"Calculated class weights: {weight_dict}")
        return weight_dict
    
    def demonstrate_effectiveness(self, y_true, y_pred_without_weights, y_pred_with_weights):
        """Demonstrate the effectiveness of class weights"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n" + "="*60)
        print("CLASS WEIGHTS EFFECTIVENESS DEMONSTRATION")
        print("="*60)
        
        print("\nWITHOUT Class Weights:")
        print(classification_report(y_true, y_pred_without_weights, target_names=['No Readmission', 'Readmission']))
        cm_without = confusion_matrix(y_true, y_pred_without_weights)
        print(f"Confusion Matrix:\n{cm_without}")
        
        print("\nWITH Class Weights (Balanced):")
        print(classification_report(y_true, y_pred_with_weights, target_names=['No Readmission', 'Readmission']))
        cm_with = confusion_matrix(y_true, y_pred_with_weights)
        print(f"Confusion Matrix:\n{cm_with}")
        
        # Calculate improvement in minority class recall
        recall_without = cm_without[1,1] / (cm_without[1,1] + cm_without[1,0]) if (cm_without[1,1] + cm_without[1,0]) > 0 else 0
        recall_with = cm_with[1,1] / (cm_with[1,1] + cm_with[1,0]) if (cm_with[1,1] + cm_with[1,0]) > 0 else 0
        
        improvement = recall_with - recall_without
        print(f"\nMinority Class Recall Improvement: {improvement:.3f} ({improvement*100:.1f}%)")
        
        return {
            'recall_improvement': improvement,
            'precision_with_weights': cm_with[1,1] / (cm_with[1,1] + cm_with[0,1]) if (cm_with[1,1] + cm_with[0,1]) > 0 else 0,
            'recall_with_weights': recall_with
        }
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.selectors = {}
    
    def statistical_chi2_selection(self, X, y, k=20):
        """Chi-square feature selection (fast)"""
        # Only for non-negative features
        X_pos = X - X.min() + 1  # Make all values positive
        selector = SelectKBest(score_func=chi2, k=k)
        X_selected = selector.fit_transform(X_pos, y)
        
        feature_names = X.columns[selector.get_support()].tolist() if hasattr(X, 'columns') else list(range(k))
        scores = dict(zip(feature_names, selector.scores_[selector.get_support()]))
        
        self.selectors['chi2'] = selector
        self.logger.info(f"Chi-square selection: {len(feature_names)} features selected")
        
        return pd.DataFrame(X_selected, columns=[f'chi2_{i}' for i in range(X_selected.shape[1])]), feature_names, scores
    
    def statistical_f_selection(self, X, y, k=20):
        """F-test feature selection (fast)"""
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        feature_names = X.columns[selector.get_support()].tolist() if hasattr(X, 'columns') else list(range(k))
        scores = dict(zip(feature_names, selector.scores_[selector.get_support()]))
        
        self.selectors['f_test'] = selector
        self.logger.info(f"F-test selection: {len(feature_names)} features selected")
        
        return pd.DataFrame(X_selected, columns=[f'ftest_{i}' for i in range(X_selected.shape[1])]), feature_names, scores
    
    def rf_importance_selection(self, X, y, k=20):
        """Random Forest importance selection (fast)"""
        # Use small RF for speed
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        selector = SelectFromModel(rf, max_features=k, prefit=True)
        X_selected = selector.transform(X)
        
        feature_names = X.columns[selector.get_support()].tolist() if hasattr(X, 'columns') else list(range(k))
        scores = dict(zip(feature_names, rf.feature_importances_[selector.get_support()]))
        
        self.selectors['rf_importance'] = selector
        self.logger.info(f"RF importance selection: {len(feature_names)} features selected")
        
        return pd.DataFrame(X_selected, columns=[f'rf_{i}' for i in range(X_selected.shape[1])]), feature_names, scores
    
    def combined_selection(self, X, y, k=20):
        """Combined feature selection (fast)"""
        # Use results from previous methods if available
        if not hasattr(self, 'all_scores'):
            # Quick RF for combined scoring
            rf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42)
            rf.fit(X, y)
            
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
                importance_scores = dict(zip(feature_names, rf.feature_importances_))
            else:
                feature_names = list(range(X.shape[1]))
                importance_scores = dict(zip(feature_names, rf.feature_importances_))
            
            # Select top k features
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat[0] for feat in sorted_features[:k]]
            selected_scores = dict(sorted_features[:k])
            
            if hasattr(X, 'columns'):
                X_selected = X[selected_features]
            else:
                feature_indices = selected_features
                X_selected = X[:, feature_indices]
        
        self.logger.info(f"Combined selection: {len(selected_features)} features selected")
        
        return X_selected, selected_features, selected_scores
    
    def select_features_all_methods(self, X, y):
        """Apply all 4 feature selection methods"""
        results = {}
        k = self.config['feature_selection']['n_features']
        
        # Method 1: Chi-square
        try:
            X_chi2, features_chi2, scores_chi2 = self.statistical_chi2_selection(X, y, k)
            results['chi2'] = {
                'data': X_chi2,
                'features': features_chi2,
                'scores': scores_chi2
            }
        except Exception as e:
            self.logger.error(f"Chi-square selection failed: {e}")
        
        # Method 2: F-test
        try:
            X_f, features_f, scores_f = self.statistical_f_selection(X, y, k)
            results['f_test'] = {
                'data': X_f,
                'features': features_f,
                'scores': scores_f
            }
        except Exception as e:
            self.logger.error(f"F-test selection failed: {e}")
        
        # Method 3: RF importance
        try:
            X_rf, features_rf, scores_rf = self.rf_importance_selection(X, y, k)
            results['rf_importance'] = {
                'data': X_rf,
                'features': features_rf,
                'scores': scores_rf
            }
        except Exception as e:
            self.logger.error(f"RF importance selection failed: {e}")
        
        # Method 4: Combined
        try:
            X_combined, features_combined, scores_combined = self.combined_selection(X, y, k)
            results['combined'] = {
                'data': X_combined,
                'features': features_combined,
                'scores': scores_combined
            }
        except Exception as e:
            self.logger.error(f"Combined selection failed: {e}")
        
        return results

class FastBaseModels:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.trained_models = {}
    
    def initialize_models(self, class_weights=None):
        """Initialize fast-training models with class weights for imbalance"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=500,
                class_weight='balanced',  # Automatically handles imbalance
                n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                class_weight='balanced',  # Automatically handles imbalance
                n_jobs=-1
            ),
            'naive_bayes': GaussianNB(),  # Handles imbalance through prior probabilities
            'decision_tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=15,
                class_weight='balanced'  # Automatically handles imbalance
            )
        }
        
        # Log class weight strategy
        self.logger.info("Initialized 4 fast-training base models:")
        self.logger.info("- Logistic Regression: class_weight='balanced'")
        self.logger.info("- Random Forest: class_weight='balanced'") 
        self.logger.info("- Naive Bayes: Automatic prior adjustment")
        self.logger.info("- Decision Tree: class_weight='balanced'")
        
        return self.models
    
    def train_models_with_imbalance_handling(self, X_train, y_train):
        """Train models with explicit class imbalance demonstration"""
        # First, analyze the imbalance
        imbalance_handler = ClassImbalanceHandler(self.config)
        imbalance_info = imbalance_handler.analyze_imbalance(y_train)
        
        print(f"\nTRAINING SET CLASS IMBALANCE:")
        for class_label, count in imbalance_info['class_counts'].items():
            percentage = imbalance_info['class_percentages'][class_label]
            print(f"Class {class_label}: {count} samples ({percentage:.1f}%)")
        
        if 'imbalance_ratio' in imbalance_info:
            print(f"Imbalance Ratio: {imbalance_info['imbalance_ratio']:.2f}:1")
        
        # Calculate and display class weights
        class_weights = imbalance_handler.calculate_class_weights(y_train)
        print(f"Applied Class Weights: {class_weights}")
        
        # Initialize models with class weights
        self.initialize_models(class_weights)
        
        predictions = {}
        confidence_scores = {}
        
        # Train each model and demonstrate effectiveness
        for model_name, model in self.models.items():
            try:
                start_time = datetime.now()
                model.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Get predictions and confidence scores
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X_train)
                    predictions[model_name] = pred_proba[:, 1]  # Positive class probability
                    confidence_scores[model_name] = np.max(pred_proba, axis=1)
                else:
                    predictions[model_name] = model.predict(X_train)
                    confidence_scores[model_name] = np.ones(len(X_train)) * 0.5
                
                self.trained_models[model_name] = model
                
                # Show class distribution in predictions for this model
                binary_preds = (predictions[model_name] > 0.5).astype(int)
                pred_counts = pd.Series(binary_preds).value_counts()
                pred_ratio = pred_counts[0] / pred_counts[1] if 1 in pred_counts else float('inf')
                
                self.logger.info(f"{model_name}: Trained in {training_time:.2f}s")
                self.logger.info(f"  Prediction distribution: {pred_counts.to_dict()}")
                self.logger.info(f"  Prediction ratio: {pred_ratio:.2f}:1 (vs original {imbalance_info.get('imbalance_ratio', 'N/A'):.2f}:1)")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
        
        return predictions, confidence_scores, imbalance_info

class TensorFlowMetaLearner:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        
        # Set TensorFlow to be less verbose
        tf.get_logger().setLevel('ERROR')
    
    def create_neural_network(self, input_dim):
        """Create optimized neural network for meta-learning"""
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def prepare_meta_features(self, base_predictions, confidence_scores, original_features=None):
        """Prepare meta-features including confidence scores"""
        # Combine base model predictions
        pred_df = pd.DataFrame(base_predictions)
        
        # Add confidence scores as features
        conf_df = pd.DataFrame(confidence_scores)
        conf_df.columns = [f'{col}_confidence' for col in conf_df.columns]
        
        # Combine predictions and confidence scores
        meta_features = pd.concat([pred_df, conf_df], axis=1)
        
        # Optionally include original features
        if original_features is not None:
            meta_features = pd.concat([original_features.reset_index(drop=True), 
                                     meta_features.reset_index(drop=True)], axis=1)
        
        return meta_features
    
    def train_meta_learner(self, base_predictions, confidence_scores, y_train, original_features=None):
        """Train the meta-learning neural network"""
        # Prepare meta-features
        meta_features = self.prepare_meta_features(base_predictions, confidence_scores, original_features)
        
        # Scale features
        X_meta_scaled = self.scaler.fit_transform(meta_features)
        
        # Create and train neural network
        input_dim = X_meta_scaled.shape[1]
        self.model = self.create_neural_network(input_dim)
        
        # Training callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train with validation split
        history = self.model.fit(
            X_meta_scaled, y_train,
            epochs=100,  # Reduced epochs for speed
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0  # Silent training
        )
        
        final_val_loss = min(history.history['val_loss'])
        self.logger.info(f"Meta-learner trained. Final validation loss: {final_val_loss:.4f}")
        
        return self.model


class FastFeatureSelector:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.selectors = {}
    
    def statistical_chi2_selection(self, X, y, k=20):
        """Chi-square feature selection (fast)"""
        # Only for non-negative features
        X_pos = X - X.min() + 1  # Make all values positive
        selector = SelectKBest(score_func=chi2, k=k)
        X_selected = selector.fit_transform(X_pos, y)
        
        feature_names = X.columns[selector.get_support()].tolist() if hasattr(X, 'columns') else list(range(k))
        scores = dict(zip(feature_names, selector.scores_[selector.get_support()]))
        
        self.selectors['chi2'] = selector
        self.logger.info(f"Chi-square selection: {len(feature_names)} features selected")
        
        return pd.DataFrame(X_selected, columns=[f'chi2_{i}' for i in range(X_selected.shape[1])]), feature_names, scores
    
    def statistical_f_selection(self, X, y, k=20):
        """F-test feature selection (fast)"""
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        feature_names = X.columns[selector.get_support()].tolist() if hasattr(X, 'columns') else list(range(k))
        scores = dict(zip(feature_names, selector.scores_[selector.get_support()]))
        
        self.selectors['f_test'] = selector
        self.logger.info(f"F-test selection: {len(feature_names)} features selected")
        
        return pd.DataFrame(X_selected, columns=[f'ftest_{i}' for i in range(X_selected.shape[1])]), feature_names, scores
    
    def rf_importance_selection(self, X, y, k=20):
        """Random Forest importance selection (fast)"""
        # Use small RF for speed
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        selector = SelectFromModel(rf, max_features=k, prefit=True)
        X_selected = selector.transform(X)
        
        feature_names = X.columns[selector.get_support()].tolist() if hasattr(X, 'columns') else list(range(k))
        scores = dict(zip(feature_names, rf.feature_importances_[selector.get_support()]))
        
        self.selectors['rf_importance'] = selector
        self.logger.info(f"RF importance selection: {len(feature_names)} features selected")
        
        return pd.DataFrame(X_selected, columns=[f'rf_{i}' for i in range(X_selected.shape[1])]), feature_names, scores
    
    def combined_selection(self, X, y, k=20):
        """Combined feature selection (fast)"""
        # Use results from previous methods if available
        if not hasattr(self, 'all_scores'):
            # Quick RF for combined scoring
            rf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42)
            rf.fit(X, y)
            
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
                importance_scores = dict(zip(feature_names, rf.feature_importances_))
            else:
                feature_names = list(range(X.shape[1]))
                importance_scores = dict(zip(feature_names, rf.feature_importances_))
            
            # Select top k features
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat[0] for feat in sorted_features[:k]]
            selected_scores = dict(sorted_features[:k])
            
            if hasattr(X, 'columns'):
                X_selected = X[selected_features]
            else:
                feature_indices = selected_features
                X_selected = X[:, feature_indices]
        
        self.logger.info(f"Combined selection: {len(selected_features)} features selected")
        
        return X_selected, selected_features, selected_scores
    
    def select_features_all_methods(self, X, y):
        """Apply all 4 feature selection methods"""
        results = {}
        k = self.config['feature_selection']['n_features']
        
        # Method 1: Chi-square
        try:
            X_chi2, features_chi2, scores_chi2 = self.statistical_chi2_selection(X, y, k)
            results['chi2'] = {
                'data': X_chi2,
                'features': features_chi2,
                'scores': scores_chi2
            }
        except Exception as e:
            self.logger.error(f"Chi-square selection failed: {e}")
        
        # Method 2: F-test
        try:
            X_f, features_f, scores_f = self.statistical_f_selection(X, y, k)
            results['f_test'] = {
                'data': X_f,
                'features': features_f,
                'scores': scores_f
            }
        except Exception as e:
            self.logger.error(f"F-test selection failed: {e}")
        
        # Method 3: RF importance
        try:
            X_rf, features_rf, scores_rf = self.rf_importance_selection(X, y, k)
            results['rf_importance'] = {
                'data': X_rf,
                'features': features_rf,
                'scores': scores_rf
            }
        except Exception as e:
            self.logger.error(f"RF importance selection failed: {e}")
        
        # Method 4: Combined
        try:
            X_combined, features_combined, scores_combined = self.combined_selection(X, y, k)
            results['combined'] = {
                'data': X_combined,
                'features': features_combined,
                'scores': scores_combined
            }
        except Exception as e:
            self.logger.error(f"Combined selection failed: {e}")
        
        return results

def main():
    """Main execution function"""
    print("=" * 60)
    print("OPTIMIZED DIABETES READMISSION PREDICTION PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = OptimizedDiabetesReadmissionPipeline()
    processor = DataProcessor(pipeline.config)
    feature_selector = FastFeatureSelector(pipeline.config)
    base_models = FastBaseModels(pipeline.config)
    
    try:
        # Step 1: Load and preprocess data
        print("\nStep 1: Loading and preprocessing data...")
        df_raw = processor.load_data(pipeline.config['data']['raw_path'])
        
        # Handle missing values
        df_clean = processor.handle_missing_values(df_raw)
        
        # Encode categorical variables
        df_encoded, label_encoders = processor.encode_categorical_variables(df_clean)
        
        # Process target
        df_processed = processor.preprocess_target(df_encoded)
        
        # Prepare features and target
        target_col = 'readmitted_binary'
        exclude_cols = ['readmitted', 'readmitted_binary', 'encounter_id', 'patient_nbr']
        
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        X = df_processed[feature_cols].select_dtypes(include=[np.number])  # Only numerical features
        y = df_processed[target_col]
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Analyze class imbalance
        class_counts = y.value_counts()
        imbalance_ratio = class_counts[0] / class_counts[1] if 1 in class_counts else float('inf')
        minority_percentage = (class_counts[1] / len(y) * 100) if 1 in class_counts else 0
        
        print(f"\nCLASS IMBALANCE ANALYSIS:")
        print(f"No Readmission (0): {class_counts[0]} samples ({100-minority_percentage:.1f}%)")
        if 1 in class_counts:
            print(f"Readmission (1): {class_counts[1]} samples ({minority_percentage:.1f}%)")
            print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        else:
            print("WARNING: No positive class samples found!")
        
        if imbalance_ratio > 2:
            print(f"SEVERE IMBALANCE DETECTED - Class weights will be applied automatically")
            
            # Demonstrate class weight effectiveness with a quick test
            print(f"\nDEMONSTRATING CLASS WEIGHTS EFFECTIVENESS:")
            X_sample = X.sample(min(1000, len(X)), random_state=42)
            y_sample = y.loc[X_sample.index]
            
            # Quick test: train simple model with and without class weights
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
            
            X_test_train, X_test_val, y_test_train, y_test_val = train_test_split(
                X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
            )
            
            # Without class weights
            model_no_weights = LogisticRegression(random_state=42, max_iter=200)
            model_no_weights.fit(X_test_train, y_test_train)
            pred_no_weights = model_no_weights.predict(X_test_val)
            
            # With class weights  
            model_with_weights = LogisticRegression(random_state=42, max_iter=200, class_weight='balanced')
            model_with_weights.fit(X_test_train, y_test_train)
            pred_with_weights = model_with_weights.predict(X_test_val)
            
            print(f"\nSample Test Results (n={len(y_test_val)}):")
            print(f"WITHOUT class weights - Minority class predictions: {sum(pred_no_weights)}/{len(pred_no_weights)} ({sum(pred_no_weights)/len(pred_no_weights)*100:.1f}%)")
            print(f"WITH class weights - Minority class predictions: {sum(pred_with_weights)}/{len(pred_with_weights)} ({sum(pred_with_weights)/len(pred_with_weights)*100:.1f}%)")
            print(f"Actual minority class: {sum(y_test_val)}/{len(y_test_val)} ({sum(y_test_val)/len(y_test_val)*100:.1f}%)")
        
        # Step 2: Feature selection
        print("\nStep 2: Running feature selection methods...")
        feature_results = feature_selector.select_features_all_methods(X, y)
        
        # Save datasets
        os.makedirs(pipeline.config['data']['processed_path'], exist_ok=True)
        
        # Save cleaned dataset
        cleaned_data = pd.concat([X, y], axis=1)
        cleaned_path = os.path.join(pipeline.config['data']['processed_path'], 'cleaned_data.csv')
        cleaned_data.to_csv(cleaned_path, index=False)
        print(f"Cleaned dataset saved: {cleaned_path}")
        
        # Save feature-selected datasets
        os.makedirs(os.path.join(pipeline.config['data']['processed_path'], 'feature_selected'), exist_ok=True)
        
        dataset_info = {}
        dataset_info['cleaned'] = {
            'path': cleaned_path,
            'shape': cleaned_data.shape,
            'features': X.columns.tolist()
        }
        
        for method_name, results in feature_results.items():
            if 'data' in results:
                # Add target to feature-selected data
                selected_data = pd.concat([results['data'], y.reset_index(drop=True)], axis=1)
                
                save_path = os.path.join(
                    pipeline.config['data']['processed_path'], 
                    'feature_selected', 
                    f'{method_name}_selected.csv'
                )
                selected_data.to_csv(save_path, index=False)
                
                dataset_info[method_name] = {
                    'path': save_path,
                    'shape': selected_data.shape,
                    'features': results['features'],
                    'scores': results['scores']
                }
                
                print(f"{method_name} dataset saved: {save_path}")
                print(f"  Shape: {selected_data.shape}")
                print(f"  Top 5 features: {results['features'][:5]}")
        
        # Summary report
        summary_path = os.path.join(pipeline.config['data']['processed_path'], 'dataset_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("DIABETES READMISSION PREDICTION - DATASET SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            for dataset_name, info in dataset_info.items():
                f.write(f"Dataset: {dataset_name.upper()}\n")
                f.write(f"Path: {info['path']}\n")
                f.write(f"Shape: {info['shape']}\n")
                if 'features' in info:
                    f.write(f"Number of features: {len(info['features'])}\n")
                    if len(info['features']) <= 20:
                        f.write(f"Features: {info['features']}\n")
                f.write("\n")
        
        print(f"\nDataset summary saved: {summary_path}")
        
        print("\n" + "=" * 60)
        print("FEATURE SELECTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nGenerated datasets:")
        print(f"1. Cleaned dataset: {cleaned_data.shape[0]} samples, {cleaned_data.shape[1]-1} features")
        
        for i, (method_name, results) in enumerate(feature_results.items(), 2):
            if 'data' in results:
                print(f"{i}. {method_name}: {results['data'].shape[0]} samples, {results['data'].shape[1]} features")
        
        print(f"\nNext steps:")
        print(f"1. Use these datasets for base model training")
        print(f"2. Implement meta-learning with confidence scores")
        print(f"3. Evaluate performance across 3 train-test splits")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()