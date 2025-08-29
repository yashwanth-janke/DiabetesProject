import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging

class TensorFlowMetaLearner:
    def __init__(self, config, meta_model='neural_network'):
        self.config = config
        self.meta_model_type = meta_model
        self.meta_model = None
        self.base_models = {}
        self.scaler = None
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def create_neural_network_model(self, input_dim):
        """Create TensorFlow neural network model for meta-learning"""
        model = Sequential([
            # Input layer with batch normalization
            Dense(128, activation='relu', input_dim=input_dim, name='input_layer'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers with decreasing size
            Dense(64, activation='relu', name='hidden_layer_1'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu', name='hidden_layer_2'),
            Dropout(0.1),
            
            # Output layer for binary classification
            Dense(1, activation='sigmoid', name='output_layer')
        ])
        
        # Compile model with advanced metrics
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        self.logger.info(f"Created neural network with {input_dim} input features")
        self.logger.info(f"Model architecture: {model.summary()}")
        
        return model
    
    def create_meta_features(self, base_models, X_train, y_train, cv_folds=5):
        """Create meta-features using cross-validation predictions"""
        n_samples = X_train.shape[0]
        n_models = len(base_models)
        
        # Initialize meta-features array
        meta_features = np.zeros((n_samples, n_models))
        
        # Use stratified K-fold for better class distribution
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            self.logger.info(f"Processing fold {fold + 1}/{cv_folds}")
            
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            
            # Train base models on fold training data
            fold_predictions = {}
            for model_name, model in base_models.items():
                try:
                    # Clone and train model
                    model_clone = model.__class__(**model.get_params())
                    model_clone.fit(X_fold_train, y_fold_train)
                    
                    # Get predictions for validation set
                    pred_proba = model_clone.predict_proba(X_fold_val)[:, 1]
                    fold_predictions[model_name] = pred_proba
                    
                except Exception as e:
                    self.logger.error(f"Error with {model_name} in fold {fold}: {e}")
                    fold_predictions[model_name] = np.zeros(len(val_idx))
            
            # Store predictions in meta-features array
            for i, (model_name, pred) in enumerate(fold_predictions.items()):
                meta_features[val_idx, i] = pred
        
        # Create DataFrame with model names as columns
        model_names = list(base_models.keys())
        meta_df = pd.DataFrame(meta_features, columns=[f'meta_{name}' for name in model_names])
        
        return meta_df
    
    def initialize_meta_model(self, input_dim):
        """Initialize the meta-learning model"""
        if self.meta_model_type == 'neural_network':
            self.meta_model = self.create_neural_network_model(input_dim)
        elif self.meta_model_type == 'logistic_regression':
            self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.meta_model_type == 'random_forest':
            self.meta_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported meta-model type: {self.meta_model_type}")
    
    def train_stacking_ensemble(self, base_models, X_train, y_train, X_original_features=None):
        """Train complete stacking ensemble with TensorFlow neural network"""
        # Step 1: Create meta-features
        self.logger.info("Creating meta-features using cross-validation...")
        meta_features = self.create_meta_features(base_models, X_train, y_train)
        
        # Step 2: Combine original features with meta-features
        if X_original_features is not None:
            self.logger.info("Combining original features with meta-features...")
            combined_features = pd.concat([
                X_original_features.reset_index(drop=True), 
                meta_features.reset_index(drop=True)
            ], axis=1)
        else:
            combined_features = meta_features
        
        # Step 3: Scale features for neural network
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(combined_features)
        
        # Step 4: Initialize and train meta-model
        input_dim = X_scaled.shape[1]
        self.initialize_meta_model(input_dim)
        
        if self.meta_model_type == 'neural_network':
            # TensorFlow training with callbacks
            self.logger.info("Training TensorFlow neural network meta-learner...")
            
            # Callbacks for better training
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
            
            # Train the model
            history = self.meta_model.fit(
                X_scaled, y_train,
                epochs=200,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Log training history
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            final_auc = history.history['auc'][-1]
            final_val_auc = history.history['val_auc'][-1]
            
            self.logger.info(f"Training completed - Loss: {final_loss:.4f}, Val Loss: {final_val_loss:.4f}")
            self.logger.info(f"Final AUC: {final_auc:.4f}, Val AUC: {final_val_auc:.4f}")
            
            # Store training history for analysis
            self.training_history = history
            
        else:
            # Traditional sklearn model training
            self.logger.info(f"Training {self.meta_model_type} meta-learner...")
            self.meta_model.fit(X_scaled, y_train)
        
        # Store base models for prediction
        self.base_models = base_models
        
        self.logger.info("Stacking ensemble training completed")
        return self.meta_model
    
    def predict(self, X_test, X_original_test=None):
        """Make predictions using stacking ensemble"""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call train_stacking_ensemble first.")
        
        # Get base model predictions
        base_predictions = []
        model_names = list(self.base_models.keys())
        
        for model_name, model in self.base_models.items():
            try:
                pred_proba = model.predict_proba(X_test)[:, 1]
                base_predictions.append(pred_proba)
            except Exception as e:
                self.logger.error(f"Error getting predictions from {model_name}: {e}")
                base_predictions.append(np.zeros(X_test.shape[0]))
        
        # Create meta-features
        meta_features = pd.DataFrame(
            np.column_stack(base_predictions),
            columns=[f'meta_{name}' for name in model_names]
        )
        
        # Combine with original features if used during training
        if X_original_test is not None:
            combined_features = pd.concat([
                X_original_test.reset_index(drop=True), 
                meta_features.reset_index(drop=True)
            ], axis=1)
        else:
            combined_features = meta_features
        
        # Scale features using the same scaler from training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(combined_features)
        else:
            X_scaled = combined_features.values
        
        # Get final predictions
        if self.meta_model_type == 'neural_network':
            # TensorFlow model returns probabilities directly
            final_predictions = self.meta_model.predict(X_scaled).flatten()
        else:
            # Sklearn models
            final_predictions = self.meta_model.predict_proba(X_scaled)[:, 1]
        
        return final_predictions
    
    def predict_with_uncertainty(self, X_test, X_original_test=None, n_samples=100):
        """Make predictions with uncertainty estimation using Monte Carlo Dropout"""
        if self.meta_model_type != 'neural_network':
            raise ValueError("Uncertainty estimation only available for neural network meta-learner")
        
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call train_stacking_ensemble first.")
        
        # Prepare features same as in predict method
        base_predictions = []
        model_names = list(self.base_models.keys())
        
        for model_name, model in self.base_models.items():
            try:
                pred_proba = model.predict_proba(X_test)[:, 1]
                base_predictions.append(pred_proba)
            except Exception as e:
                self.logger.error(f"Error getting predictions from {model_name}: {e}")
                base_predictions.append(np.zeros(X_test.shape[0]))
        
        meta_features = pd.DataFrame(
            np.column_stack(base_predictions),
            columns=[f'meta_{name}' for name in model_names]
        )
        
        if X_original_test is not None:
            combined_features = pd.concat([
                X_original_test.reset_index(drop=True), 
                meta_features.reset_index(drop=True)
            ], axis=1)
        else:
            combined_features = meta_features
        
        X_scaled = self.scaler.transform(combined_features)
        
        # Monte Carlo sampling with dropout enabled
        predictions = []
        for _ in range(n_samples):
            # Enable training mode to keep dropout active
            pred = self.meta_model(X_scaled, training=True)
            predictions.append(pred.numpy().flatten())
        
        predictions = np.array(predictions)
        
        # Calculate mean prediction and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    def get_feature_importance(self):
        """Get feature importance from meta-model if available"""
        if self.meta_model_type == 'neural_network':
            # For neural networks, we can use gradient-based importance
            self.logger.warning("Feature importance for neural networks requires gradient analysis")
            return None
        elif hasattr(self.meta_model, 'feature_importances_'):
            return self.meta_model.feature_importances_
        elif hasattr(self.meta_model, 'coef_'):
            return np.abs(self.meta_model.coef_[0])
        else:
            return None
    
    def save_model(self, filepath):
        """Save the trained meta-learner model"""
        if self.meta_model_type == 'neural_network':
            self.meta_model.save(f"{filepath}_neural_network.h5")
            # Also save the scaler
            import joblib
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        else:
            import joblib
            joblib.dump(self.meta_model, f"{filepath}_{self.meta_model_type}.pkl")
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        self.logger.info(f"Meta-learner model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained meta-learner model"""
        if self.meta_model_type == 'neural_network':
            self.meta_model = tf.keras.models.load_model(f"{filepath}_neural_network.h5")
            import joblib
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        else:
            import joblib
            self.meta_model = joblib.load(f"{filepath}_{self.meta_model_type}.pkl")
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        
        self.logger.info(f"Meta-learner model loaded from {filepath}")

# Alias for backward compatibility
StackingMetaLearner = TensorFlowMetaLearner