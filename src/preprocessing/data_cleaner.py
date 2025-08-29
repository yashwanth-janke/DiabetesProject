import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
import logging

class DataCleaner:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path):
        """Load the diabetes dataset"""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df):
        """Handle missing values with multiple strategies"""
        df_clean = df.copy()
        
        # Identify columns with high missing percentage
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        high_missing_cols = missing_pct[missing_pct > self.config['preprocessing']['missing_threshold']].index
        
        self.logger.info(f"Dropping columns with >{self.config['preprocessing']['missing_threshold']*100}% missing: {list(high_missing_cols)}")
        df_clean = df_clean.drop(columns=high_missing_cols)
        
        # Handle remaining missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Numeric: KNN imputation
        if len(numeric_cols) > 0:
            imputer_numeric = KNNImputer(n_neighbors=5)
            df_clean[numeric_cols] = imputer_numeric.fit_transform(df_clean[numeric_cols])
        
        # Categorical: Mode imputation
        if len(categorical_cols) > 0:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            df_clean[categorical_cols] = imputer_categorical.fit_transform(df_clean[categorical_cols])
        
        return df_clean
    
    def encode_categorical_variables(self, df):
        """Encode categorical variables with domain-specific logic"""
        df_encoded = df.copy()
        
        # ICD-9 code grouping for diagnoses
        df_encoded = self._group_icd9_codes(df_encoded)
        
        # Medication encoding
        df_encoded = self._encode_medications(df_encoded)
        
        # HbA1c results encoding
        df_encoded = self._encode_hba1c(df_encoded)
        
        # General categorical encoding
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop(['readmitted'], errors='ignore')  # Exclude target
        
        if self.config['preprocessing']['categorical_encoding'] == 'label':
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le
        
        return df_encoded, label_encoders
    
    def _group_icd9_codes(self, df):
        """Group ICD-9 codes into meaningful categories"""
        # Define ICD-9 groupings based on domain knowledge
        icd9_groups = {
            'circulatory': ['390-459', '785'],
            'respiratory': ['460-519', '786'], 
            'digestive': ['520-579', '787'],
            'diabetes': ['250'],
            'injury': ['800-999'],
            'musculoskeletal': ['710-739'],
            'genitourinary': ['580-629', '788'],
            'neoplasms': ['140-239'],
            'other': []
        }
        
        for diag_col in ['diag_1', 'diag_2', 'diag_3']:
            if diag_col in df.columns:
                df[f'{diag_col}_group'] = df[diag_col].apply(self._categorize_icd9)
        
        return df
    
    def _categorize_icd9(self, code):
        """Categorize individual ICD-9 code"""
        if pd.isna(code):
            return 'missing'
        
        code_str = str(code)
        
        # Diabetes
        if code_str.startswith('250'):
            return 'diabetes'
        # Circulatory
        elif any(code_str.startswith(str(i)) for i in range(390, 460)) or code_str.startswith('785'):
            return 'circulatory'
        # Respiratory  
        elif any(code_str.startswith(str(i)) for i in range(460, 520)) or code_str.startswith('786'):
            return 'respiratory'
        # Add other categories...
        else:
            return 'other'
    
    def _encode_medications(self, df):
        """Encode medication columns"""
        med_cols = [col for col in df.columns if col in [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide',
            'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]]
        
        for col in med_cols:
            # Convert to binary: changed/not changed
            df[f'{col}_binary'] = df[col].apply(
                lambda x: 1 if x in ['Up', 'Down', 'Steady'] else 0
            )
        
        return df
    
    def _encode_hba1c(self, df):
        """Encode HbA1c results"""
        if 'A1Cresult' in df.columns:
            hba1c_mapping = {
                'None': 0,
                'Norm': 1, 
                '>7': 2,
                '>8': 3
            }
            df['A1Cresult_encoded'] = df['A1Cresult'].map(hba1c_mapping)
        
        return df
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, scaler
        
        return X_train_scaled, scaler
    
    def handle_class_imbalance(self, X, y, method='SMOTE'):
        """Handle class imbalance using various techniques"""
        if method == 'SMOTE':
            sampler = SMOTE(random_state=42)
        elif method == 'SMOTE-ENN':
            sampler = SMOTEENN(random_state=42)
        elif method == 'BorderlineSMOTE':
            sampler = BorderlineSMOTE(random_state=42)
        else:
            return X, y  # Return original if method not recognized
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        self.logger.info(f"Class distribution after {method}: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def preprocess_target(self, df):
        """Preprocess target variable"""
        # Convert readmitted to binary: <30 days = 1, else = 0
        target_mapping = {
            '<30': 1,  # Readmitted within 30 days
            '>30': 0,  # Readmitted after 30 days
            'NO': 0    # Not readmitted
        }
        
        df['readmitted_binary'] = df['readmitted'].map(target_mapping)
        return df