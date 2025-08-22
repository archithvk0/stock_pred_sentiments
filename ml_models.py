"""
Machine Learning Models for Stock Price Prediction
Includes baseline models, gradient boosting, and deep learning approaches
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import pickle
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates model performance with various metrics"""
    
    def __init__(self):
        pass
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        return metrics
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Print detailed classification report"""
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

class BaselineModels:
    """Baseline machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.evaluator = ModelEvaluator()
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray, 
                                X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train logistic regression model"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        train_pred_proba = model.predict_proba(X_train_scaled)
        train_metrics = self.evaluator.evaluate_model(y_train, train_pred, train_pred_proba)
        
        results = {
            'model': model,
            'scaler': scaler,
            'train_metrics': train_metrics
        }
        
        if X_val is not None:
            val_pred = model.predict(X_val_scaled)
            val_pred_proba = model.predict_proba(X_val_scaled)
            val_metrics = self.evaluator.evaluate_model(y_val, val_pred, val_pred_proba)
            results['val_metrics'] = val_metrics
        
        self.models['logistic_regression'] = model
        self.scalers['logistic_regression'] = scaler
        
        return results
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train random forest model"""
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        train_pred_proba = model.predict_proba(X_train)
        train_metrics = self.evaluator.evaluate_model(y_train, train_pred, train_pred_proba)
        
        results = {
            'model': model,
            'train_metrics': train_metrics
        }
        
        if X_val is not None:
            val_pred = model.predict(X_val)
            val_pred_proba = model.predict_proba(X_val)
            val_metrics = self.evaluator.evaluate_model(y_val, val_pred, val_pred_proba)
            results['val_metrics'] = val_metrics
        
        self.models['random_forest'] = model
        
        return results
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train SVM model"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
        
        # Train model
        model = SVC(probability=True, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train_scaled)
        train_pred_proba = model.predict_proba(X_train_scaled)
        train_metrics = self.evaluator.evaluate_model(y_train, train_pred, train_pred_proba)
        
        results = {
            'model': model,
            'scaler': scaler,
            'train_metrics': train_metrics
        }
        
        if X_val is not None:
            val_pred = model.predict(X_val_scaled)
            val_pred_proba = model.predict_proba(X_val_scaled)
            val_metrics = self.evaluator.evaluate_model(y_val, val_pred, val_pred_proba)
            results['val_metrics'] = val_metrics
        
        self.models['svm'] = model
        self.scalers['svm'] = scaler
        
        return results

class GradientBoostingModels:
    """Gradient boosting models (XGBoost, LightGBM)"""
    
    def __init__(self):
        self.models = {}
        self.evaluator = ModelEvaluator()
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train XGBoost model"""
        # Prepare data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(dtrain, 'train'), (dval, 'val')]
        else:
            watchlist = [(dtrain, 'train')]
        
        # Parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=watchlist,
            early_stopping_rounds=10 if X_val is not None else None,
            verbose_eval=False
        )
        
        # Evaluate
        train_pred_proba = model.predict(dtrain)
        train_pred = (train_pred_proba > 0.5).astype(int)
        train_metrics = self.evaluator.evaluate_model(y_train, train_pred, 
                                                    np.column_stack([1-train_pred_proba, train_pred_proba]))
        
        results = {
            'model': model,
            'train_metrics': train_metrics
        }
        
        if X_val is not None:
            val_pred_proba = model.predict(dval)
            val_pred = (val_pred_proba > 0.5).astype(int)
            val_metrics = self.evaluator.evaluate_model(y_val, val_pred,
                                                      np.column_stack([1-val_pred_proba, val_pred_proba]))
            results['val_metrics'] = val_metrics
        
        self.models['xgboost'] = model
        
        return results
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train LightGBM model"""
        # Prepare data
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'random_state': 42
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data] if X_val is not None else None,
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10) if X_val is not None else None],
            verbose_eval=False
        )
        
        # Evaluate
        train_pred_proba = model.predict(X_train)
        train_pred = (train_pred_proba > 0.5).astype(int)
        train_metrics = self.evaluator.evaluate_model(y_train, train_pred,
                                                    np.column_stack([1-train_pred_proba, train_pred_proba]))
        
        results = {
            'model': model,
            'train_metrics': train_metrics
        }
        
        if X_val is not None:
            val_pred_proba = model.predict(X_val)
            val_pred = (val_pred_proba > 0.5).astype(int)
            val_metrics = self.evaluator.evaluate_model(y_val, val_pred,
                                                      np.column_stack([1-val_pred_proba, val_pred_proba]))
            results['val_metrics'] = val_metrics
        
        self.models['lightgbm'] = model
        
        return results

class DeepLearningModels:
    """Deep learning models (LSTM, Dense Neural Networks)"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.evaluator = ModelEvaluator()
    
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Create LSTM model architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_dense_model(self, input_dim: int) -> tf.keras.Model:
        """Create dense neural network model"""
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_sequences(self, data: np.ndarray, target: np.ndarray, 
                         sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM model"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray = None, y_val: np.ndarray = None,
                  sequence_length: int = 10) -> Dict[str, Any]:
        """Train LSTM model"""
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train, sequence_length)
        
        if X_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val, sequence_length)
        
        # Create and train model
        model = self.create_lstm_model((sequence_length, X_train.shape[1]))
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq) if X_val is not None else None,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        train_pred_proba = model.predict(X_train_seq)
        train_pred = (train_pred_proba > 0.5).astype(int)
        train_metrics = self.evaluator.evaluate_model(y_train_seq, train_pred.flatten(),
                                                    np.column_stack([1-train_pred_proba.flatten(), train_pred_proba.flatten()]))
        
        results = {
            'model': model,
            'scaler': scaler,
            'history': history,
            'train_metrics': train_metrics,
            'sequence_length': sequence_length
        }
        
        if X_val is not None:
            val_pred_proba = model.predict(X_val_seq)
            val_pred = (val_pred_proba > 0.5).astype(int)
            val_metrics = self.evaluator.evaluate_model(y_val_seq, val_pred.flatten(),
                                                      np.column_stack([1-val_pred_proba.flatten(), val_pred_proba.flatten()]))
            results['val_metrics'] = val_metrics
        
        self.models['lstm'] = model
        self.scalers['lstm'] = scaler
        
        return results
    
    def train_dense_nn(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train dense neural network"""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
        
        # Create and train model
        model = self.create_dense_model(X_train.shape[1])
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val) if X_val is not None else None,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        train_pred_proba = model.predict(X_train_scaled)
        train_pred = (train_pred_proba > 0.5).astype(int)
        train_metrics = self.evaluator.evaluate_model(y_train, train_pred.flatten(),
                                                    np.column_stack([1-train_pred_proba.flatten(), train_pred_proba.flatten()]))
        
        results = {
            'model': model,
            'scaler': scaler,
            'history': history,
            'train_metrics': train_metrics
        }
        
        if X_val is not None:
            val_pred_proba = model.predict(X_val_scaled)
            val_pred = (val_pred_proba > 0.5).astype(int)
            val_metrics = self.evaluator.evaluate_model(y_val, val_pred.flatten(),
                                                      np.column_stack([1-val_pred_proba.flatten(), val_pred_proba.flatten()]))
            results['val_metrics'] = val_metrics
        
        self.models['dense_nn'] = model
        self.scalers['dense_nn'] = scaler
        
        return results

class ModelEnsemble:
    """Ensemble of multiple models"""
    
    def __init__(self):
        self.models = {}
        self.evaluator = ModelEvaluator()
    
    def add_model(self, name: str, model: Any, scaler: Any = None):
        """Add a model to the ensemble"""
        self.models[name] = {
            'model': model,
            'scaler': scaler
        }
    
    def predict_ensemble(self, X: np.ndarray, method: str = 'voting') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions
        
        Args:
            X: Input features
            method: 'voting' for majority vote, 'average' for average probabilities
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        predictions = []
        probabilities = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            scaler = model_info['scaler']
            
            # Scale features if scaler exists
            X_scaled = scaler.transform(X) if scaler is not None else X
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
                pred = model.predict(X_scaled)
            elif hasattr(model, 'predict'):
                # For XGBoost and LightGBM
                if name == 'xgboost':
                    dtest = xgb.DMatrix(X_scaled)
                    proba = model.predict(dtest)
                    pred = (proba > 0.5).astype(int)
                    proba = np.column_stack([1-proba, proba])
                elif name == 'lightgbm':
                    proba = model.predict(X_scaled)
                    pred = (proba > 0.5).astype(int)
                    proba = np.column_stack([1-proba, proba])
                else:
                    proba = model.predict(X_scaled)
                    pred = (proba > 0.5).astype(int)
                    proba = np.column_stack([1-proba, proba])
            else:
                continue
            
            predictions.append(pred)
            probabilities.append(proba[:, 1])  # Positive class probability
        
        if not predictions:
            return np.array([]), np.array([])
        
        # Combine predictions
        if method == 'voting':
            final_pred = np.mean(predictions, axis=0) > 0.5
        else:  # average
            final_pred = np.mean(probabilities, axis=0) > 0.5
        
        final_proba = np.mean(probabilities, axis=0)
        
        return final_pred.astype(int), np.column_stack([1-final_proba, final_proba])

class ModelManager:
    """Main model management class"""
    
    def __init__(self):
        self.baseline_models = BaselineModels()
        self.gb_models = GradientBoostingModels()
        self.dl_models = DeepLearningModels()
        self.ensemble = ModelEnsemble()
        self.evaluator = ModelEvaluator()
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train all available models"""
        results = {}
        
        # Train baseline models
        logger.info("Training baseline models...")
        results['logistic_regression'] = self.baseline_models.train_logistic_regression(
            X_train, y_train, X_val, y_val
        )
        results['random_forest'] = self.baseline_models.train_random_forest(
            X_train, y_train, X_val, y_val
        )
        results['svm'] = self.baseline_models.train_svm(
            X_train, y_train, X_val, y_val
        )
        
        # Train gradient boosting models
        logger.info("Training gradient boosting models...")
        results['xgboost'] = self.gb_models.train_xgboost(
            X_train, y_train, X_val, y_val
        )
        results['lightgbm'] = self.gb_models.train_lightgbm(
            X_train, y_train, X_val, y_val
        )
        
        # Train deep learning models
        logger.info("Training deep learning models...")
        results['dense_nn'] = self.dl_models.train_dense_nn(
            X_train, y_train, X_val, y_val
        )
        
        # Create ensemble
        self._create_ensemble()
        
        return results
    
    def _create_ensemble(self):
        """Create ensemble from trained models"""
        # Add baseline models
        for name, model in self.baseline_models.models.items():
            scaler = self.baseline_models.scalers.get(name)
            self.ensemble.add_model(name, model, scaler)
        
        # Add gradient boosting models
        for name, model in self.gb_models.models.items():
            self.ensemble.add_model(name, model)
        
        # Add deep learning models
        for name, model in self.dl_models.models.items():
            scaler = self.dl_models.scalers.get(name)
            self.ensemble.add_model(name, model, scaler)
    
    def save_models(self, filepath: str):
        """Save all models to disk"""
        models_to_save = {
            'baseline_models': self.baseline_models,
            'gb_models': self.gb_models,
            'dl_models': self.dl_models,
            'ensemble': self.ensemble
        }
        
        joblib.dump(models_to_save, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load models from disk"""
        models_loaded = joblib.load(filepath)
        
        self.baseline_models = models_loaded['baseline_models']
        self.gb_models = models_loaded['gb_models']
        self.dl_models = models_loaded['dl_models']
        self.ensemble = models_loaded['ensemble']
        
        logger.info(f"Models loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Test the ML models
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                             n_redundant=5, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    model_manager = ModelManager()
    results = model_manager.train_all_models(X_train, y_train, X_test, y_test)
    
    # Print results
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"Train Accuracy: {result['train_metrics']['accuracy']:.4f}")
        if 'val_metrics' in result:
            print(f"Test Accuracy: {result['val_metrics']['accuracy']:.4f}")
    
    # Test ensemble
    ensemble_pred, ensemble_proba = model_manager.ensemble.predict_ensemble(X_test)
    ensemble_metrics = model_manager.evaluator.evaluate_model(y_test, ensemble_pred, ensemble_proba)
    print(f"\nENSEMBLE:")
    print(f"Test Accuracy: {ensemble_metrics['accuracy']:.4f}") 