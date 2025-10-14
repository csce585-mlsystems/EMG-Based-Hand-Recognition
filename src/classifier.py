import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Dict, Tuple, Optional, Any
import json
import os
from datetime import datetime

class GestureClassifier:
    """
    Production-ready gesture classifier with proper model management.
    """
    
    def __init__(self, classifier_type: str = 'svm'):
        """
        Initialize classifier.
        
        Args:
            classifier_type: Type of classifier ('svm', 'rf', 'lda')
        """
        self.classifier_type = classifier_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.metadata = {}
        
        # Store training data for potential retraining
        self.X_train = None
        self.y_train = None
        
    def create_model(self, **kwargs):
        """Create classifier model based on type."""
        if self.classifier_type == 'svm':
            self.model = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 10.0),
                gamma=kwargs.get('gamma', 'scale'),
                probability=True,  # Always enable probability
                random_state=42
            )
        elif self.classifier_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=42
            )
        elif self.classifier_type == 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            self.model = LinearDiscriminantAnalysis()
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
            
    def train(self, X: np.ndarray, y: np.ndarray, 
             cv_folds: int = 5, **model_params) -> Dict[str, Any]:
        """
        Train classifier with cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            cv_folds: Number of CV folds
            **model_params: Model-specific parameters
            
        Returns:
            Training results dictionary
        """
        # Store training data
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Preprocess
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create model
        self.create_model(**model_params)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        
        # Final training
        self.model.fit(X_scaled, y_encoded)
        self.is_trained = True
        
        # Store metadata
        self.metadata = {
            'classifier_type': self.classifier_type,
            'model_params': model_params,
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'classes': list(self.label_encoder.classes_),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X)
        }
        
        results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'metadata': self.metadata
        }
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict gesture labels."""
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")
            
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict_single(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict single sample with confidence.
        
        Args:
            features: 1D feature vector
            
        Returns:
            (predicted_label, confidence)
        """
        features = features.reshape(1, -1)
        label = self.predict(features)[0]
        proba = self.predict_proba(features)[0]
        confidence = np.max(proba)
        
        return label, confidence
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate classifier performance."""
        y_pred = self.predict(X)
        
        results = {
            'accuracy': accuracy_score(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'classes': list(self.label_encoder.classes_)
        }
        
        return results
    
    def save_model(self, filepath: str):
        """Save complete model package."""
        base_path = os.path.splitext(filepath)[0]
        
        # Save model components
        joblib.dump(self.model, f"{base_path}_model.joblib")
        joblib.dump(self.scaler, f"{base_path}_scaler.joblib")
        joblib.dump(self.label_encoder, f"{base_path}_encoder.joblib")
        
        # Save metadata
        with open(f"{base_path}_metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        print(f"Model saved to {base_path}_*")
    
    def load_model(self, filepath: str):
        """Load complete model package."""
        base_path = os.path.splitext(filepath)[0]
        
        # Load model components
        self.model = joblib.load(f"{base_path}_model.joblib")
        self.scaler = joblib.load(f"{base_path}_scaler.joblib")
        self.label_encoder = joblib.load(f"{base_path}_encoder.joblib")
        
        # Load metadata
        with open(f"{base_path}_metadata.json", 'r') as f:
            self.metadata = json.load(f)
            
        self.is_trained = True
        self.classifier_type = self.metadata.get('classifier_type', 'svm')
        
        print(f"Model loaded from {base_path}_*")