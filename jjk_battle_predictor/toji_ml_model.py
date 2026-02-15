"""
Toji Zenin Battle Predictor - ML Model
Predicts outcomes of Toji vs Cursed Spirits battles
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class TojiBattlePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def prepare_features(self, df):
        """Extract features for ML model"""
        feature_cols = [
            'toji_speed', 'toji_physical', 'toji_combat_iq', 'toji_stealth',
            'toji_weapon_mastery', 'toji_ce_resistance', 'toji_durability',
            'opp_physical', 'opp_speed', 'opp_durability', 'opp_cursed_energy',
            'opp_technique_power', 'opp_intelligence', 'opp_domain',
            'opp_regeneration', 'opp_size',
            'speed_advantage', 'physical_advantage', 'iq_advantage',
            'domain_nullification', 'instant_kill_potential', 'aoe_attacks',
            'adaptation_ability', 'infinity_barrier'
        ]
        
        self.feature_columns = feature_cols
        return df[feature_cols]
    
    def train(self, df):
        """Train the battle prediction model"""
        print("=" * 60)
        print("TOJI ZENIN BATTLE PREDICTOR - TRAINING")
        print("=" * 60)
        
        # Prepare features and labels
        X = self.prepare_features(df)
        y = df['outcome']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("\n🎯 Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Train Gradient Boosting
        print("🎯 Training Gradient Boosting Classifier...")
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate both models
        rf_score = rf_model.score(X_test_scaled, y_test)
        gb_score = gb_model.score(X_test_scaled, y_test)
        
        print(f"\n📊 Random Forest Accuracy: {rf_score:.3f}")
        print(f"📊 Gradient Boosting Accuracy: {gb_score:.3f}")
        
        # Choose best model
        if rf_score >= gb_score:
            self.model = rf_model
            print("\n✅ Selected Random Forest as final model")
        else:
            self.model = gb_model
            print("\n✅ Selected Gradient Boosting as final model")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=3)
        print(f"📊 Cross-validation scores: {cv_scores}")
        print(f"📊 Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Test set performance
        y_pred = self.model.predict(X_test_scaled)
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self._print_feature_importance()
        
        return self.model
    
    def _print_feature_importance(self):
        """Print top feature importances"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        print("\n" + "=" * 60)
        print("TOP 10 MOST IMPORTANT FEATURES")
        print("=" * 60)
        for i, idx in enumerate(indices, 1):
            print(f"{i:2d}. {self.feature_columns[idx]:25s} : {importances[idx]:.4f}")
    
    def predict_battle(self, battle_features):
        """Predict outcome of a single battle"""
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        # Prepare features
        X = pd.DataFrame([battle_features])[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Decode prediction
        outcome = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probability for each class
        prob_dict = {}
        for i, label in enumerate(self.label_encoder.classes_):
            prob_dict[label] = probabilities[i]
        
        return {
            'predicted_outcome': outcome,
            'probabilities': prob_dict,
            'win_chance': prob_dict.get('win', 0) * 100,
            'draw_chance': prob_dict.get('draw', 0) * 100,
            'loss_chance': prob_dict.get('loss', 0) * 100
        }
    
    def save_model(self, filepath='toji_battle_model.pkl'):
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n💾 Model saved to {filepath}")
    
    def load_model(self, filepath='toji_battle_model.pkl'):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        print(f"\n📂 Model loaded from {filepath}")
