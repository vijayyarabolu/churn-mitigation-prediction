"""
Prediction module for Churn Mitigation System.

Provides the prediction endpoint logic that loads the trained XGBoost model
and generates churn risk scores for individual customers or batches.
Designed to be invoked by the Bedrock AgentCore tools and the Streamlit dashboard.
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path


class ChurnPredictor:
    """
    Loads the trained XGBoost churn model and provides prediction methods.
    
    The predictor handles:
    - Single customer risk scoring
    - Batch prediction for monitoring dashboards
    - Risk tier classification (high/medium/low)
    - Prediction explanation via feature importance
    """
    
    def __init__(self, model_path: str = "models/xgboost_churn_model.pkl"):
        """
        Initialize the predictor by loading the trained model artifact.
        
        Args:
            model_path: Path to the serialized XGBoost model
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.benchmark_metrics = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the trained model and metadata from the pickle artifact.
        
        Raises:
            FileNotFoundError: If the model artifact doesn't exist
            ValueError: If the model artifact is corrupted or incompatible
        """
        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                f"Run models/train.py first to train and save the model."
            )
        
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.benchmark_metrics = model_data.get("benchmark_metrics", {})
        
        print(f"[predictor] Loaded model with {len(self.feature_names)} features")
        print(f"[predictor] Model accuracy: {self.benchmark_metrics.get('accuracy', 'N/A')}")
    
    def predict_single(self, customer_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate a churn risk prediction for a single customer.
        
        Args:
            customer_features: Dictionary mapping feature names to values
            
        Returns:
            Dict with risk_score, risk_tier, and top contributing factors
        """
        # Build feature vector in the correct order
        feature_vector = np.zeros(len(self.feature_names))
        for i, name in enumerate(self.feature_names):
            if name in customer_features:
                feature_vector[i] = customer_features[name]
        
        # Generate prediction probability
        # predict_proba returns [P(not churn), P(churn)]
        churn_probability = self.model.predict_proba(feature_vector.reshape(1, -1))[0][1]
        
        # Classify into risk tiers
        risk_tier = self._classify_risk_tier(churn_probability)
        
        # Get top contributing factors using feature importance
        top_factors = self._get_top_factors(feature_vector, n_top=5)
        
        # Generate intervention recommendation based on risk tier
        recommendation = self._get_intervention_recommendation(risk_tier, top_factors)
        
        return {
            "risk_score": float(churn_probability),
            "risk_tier": risk_tier,
            "top_factors": top_factors,
            "recommendation": recommendation,
            "model_version": self.benchmark_metrics.get("model_name", "XGBoost"),
        }
    
    def predict_batch(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate churn risk predictions for a batch of customers.
        
        Used by the Streamlit dashboard to populate the at-risk accounts
        monitoring table and by scheduled DynamoDB writes for risk score updates.
        
        Args:
            customer_df: DataFrame with customer features
            
        Returns:
            DataFrame with risk_score and risk_tier columns appended
        """
        # Ensure feature columns are in the correct order
        feature_cols = [c for c in self.feature_names if c in customer_df.columns]
        missing_cols = [c for c in self.feature_names if c not in customer_df.columns]
        
        if missing_cols:
            print(f"[predictor] Warning: {len(missing_cols)} missing features, using zeros")
        
        # Build the feature matrix
        X = np.zeros((len(customer_df), len(self.feature_names)))
        for i, name in enumerate(self.feature_names):
            if name in customer_df.columns:
                X[:, i] = customer_df[name].fillna(0).values
        
        # Generate predictions
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Append results to the dataframe
        result_df = customer_df.copy()
        result_df["risk_score"] = probabilities
        result_df["risk_tier"] = [self._classify_risk_tier(p) for p in probabilities]
        
        # Sort by risk score descending — highest risk first
        result_df = result_df.sort_values("risk_score", ascending=False)
        
        return result_df
    
    def _classify_risk_tier(self, probability: float) -> str:
        """
        Classify a churn probability into a risk tier.
        
        Thresholds calibrated on validation set performance:
        - High (≥0.7): Immediate intervention required
        - Medium (≥0.4): Proactive engagement recommended
        - Low (<0.4): Standard monitoring
        
        Args:
            probability: Churn probability from the model
            
        Returns:
            Risk tier string
        """
        if probability >= 0.7:
            return "high"
        elif probability >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_top_factors(self, feature_vector: np.ndarray, n_top: int = 5) -> List[Dict[str, Any]]:
        """
        Identify the top contributing factors to a prediction.
        
        Uses the model's global feature importance weights multiplied by
        the customer's feature values to approximate local importance.
        
        For production, consider integrating SHAP values for more accurate
        local explanations.
        
        Args:
            feature_vector: The customer's feature values
            n_top: Number of top factors to return
            
        Returns:
            List of dicts with feature name, importance, and value
        """
        importances = self.model.feature_importances_
        
        # Approximate local importance: global importance * feature deviation from mean
        local_importance = importances * np.abs(feature_vector)
        
        # Get top N indices
        top_indices = np.argsort(local_importance)[::-1][:n_top]
        
        factors = []
        for idx in top_indices:
            factors.append({
                "feature": self.feature_names[idx],
                "importance": float(local_importance[idx]),
                "value": float(feature_vector[idx]),
            })
        
        return factors
    
    def _get_intervention_recommendation(
        self,
        risk_tier: str,
        top_factors: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a plain-language intervention recommendation.
        
        Maps risk tiers and contributing factors to actionable recommendations
        for customer success teams.
        
        Args:
            risk_tier: Customer's risk classification
            top_factors: Top contributing factors to the prediction
            
        Returns:
            Human-readable intervention recommendation
        """
        if risk_tier == "high":
            factor_names = [f["feature"] for f in top_factors[:3]]
            return (
                f"URGENT: High churn risk detected. Key risk drivers: "
                f"{', '.join(factor_names)}. Recommend immediate personal outreach "
                f"from account manager with retention offer."
            )
        elif risk_tier == "medium":
            return (
                "PROACTIVE: Moderate churn risk. Recommend scheduling a check-in "
                "call within 7 days and reviewing account health metrics. Consider "
                "targeted engagement campaign."
            )
        else:
            return (
                "MONITOR: Low churn risk. Continue standard engagement cadence. "
                "Flag for review if engagement metrics decline in next 30 days."
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata for display in the Streamlit dashboard.
        
        Returns:
            Dict with model version, accuracy, feature count, etc.
        """
        return {
            "model_name": "XGBoost Churn Predictor",
            "n_features": len(self.feature_names),
            "accuracy": self.benchmark_metrics.get("accuracy"),
            "auc_roc": self.benchmark_metrics.get("auc_roc"),
            "inference_time_ms": self.benchmark_metrics.get("inference_time_ms"),
            "feature_names": self.feature_names,
        }
