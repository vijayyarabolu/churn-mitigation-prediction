"""
Centralized configuration for the Churn Mitigation Prediction System.

All AWS resource names, region settings, and environment variables are managed here.
This ensures consistent configuration across all modules and simplifies deployment
to different environments.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()


# =============================================================================
# AWS General Configuration
# =============================================================================
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# =============================================================================
# AWS DynamoDB Configuration
# =============================================================================
# Table for storing customer churn risk scores and intervention history
DYNAMODB_RISK_TABLE = os.getenv("DYNAMODB_RISK_TABLE", "churn-risk-scores")
# Table schema: partition_key = customer_id, sort_key = prediction_date
DYNAMODB_INTERVENTION_TABLE = os.getenv("DYNAMODB_INTERVENTION_TABLE", "churn-interventions")

# =============================================================================
# AWS Bedrock AgentCore Configuration
# =============================================================================
# Bedrock AgentCore agent ID — created via AWS Console or CloudFormation
BEDROCK_AGENT_ID = os.getenv("BEDROCK_AGENT_ID")
# Bedrock AgentCore agent alias ID — points to the deployed version
BEDROCK_AGENT_ALIAS_ID = os.getenv("BEDROCK_AGENT_ALIAS_ID")

# =============================================================================
# AWS CloudFront Configuration
# =============================================================================
# CloudFront distribution domain name for low-latency application serving
CLOUDFRONT_DISTRIBUTION_DOMAIN = os.getenv("CLOUDFRONT_DISTRIBUTION_DOMAIN")
CLOUDFRONT_DISTRIBUTION_ID = os.getenv("CLOUDFRONT_DISTRIBUTION_ID")

# =============================================================================
# Model Configuration
# =============================================================================
# Path to the trained model artifact
MODEL_ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "models/xgboost_churn_model.pkl")
# Feature list version — ensures feature pipeline and model stay in sync
FEATURE_VERSION = os.getenv("FEATURE_VERSION", "v2.1")

# =============================================================================
# Application Configuration
# =============================================================================
# Streamlit app settings
APP_TITLE = "Churn Mitigation Dashboard"
APP_PAGE_ICON = "📊"
# Risk threshold for flagging at-risk customers
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.7"))
MEDIUM_RISK_THRESHOLD = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.4"))

# =============================================================================
# Data Configuration
# =============================================================================
# Path to training data (IBM HR Analytics Employee Attrition dataset)
TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "data/hr_employee_attrition.csv")
# Feature pipeline output path
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed_features.csv")


def get_aws_session_config():
    """
    Returns a dictionary of AWS session configuration parameters.
    Used by boto3 clients across the application to establish connections
    to AWS services with consistent credentials and region settings.
    """
    return {
        "aws_access_key_id": AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
        "region_name": AWS_REGION,
    }


def validate_config():
    """
    Validates that all required environment variables are set.
    Called at application startup to fail fast if configuration is incomplete.
    
    Returns:
        dict: Validation results with missing variables listed
    """
    required_vars = {
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "AWS_REGION": AWS_REGION,
        "BEDROCK_AGENT_ID": BEDROCK_AGENT_ID,
        "BEDROCK_AGENT_ALIAS_ID": BEDROCK_AGENT_ALIAS_ID,
        "DYNAMODB_RISK_TABLE": DYNAMODB_RISK_TABLE,
    }
    
    missing = [key for key, value in required_vars.items() if not value]
    
    return {
        "valid": len(missing) == 0,
        "missing": missing,
        "message": f"Missing required config: {', '.join(missing)}" if missing else "All config valid"
    }
