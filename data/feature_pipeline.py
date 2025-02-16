"""
Feature Pipeline for Churn Mitigation Prediction System.

Generates engineered features from raw customer behavioral data including:
- Lag features (1-month, 3-month, 6-month lookback)
- Rolling averages (usage, engagement, support tickets)
- Engagement trend indicators (increasing, stable, declining)

Designed to work with the IBM HR Analytics Employee Attrition dataset
from Kaggle, with additional synthetic behavioral columns generated
for demonstration purposes.
"""

import pandas as pd
import numpy as np
from typing import Optional


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw customer data from CSV file.
    
    The IBM HR Analytics dataset contains employee attributes such as
    MonthlyIncome, YearsAtCompany, JobSatisfaction, etc. We treat these
    as proxy features for customer behavioral data in a churn context.
    
    Args:
        filepath: Path to the CSV file containing raw customer data
        
    Returns:
        DataFrame with raw customer behavioral data
    """
    df = pd.read_csv(filepath)
    
    # Map IBM HR dataset columns to churn-relevant feature names
    column_mapping = {
        "EmployeeNumber": "customer_id",
        "Attrition": "churned",
        "MonthlyIncome": "monthly_spend",
        "YearsAtCompany": "tenure_years",
        "JobSatisfaction": "satisfaction_score",
        "WorkLifeBalance": "engagement_score",
        "NumCompaniesWorked": "competitor_interactions",
        "TotalWorkingYears": "total_relationship_years",
        "TrainingTimesLastYear": "support_tickets_last_year",
        "YearsSinceLastPromotion": "years_since_last_upgrade",
        "PercentSalaryHike": "discount_rate_offered",
        "OverTime": "high_usage_flag",
        "DistanceFromHome": "distance_to_nearest_competitor",
        "Age": "customer_age",
        "DailyRate": "daily_usage_rate",
        "HourlyRate": "hourly_engagement_rate",
        "MonthlyRate": "monthly_engagement_rate",
        "PerformanceRating": "loyalty_rating",
    }
    
    # Rename columns that exist in the dataset
    existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_cols)
    
    # Convert churned to binary (Yes=1, No=0)
    if "churned" in df.columns:
        df["churned"] = df["churned"].map({"Yes": 1, "No": 0})
    
    # Convert high_usage_flag to binary
    if "high_usage_flag" in df.columns:
        df["high_usage_flag"] = df["high_usage_flag"].map({"Yes": 1, "No": 0})
    
    return df


def generate_synthetic_time_series(df: pd.DataFrame, n_months: int = 12) -> pd.DataFrame:
    """
    Generate synthetic monthly behavioral data for each customer.
    
    Since the IBM HR dataset is a single snapshot, we create synthetic
    monthly time series data based on existing features to demonstrate
    lag feature and rolling average calculations.
    
    Args:
        df: DataFrame with customer-level features
        n_months: Number of months of synthetic history to generate
        
    Returns:
        DataFrame with monthly time series per customer
    """
    records = []
    
    for _, row in df.iterrows():
        customer_id = row.get("customer_id", row.name)
        base_spend = row.get("monthly_spend", 5000)
        base_engagement = row.get("engagement_score", 3)
        base_usage = row.get("daily_usage_rate", 500)
        churned = row.get("churned", 0)
        
        for month in range(n_months):
            # Simulate declining engagement for churned customers
            # Churned customers show a gradual decline in the last 3-4 months
            if churned == 1 and month >= n_months - 4:
                decay_factor = 1 - (0.15 * (month - (n_months - 4)))
            else:
                decay_factor = 1.0
            
            # Add natural noise to simulate real behavioral data
            noise = np.random.normal(0, 0.05)
            
            records.append({
                "customer_id": customer_id,
                "month": month + 1,
                "monthly_spend": max(0, base_spend * decay_factor * (1 + noise)),
                "engagement_score": max(1, min(5, base_engagement * decay_factor + np.random.normal(0, 0.3))),
                "daily_usage": max(0, base_usage * decay_factor * (1 + noise)),
                "support_tickets": max(0, int(np.random.poisson(2 if churned and month >= n_months - 3 else 0.5))),
                "login_frequency": max(0, int(np.random.poisson(20 * decay_factor))),
                "churned": churned,
            })
    
    return pd.DataFrame(records)


def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag features for each customer's behavioral metrics.
    
    Lag features capture historical values at specific lookback periods:
    - 1-month lag: most recent prior behavior
    - 3-month lag: short-term historical baseline
    - 6-month lag: medium-term historical baseline
    
    These help the model detect sudden drops in engagement or spend
    that signal churn risk.
    
    Args:
        df: DataFrame with monthly time series data, sorted by customer_id and month
        
    Returns:
        DataFrame with lag features appended
    """
    # Ensure data is sorted for correct lag computation
    df = df.sort_values(["customer_id", "month"]).reset_index(drop=True)
    
    # Define metrics to create lag features for
    lag_metrics = ["monthly_spend", "engagement_score", "daily_usage", "support_tickets", "login_frequency"]
    lag_periods = [1, 3, 6]  # 1-month, 3-month, 6-month lookback
    
    for metric in lag_metrics:
        for lag in lag_periods:
            col_name = f"{metric}_lag_{lag}m"
            # Group by customer and shift to get the value from N months ago
            df[col_name] = df.groupby("customer_id")[metric].shift(lag)
    
    # Compute month-over-month deltas for spend and engagement
    # These capture the direction and magnitude of recent changes
    df["spend_delta_1m"] = df["monthly_spend"] - df.groupby("customer_id")["monthly_spend"].shift(1)
    df["engagement_delta_1m"] = df["engagement_score"] - df.groupby("customer_id")["engagement_score"].shift(1)
    df["usage_delta_1m"] = df["daily_usage"] - df.groupby("customer_id")["daily_usage"].shift(1)
    
    return df


def compute_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling averages for key behavioral metrics.
    
    Rolling averages smooth out noise and reveal sustained trends:
    - 3-month rolling: captures recent behavioral trend
    - 6-month rolling: captures medium-term behavioral baseline
    
    The ratio of short-term to long-term rolling average is a strong
    churn signal — when the 3-month average drops below the 6-month
    average, it indicates declining engagement.
    
    Args:
        df: DataFrame with monthly time series data
        
    Returns:
        DataFrame with rolling average features appended
    """
    df = df.sort_values(["customer_id", "month"]).reset_index(drop=True)
    
    rolling_metrics = ["monthly_spend", "engagement_score", "daily_usage", "login_frequency"]
    windows = [3, 6]  # 3-month and 6-month rolling windows
    
    for metric in rolling_metrics:
        for window in windows:
            col_name = f"{metric}_rolling_{window}m"
            # Rolling mean within each customer's time series
            df[col_name] = (
                df.groupby("customer_id")[metric]
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )
        
        # Compute the ratio of short-term to long-term rolling average
        # Values < 1.0 indicate declining trend — strong churn signal
        short_col = f"{metric}_rolling_3m"
        long_col = f"{metric}_rolling_6m"
        ratio_col = f"{metric}_trend_ratio"
        df[ratio_col] = df[short_col] / df[long_col].replace(0, np.nan)
    
    return df


def compute_engagement_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each customer's engagement trajectory as increasing, stable, or declining.
    
    Uses the slope of a 3-month linear regression on engagement metrics
    to determine trend direction. This categorical feature helps the model
    distinguish between customers with similar current scores but different
    trajectories.
    
    Args:
        df: DataFrame with monthly time series and rolling features
        
    Returns:
        DataFrame with trend indicator columns appended
    """
    df = df.sort_values(["customer_id", "month"]).reset_index(drop=True)
    
    def classify_trend(group: pd.DataFrame, metric: str, window: int = 3) -> pd.Series:
        """
        Compute rolling slope of a metric and classify as trend direction.
        
        Thresholds:
        - slope > 0.05: "increasing"
        - slope < -0.05: "declining"  
        - otherwise: "stable"
        """
        slopes = []
        values = group[metric].values
        
        for i in range(len(values)):
            if i < window - 1:
                slopes.append(0)  # Not enough history for trend
            else:
                # Simple linear regression slope over the window
                y = values[i - window + 1 : i + 1]
                x = np.arange(window)
                if np.std(y) == 0:
                    slopes.append(0)
                else:
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
        
        # Normalize slopes relative to the metric's range
        slope_series = pd.Series(slopes, index=group.index)
        metric_range = group[metric].max() - group[metric].min()
        if metric_range > 0:
            normalized = slope_series / metric_range
        else:
            normalized = slope_series
        
        # Classify into trend categories
        trend = pd.Series("stable", index=group.index)
        trend[normalized > 0.05] = "increasing"
        trend[normalized < -0.05] = "declining"
        
        return trend
    
    # Compute trend indicators for key engagement metrics
    trend_metrics = ["engagement_score", "daily_usage", "login_frequency"]
    
    for metric in trend_metrics:
        col_name = f"{metric}_trend"
        trends = []
        for _, group in df.groupby("customer_id"):
            trend = classify_trend(group, metric)
            trends.append(trend)
        df[col_name] = pd.concat(trends)
    
    # Create a composite engagement health score
    # Declining trends in multiple metrics compound the risk
    def composite_health(row):
        declining_count = sum([
            1 for m in trend_metrics
            if row.get(f"{m}_trend") == "declining"
        ])
        if declining_count >= 2:
            return "at_risk"
        elif declining_count == 1:
            return "warning"
        else:
            return "healthy"
    
    df["engagement_health"] = df.apply(composite_health, axis=1)
    
    return df


def run_feature_pipeline(
    raw_data_path: str,
    output_path: Optional[str] = None,
    n_months: int = 12
) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline.
    
    Pipeline steps:
    1. Load raw customer data from CSV
    2. Generate synthetic monthly time series
    3. Compute lag features (1m, 3m, 6m lookback)
    4. Compute rolling averages (3m, 6m windows)
    5. Compute engagement trend indicators
    6. Drop rows with insufficient history (NaN from lags)
    7. Save processed features to output path
    
    Args:
        raw_data_path: Path to the raw customer data CSV
        output_path: Optional path to save processed features
        n_months: Number of months of synthetic history
        
    Returns:
        DataFrame with all engineered features
    """
    print("[feature_pipeline] Loading raw data...")
    raw_df = load_raw_data(raw_data_path)
    print(f"[feature_pipeline] Loaded {len(raw_df)} customers")
    
    print("[feature_pipeline] Generating synthetic time series...")
    ts_df = generate_synthetic_time_series(raw_df, n_months=n_months)
    print(f"[feature_pipeline] Generated {len(ts_df)} monthly records")
    
    print("[feature_pipeline] Computing lag features...")
    featured_df = compute_lag_features(ts_df)
    
    print("[feature_pipeline] Computing rolling averages...")
    featured_df = compute_rolling_averages(featured_df)
    
    print("[feature_pipeline] Computing engagement trend indicators...")
    featured_df = compute_engagement_trend_indicators(featured_df)
    
    # Drop rows where lag features couldn't be computed (first N months)
    initial_rows = len(featured_df)
    featured_df = featured_df.dropna(subset=[
        "monthly_spend_lag_6m",
        "engagement_score_lag_6m"
    ])
    print(f"[feature_pipeline] Dropped {initial_rows - len(featured_df)} rows with insufficient history")
    
    # One-hot encode categorical trend indicators
    trend_cols = [c for c in featured_df.columns if c.endswith("_trend")]
    featured_df = pd.get_dummies(featured_df, columns=trend_cols, prefix_sep="_is_")
    
    # One-hot encode engagement health
    featured_df = pd.get_dummies(featured_df, columns=["engagement_health"], prefix_sep="_is_")
    
    if output_path:
        featured_df.to_csv(output_path, index=False)
        print(f"[feature_pipeline] Saved processed features to {output_path}")
    
    print(f"[feature_pipeline] Final dataset: {featured_df.shape[0]} rows, {featured_df.shape[1]} columns")
    
    return featured_df


if __name__ == "__main__":
    import config
    
    # Run the full pipeline using paths from config
    result = run_feature_pipeline(
        raw_data_path=config.TRAINING_DATA_PATH,
        output_path=config.PROCESSED_DATA_PATH,
        n_months=12
    )
    print(f"\nFeature columns: {list(result.columns)}")
    print(f"\nSample data:\n{result.head()}")
