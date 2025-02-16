"""
DynamoDB operations for churn risk score storage and intervention tracking.

Uses AWS DynamoDB to persist:
- Customer churn risk scores (partition key: customer_id, sort key: prediction_date)
- Intervention history and outcomes

All operations use real boto3 DynamoDB calls. Table names and region
are pulled from centralized config.
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List, Optional

import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError

from config import (
    DYNAMODB_RISK_TABLE,
    DYNAMODB_INTERVENTION_TABLE,
    get_aws_session_config,
)


class DynamoRiskStore:
    """
    Manages DynamoDB operations for customer risk scores and interventions.
    
    Table schema for risk scores:
        - Partition key: customer_id (String)
        - Sort key: prediction_date (String, ISO format)
        - Attributes: risk_score, risk_tier, top_factors, recommendation, model_version
    
    Table schema for interventions:
        - Partition key: customer_id (String)
        - Sort key: intervention_date (String, ISO format)
        - Attributes: intervention_type, action_taken, outcome, owner
    """
    
    def __init__(self):
        """
        Initialize DynamoDB resource and table references.
        
        Uses boto3.resource for the higher-level Table interface which provides
        automatic type serialization (Python dict -> DynamoDB map, etc.)
        """
        aws_config = get_aws_session_config()
        
        # Create DynamoDB resource — resource interface is preferred for table operations
        # as it handles type marshalling automatically
        self.dynamodb = boto3.resource(
            "dynamodb",
            region_name=aws_config["region_name"],
            aws_access_key_id=aws_config["aws_access_key_id"],
            aws_secret_access_key=aws_config["aws_secret_access_key"],
        )
        
        # Reference the risk scores table
        self.risk_table = self.dynamodb.Table(DYNAMODB_RISK_TABLE)
        
        # Reference the interventions table
        self.intervention_table = self.dynamodb.Table(DYNAMODB_INTERVENTION_TABLE)
    
    def store_risk_score(self, customer_id: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a churn risk prediction in DynamoDB.
        
        Each prediction is stored with a timestamp sort key, creating a
        historical record of risk scores over time. This enables trend
        analysis and model performance monitoring.
        
        Args:
            customer_id: Unique customer identifier
            prediction: Dict from ChurnPredictor.predict_single() with risk_score,
                       risk_tier, top_factors, recommendation
            
        Returns:
            Dict with the stored item details
        """
        prediction_date = datetime.utcnow().isoformat()
        
        # Convert float values to Decimal for DynamoDB compatibility
        # DynamoDB does not support Python float type — must use Decimal
        item = {
            "customer_id": customer_id,
            "prediction_date": prediction_date,
            "risk_score": Decimal(str(round(prediction["risk_score"], 4))),
            "risk_tier": prediction["risk_tier"],
            "top_factors": json.dumps(prediction.get("top_factors", [])),
            "recommendation": prediction.get("recommendation", ""),
            "model_version": prediction.get("model_version", "unknown"),
            "ttl": int(datetime.utcnow().timestamp()) + (90 * 86400),  # 90-day TTL
        }
        
        # put_item writes the item to DynamoDB, overwriting any existing item
        # with the same partition key + sort key combination
        self.risk_table.put_item(Item=item)
        
        return {"status": "stored", "customer_id": customer_id, "prediction_date": prediction_date}
    
    def get_latest_risk_score(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent risk score for a customer.
        
        Queries the risk table using the customer_id partition key and
        returns the item with the most recent prediction_date sort key.
        
        Args:
            customer_id: Customer to look up
            
        Returns:
            Dict with the latest risk score data, or None if not found
        """
        # Query with ScanIndexForward=False returns items in descending sort key order
        # Limit=1 gets only the most recent prediction
        response = self.risk_table.query(
            KeyConditionExpression=Key("customer_id").eq(customer_id),
            ScanIndexForward=False,  # Descending order by prediction_date
            Limit=1,
        )
        
        items = response.get("Items", [])
        if not items:
            return None
        
        item = items[0]
        # Convert Decimal back to float for Python processing
        if "risk_score" in item:
            item["risk_score"] = float(item["risk_score"])
        
        return item
    
    def get_risk_score_history(
        self,
        customer_id: str,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical risk scores for a customer.
        
        Returns up to `limit` most recent predictions, useful for
        displaying risk score trends in the Streamlit dashboard.
        
        Args:
            customer_id: Customer to look up
            limit: Maximum number of historical scores to return
            
        Returns:
            List of risk score records ordered by date descending
        """
        response = self.risk_table.query(
            KeyConditionExpression=Key("customer_id").eq(customer_id),
            ScanIndexForward=False,
            Limit=limit,
        )
        
        items = response.get("Items", [])
        for item in items:
            if "risk_score" in item:
                item["risk_score"] = float(item["risk_score"])
        
        return items
    
    def store_intervention(
        self,
        customer_id: str,
        intervention_type: str,
        action_taken: str,
        outcome: str = "pending",
        owner: str = "",
    ) -> Dict[str, Any]:
        """
        Record an intervention action in DynamoDB.
        
        Tracks all interventions attempted for each customer, enabling
        analysis of which intervention strategies are most effective
        for different risk profiles.
        
        Args:
            customer_id: Customer the intervention targets
            intervention_type: Category (proactive/reactive)
            action_taken: Description of the intervention
            outcome: Result (pending/successful/unsuccessful)
            owner: Person or team responsible
            
        Returns:
            Dict with stored intervention details
        """
        intervention_date = datetime.utcnow().isoformat()
        
        item = {
            "customer_id": customer_id,
            "intervention_date": intervention_date,
            "intervention_type": intervention_type,
            "action_taken": action_taken,
            "outcome": outcome,
            "owner": owner,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        # Write the intervention record to DynamoDB
        self.intervention_table.put_item(Item=item)
        
        return {"status": "recorded", "customer_id": customer_id, "date": intervention_date}
    
    def get_intervention_history(
        self,
        customer_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve intervention history for a customer.
        
        Args:
            customer_id: Customer to look up
            limit: Maximum number of interventions to return
            
        Returns:
            List of intervention records ordered by date descending
        """
        response = self.intervention_table.query(
            KeyConditionExpression=Key("customer_id").eq(customer_id),
            ScanIndexForward=False,
            Limit=limit,
        )
        
        return response.get("Items", [])
    
    def update_intervention_outcome(
        self,
        customer_id: str,
        intervention_date: str,
        outcome: str,
    ) -> Dict[str, Any]:
        """
        Update the outcome of a previously recorded intervention.
        
        Called when the result of an intervention is known (e.g., customer
        renewed, customer churned despite intervention).
        
        Args:
            customer_id: Customer ID (partition key)
            intervention_date: Date of the intervention (sort key)
            outcome: Updated outcome value
            
        Returns:
            Dict with update confirmation
        """
        # UpdateExpression modifies specific attributes without replacing the entire item
        self.intervention_table.update_item(
            Key={
                "customer_id": customer_id,
                "intervention_date": intervention_date,
            },
            UpdateExpression="SET outcome = :outcome, updated_at = :updated_at",
            ExpressionAttributeValues={
                ":outcome": outcome,
                ":updated_at": datetime.utcnow().isoformat(),
            },
        )
        
        return {"status": "updated", "customer_id": customer_id, "outcome": outcome}
    
    def scan_all_risk_scores(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Scan the risk table to get all current risk scores.
        
        WARNING: Scan operations read every item in the table and should be
        used sparingly. In production, use a GSI on risk_tier for efficient
        querying. This scan is acceptable for dashboard initialization with
        small-to-medium datasets.
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of all risk score records (most recent per customer)
        """
        # Scan the entire table — expensive for large datasets but acceptable
        # for dashboards with < 10k customers
        response = self.risk_table.scan(Limit=limit)
        items = response.get("Items", [])
        
        # Handle pagination for large tables
        while "LastEvaluatedKey" in response and len(items) < limit:
            response = self.risk_table.scan(
                ExclusiveStartKey=response["LastEvaluatedKey"],
                Limit=limit - len(items),
            )
            items.extend(response.get("Items", []))
        
        # Convert Decimal to float
        for item in items:
            if "risk_score" in item:
                item["risk_score"] = float(item["risk_score"])
        
        # Deduplicate to keep only the most recent score per customer
        latest_by_customer = {}
        for item in items:
            cid = item["customer_id"]
            if cid not in latest_by_customer or item["prediction_date"] > latest_by_customer[cid]["prediction_date"]:
                latest_by_customer[cid] = item
        
        return list(latest_by_customer.values())
    
    def batch_store_risk_scores(
        self,
        predictions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Store multiple risk predictions in a single batch write.
        
        Uses DynamoDB batch_writer for efficient bulk writes. The batch
        writer automatically handles chunking into 25-item batches
        (DynamoDB's limit per batch_write_item call) and retries for
        unprocessed items.
        
        Args:
            predictions: List of prediction dicts with customer_id and scores
            
        Returns:
            Dict with count of items stored
        """
        prediction_date = datetime.utcnow().isoformat()
        count = 0
        
        # batch_writer handles chunking and retry logic automatically
        with self.risk_table.batch_writer() as batch:
            for pred in predictions:
                item = {
                    "customer_id": str(pred["customer_id"]),
                    "prediction_date": prediction_date,
                    "risk_score": Decimal(str(round(pred["risk_score"], 4))),
                    "risk_tier": pred["risk_tier"],
                    "recommendation": pred.get("recommendation", ""),
                }
                batch.put_item(Item=item)
                count += 1
        
        return {"status": "batch_complete", "items_stored": count}
