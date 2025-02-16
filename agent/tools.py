"""
Tool definitions for Bedrock AgentCore Churn Mitigation Agent.

Defines the tools (action groups) that the AgentCore agent can invoke.
The tool interface is provider-swappable — AgentCore handles routing
to the appropriate backing model while these tools remain model-agnostic.
"""

import json
from typing import Dict, Any, List
from datetime import datetime


# =============================================================================
# Tool Schema Definitions (OpenAPI format for AgentCore)
# =============================================================================

TOOL_SCHEMAS = {
    "churn_score_query": {
        "name": "churn_score_query",
        "description": (
            "Query the churn risk score for a specific customer by their customer ID. "
            "Returns the risk score (0-1), risk tier (high/medium/low), top contributing "
            "factors, and recommended intervention."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "The unique identifier for the customer to query",
                "required": True,
            }
        },
    },
    "intervention_recommendation": {
        "name": "intervention_recommendation",
        "description": (
            "Generate a detailed intervention recommendation for an at-risk customer. "
            "Considers the customer's risk score, historical engagement trends, and "
            "previous intervention outcomes."
        ),
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "The unique identifier for the customer",
                "required": True,
            },
            "intervention_type": {
                "type": "string",
                "description": "Type of intervention: 'proactive', 'reactive', or 'auto'",
                "required": False,
                "default": "auto",
            },
        },
    },
    "portfolio_risk_summary": {
        "name": "portfolio_risk_summary",
        "description": (
            "Get a summary of churn risk across the entire customer portfolio. "
            "Returns counts by risk tier and the most at-risk accounts."
        ),
        "parameters": {
            "limit": {
                "type": "integer",
                "description": "Max number of at-risk accounts to return",
                "required": False,
                "default": 10,
            },
        },
    },
}


def handle_churn_score_query(customer_id: str, dynamo_store) -> Dict[str, Any]:
    """
    Handle the churn_score_query tool invocation.

    Checks DynamoDB for a cached recent prediction for the customer.

    Args:
        customer_id: Customer to query
        dynamo_store: DynamoDB risk store instance

    Returns:
        Dict with risk score, tier, factors, and recommendation
    """
    cached_score = dynamo_store.get_latest_risk_score(customer_id)

    if cached_score:
        return {
            "customer_id": customer_id,
            "risk_score": cached_score["risk_score"],
            "risk_tier": cached_score["risk_tier"],
            "prediction_date": cached_score["prediction_date"],
            "source": "cached",
            "recommendation": cached_score.get("recommendation", ""),
        }

    return {
        "customer_id": customer_id,
        "status": "no_prediction_found",
        "message": f"No churn risk score found for customer {customer_id}.",
    }


def handle_intervention_recommendation(
    customer_id: str,
    intervention_type: str,
    dynamo_store,
) -> Dict[str, Any]:
    """
    Handle the intervention_recommendation tool invocation.

    Generates a tailored intervention plan based on the customer's risk
    profile and previous intervention history.

    Args:
        customer_id: Customer to generate recommendation for
        intervention_type: 'proactive', 'reactive', or 'auto'
        dynamo_store: DynamoDB risk store instance

    Returns:
        Dict with recommended actions and expected outcomes
    """
    risk_data = dynamo_store.get_latest_risk_score(customer_id)

    if not risk_data:
        return {
            "customer_id": customer_id,
            "status": "error",
            "message": f"No risk data found for customer {customer_id}",
        }

    risk_tier = risk_data.get("risk_tier", "unknown")
    risk_score = risk_data.get("risk_score", 0.0)

    if intervention_type == "auto":
        intervention_type = "reactive" if risk_tier == "high" else "proactive"

    previous_interventions = dynamo_store.get_intervention_history(customer_id)
    recommendations = _build_intervention_plan(
        risk_tier, risk_score, intervention_type, previous_interventions
    )

    return {
        "customer_id": customer_id,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "intervention_type": intervention_type,
        "recommendations": recommendations,
        "previous_intervention_count": len(previous_interventions),
        "generated_at": datetime.utcnow().isoformat(),
    }


def handle_portfolio_risk_summary(limit: int, dynamo_store) -> Dict[str, Any]:
    """
    Handle the portfolio_risk_summary tool invocation.

    Scans DynamoDB risk table to aggregate portfolio-level metrics.

    Args:
        limit: Maximum number of at-risk accounts to return
        dynamo_store: DynamoDB risk store instance

    Returns:
        Dict with portfolio metrics and top at-risk accounts
    """
    all_scores = dynamo_store.scan_all_risk_scores()

    if not all_scores:
        return {"status": "empty", "message": "No risk scores found"}

    tier_counts = {"high": 0, "medium": 0, "low": 0}
    for score in all_scores:
        tier = score.get("risk_tier", "low")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    total = len(all_scores)
    sorted_scores = sorted(all_scores, key=lambda x: x.get("risk_score", 0), reverse=True)

    return {
        "total_customers": total,
        "risk_distribution": {
            tier: {"count": count, "percentage": f"{count/total*100:.1f}%"}
            for tier, count in tier_counts.items()
        },
        "top_at_risk_accounts": [
            {"customer_id": s["customer_id"], "risk_score": s["risk_score"], "risk_tier": s["risk_tier"]}
            for s in sorted_scores[:limit]
        ],
    }


def _build_intervention_plan(risk_tier, risk_score, intervention_type, previous_interventions):
    """
    Build a structured intervention plan based on risk profile.
    """
    has_prior = len(previous_interventions) > 0

    if risk_tier == "high":
        plan = [
            {"action": "Executive escalation call", "priority": "P0 — within 24 hours",
             "expected_impact": "30% retention rate", "owner": "VP Customer Success"},
            {"action": "Custom retention offer (20-30% discount)", "priority": "P0 — prepare before call",
             "expected_impact": "45% acceptance rate", "owner": "Account Manager"},
            {"action": "Product roadmap preview", "priority": "P1 — within 48 hours",
             "expected_impact": "Increases switching cost", "owner": "Product Manager"},
        ]
        if has_prior:
            plan.append({"action": "Customer Advisory Board invitation", "priority": "P1",
                         "expected_impact": "25% retention lift", "owner": "VP Customer Success"})
        return plan

    elif risk_tier == "medium":
        return [
            {"action": "Proactive health check call", "priority": "P1 — within 7 days",
             "expected_impact": "50% reduction in escalation", "owner": "Account Manager"},
            {"action": "Targeted feature adoption campaign", "priority": "P2 — within 14 days",
             "expected_impact": "15% churn probability reduction", "owner": "Customer Success"},
            {"action": "Quarterly business review scheduling", "priority": "P2 — within 30 days",
             "expected_impact": "Prevents engagement drift", "owner": "Account Manager"},
        ]

    return [
        {"action": "Continue standard engagement", "priority": "P3",
         "expected_impact": "Maintains healthy relationship", "owner": "Customer Success"},
        {"action": "Include in product update newsletter", "priority": "P3",
         "expected_impact": "Keeps customer informed", "owner": "Marketing"},
    ]


def get_tool_definitions() -> Dict[str, Dict]:
    """Return all tool definitions in Bedrock AgentCore format."""
    return TOOL_SCHEMAS
