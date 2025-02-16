"""
AWS Bedrock AgentCore Integration for Churn Mitigation System.

Provides a conversational AI interface powered by AWS Bedrock AgentCore
that allows business users to query churn risk data using natural language.

The agent supports:
- Natural language queries about customer churn risk
- Intervention recommendations based on model predictions
- Portfolio-level risk summaries
- Provider-swappable tool interface via AgentCore's built-in routing

Architecture:
- AgentCore agent configured with custom tools (churn_score_query, intervention_recommendation)
- Agent invoked via boto3 bedrock-agent-runtime client
- Session management enables multi-turn conversations
- Provider-swappable: AgentCore handles model routing internally,
  supporting Anthropic Claude and Amazon Titan as backing models
"""

import json
import uuid
from typing import Dict, Any, Optional, Generator

import boto3
from botocore.config import Config

from config import (
    BEDROCK_AGENT_ID,
    BEDROCK_AGENT_ALIAS_ID,
    AWS_REGION,
    get_aws_session_config,
)


class BedrockChurnAgent:
    """
    Conversational agent powered by AWS Bedrock AgentCore.
    
    AgentCore was chosen over direct LLM API calls because:
    1. Built-in tool routing eliminates custom orchestration code
    2. Provider-swappable: switch between Claude and Titan without code changes
    3. Native session management for multi-turn conversations
    4. Guardrails integration for response safety
    5. Managed infrastructure reduces operational overhead
    
    The agent is configured with two custom tools:
    - churn_score_query: Retrieves risk scores for specific customers
    - intervention_recommendation: Generates retention action plans
    """
    
    def __init__(self):
        """
        Initialize the Bedrock AgentCore client.
        
        Uses bedrock-agent-runtime for invoking the agent at runtime.
        The agent itself is configured in the AWS Console or via CloudFormation
        with the tools defined in agent/tools.py.
        """
        aws_config = get_aws_session_config()
        
        # Configure boto3 client with retry settings for production reliability
        retry_config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            read_timeout=120,  # Agent responses can take time with tool calls
        )
        
        # bedrock-agent-runtime client — used for invoking agents at runtime
        # This is separate from the bedrock-agent client used for agent management
        self.agent_client = boto3.client(
            "bedrock-agent-runtime",
            region_name=aws_config["region_name"],
            aws_access_key_id=aws_config["aws_access_key_id"],
            aws_secret_access_key=aws_config["aws_secret_access_key"],
            config=retry_config,
        )
        
        self.agent_id = BEDROCK_AGENT_ID
        self.agent_alias_id = BEDROCK_AGENT_ALIAS_ID
        
        # Session tracking for multi-turn conversations
        self.sessions: Dict[str, str] = {}
    
    def create_session(self) -> str:
        """
        Create a new conversation session.
        
        Each session maintains context across multiple turns, allowing
        the agent to reference previous queries and build on prior analysis.
        
        Returns:
            Session ID string for tracking the conversation
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = session_id
        return session_id
    
    def invoke_agent(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        enable_trace: bool = False,
    ) -> Dict[str, Any]:
        """
        Send a natural language query to the Bedrock AgentCore agent.
        
        The agent processes the query through these steps:
        1. Parses the user's intent
        2. Determines which tools to invoke (if any)
        3. Executes tool calls and collects results
        4. Synthesizes a natural language response
        
        Args:
            prompt: Natural language query from the user
            session_id: Existing session ID for multi-turn conversations
            enable_trace: If True, returns detailed agent reasoning trace
            
        Returns:
            Dict with response text, session_id, and optional trace data
        """
        if session_id is None:
            session_id = self.create_session()
        
        # Invoke the agent via the bedrock-agent-runtime API
        # This sends the prompt to the AgentCore agent which handles:
        # - Intent classification
        # - Tool selection and invocation
        # - Response generation with the configured backing model
        response = self.agent_client.invoke_agent(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            sessionId=session_id,
            inputText=prompt,
            enableTrace=enable_trace,
        )
        
        # Parse the streaming response from AgentCore
        # The response comes as an EventStream that we need to consume
        completion_text = ""
        trace_data = []
        
        event_stream = response.get("completion", [])
        for event in event_stream:
            # Extract text chunks from the response stream
            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    completion_text += chunk["bytes"].decode("utf-8")
            
            # Capture trace information if enabled
            # Traces show the agent's reasoning, tool calls, and intermediate steps
            if "trace" in event and enable_trace:
                trace_data.append(event["trace"])
        
        result = {
            "response": completion_text,
            "session_id": session_id,
        }
        
        if enable_trace:
            result["trace"] = trace_data
            result["tool_calls"] = self._extract_tool_calls(trace_data)
        
        return result
    
    def invoke_agent_streaming(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream the agent's response token by token.
        
        Used by the Streamlit UI to display responses as they are generated,
        providing a better user experience for longer analytical responses.
        
        Args:
            prompt: Natural language query from the user
            session_id: Existing session ID for multi-turn conversations
            
        Yields:
            Text chunks as they arrive from the agent
        """
        if session_id is None:
            session_id = self.create_session()
        
        response = self.agent_client.invoke_agent(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            sessionId=session_id,
            inputText=prompt,
            enableTrace=False,
        )
        
        event_stream = response.get("completion", [])
        for event in event_stream:
            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    text = chunk["bytes"].decode("utf-8")
                    yield text
    
    def _extract_tool_calls(self, trace_data: list) -> list:
        """
        Extract tool call information from agent traces.
        
        Parses the trace data to identify which tools the agent invoked,
        what inputs it provided, and what results were returned. This is
        displayed in the Streamlit UI for transparency.
        
        Args:
            trace_data: Raw trace events from the agent response
            
        Returns:
            List of tool call summaries
        """
        tool_calls = []
        
        for trace in trace_data:
            trace_detail = trace.get("trace", {})
            
            # Check for orchestration traces which contain tool invocations
            if "orchestrationTrace" in trace_detail:
                orch = trace_detail["orchestrationTrace"]
                
                # Action group invocation traces show tool calls
                if "invocationInput" in orch:
                    invocation = orch["invocationInput"]
                    if "actionGroupInvocationInput" in invocation:
                        action = invocation["actionGroupInvocationInput"]
                        tool_calls.append({
                            "tool": action.get("actionGroupName", "unknown"),
                            "function": action.get("function", "unknown"),
                            "parameters": action.get("parameters", []),
                        })
                
                # Observation traces show tool results
                if "observation" in orch:
                    observation = orch["observation"]
                    if "actionGroupInvocationOutput" in observation:
                        output = observation["actionGroupInvocationOutput"]
                        if tool_calls:
                            tool_calls[-1]["result"] = output.get("text", "")
        
        return tool_calls
    
    def end_session(self, session_id: str):
        """
        Clean up a conversation session.
        
        Removes the session from local tracking. AgentCore manages
        its own session state on the server side.
        
        Args:
            session_id: Session to terminate
        """
        if session_id in self.sessions:
            del self.sessions[session_id]


def get_agent() -> BedrockChurnAgent:
    """
    Factory function that returns a configured BedrockChurnAgent instance.
    
    Used by app.py and other modules to get a consistent agent instance
    without needing to manage configuration directly.
    
    Returns:
        Configured BedrockChurnAgent ready for invocation
    """
    return BedrockChurnAgent()
