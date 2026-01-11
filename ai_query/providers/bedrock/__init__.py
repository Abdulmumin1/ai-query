"""AWS Bedrock provider for ai-query.

This provider requires boto3 to be installed:
    pip install ai-query[bedrock]
"""

from ai_query.providers.bedrock.provider import bedrock, BedrockProvider

__all__ = ["bedrock", "BedrockProvider"]
