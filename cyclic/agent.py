import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from litellm import acompletion
from loguru import logger


class AgentResponse(BaseModel):
    """Structured response from the Agent."""
    code: str = Field(description="The Python code to execute")
    reasoning: str = Field(description="Explanation of why this code was generated")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")


class Agent:
    """Agent that generates Python code using LLMs via LiteLLM."""

    SYSTEM_PROMPT = """You are a helpful Python code generation assistant.
Generate Python code that solves the user's request.

You must respond with a JSON object containing:
- "code": The Python code to execute (as a string)
- "reasoning": A brief explanation of your approach
- "confidence": A confidence score between 0.0 and 1.0

Important constraints:
- Do NOT use dangerous imports: os, subprocess, sys, socket, shutil, importlib
- Do NOT use dangerous functions: eval, exec, open, __import__
- Keep code simple and focused on the task
- The code will run in an isolated Docker container with no network access

Example response:
{
  "code": "print('Hello, World!')",
  "reasoning": "Simple print statement to greet the user",
  "confidence": 0.95
}"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """
        Initialize the Agent.

        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "claude-3-haiku")
            api_key: API key for the LLM provider (defaults to env var)
            base_url: Custom base URL for the API (optional)
            temperature: Sampling temperature (0.0 to 1.0)
        """
        self.model = model
        self.api_key = api_key or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.temperature = temperature

        if not self.api_key:
            logger.warning("No API key found. Set LITELLM_API_KEY or OPENAI_API_KEY environment variable.")

    async def generate(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
    ) -> AgentResponse:
        """
        Generate Python code based on a prompt and conversation history.

        Args:
            prompt: User's request/task description
            history: Previous conversation messages (optional)

        Returns:
            AgentResponse with generated code and reasoning

        Raises:
            ValueError: If API key is missing or response is invalid
        """
        if not self.api_key:
            raise ValueError("API key is required. Set LITELLM_API_KEY or OPENAI_API_KEY environment variable.")

        messages = self._build_messages(prompt, history)

        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
                api_key=self.api_key,
                base_url=self.base_url,
            )

            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty or invalid response from LLM")

            content = response.choices[0].message.content

            # Parse JSON response
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                raise ValueError(f"Invalid JSON response from LLM: {e}")

            # Validate and create AgentResponse
            return AgentResponse(**data)

        except Exception as e:
            logger.error(f"Agent generation failed: {e}")
            raise

    def _build_messages(self, prompt: str, history: Optional[List[dict]] = None) -> List[dict]:
        """Build the message list for the LLM API."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Add history if provided
        if history:
            for msg in history:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    raise ValueError(f"Invalid message format in history: {msg}")
            messages.extend(history)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        return messages


if __name__ == "__main__":
    # Quick Test
    import asyncio

    async def main():
        agent = Agent()

        try:
            response = await agent.generate("Write Python code that prints 'Hello from Agent!'")
            print(f"Generated Code:\n{response.code}\n")
            print(f"Reasoning: {response.reasoning}\n")
            print(f"Confidence: {response.confidence}\n")
            print(f"Full Response: {response.model_dump_json(indent=2)}")
        except ValueError as e:
            print(f"Error: {e}")
            print("\nNote: Set LITELLM_API_KEY or OPENAI_API_KEY environment variable to test.")

    asyncio.run(main())

