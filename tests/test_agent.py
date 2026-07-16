"""
Unit tests for the pure, synchronous parts of cyclic.agent.Agent.
These are plain sync tests with no Docker or network dependency.
"""

import pytest
from cyclic.agent import Agent


def test_agent_constructs_without_api_key():
    """Test that Agent() can be instantiated with no env vars and no network call."""
    Agent()


def test_build_messages_without_history():
    """Test that _build_messages with no history returns system + user messages."""
    agent = Agent()

    messages = agent._build_messages("hello")

    assert messages == [
        {"role": "system", "content": Agent.SYSTEM_PROMPT},
        {"role": "user", "content": "hello"},
    ]


def test_build_messages_with_history():
    """Test that _build_messages includes history between system and user messages."""
    agent = Agent()

    messages = agent._build_messages(
        "hello", history=[{"role": "user", "content": "prior"}]
    )

    assert messages == [
        {"role": "system", "content": Agent.SYSTEM_PROMPT},
        {"role": "user", "content": "prior"},
        {"role": "user", "content": "hello"},
    ]


def test_build_messages_rejects_invalid_history():
    """Test that invalid history entries raise ValueError."""
    agent = Agent()

    with pytest.raises(ValueError):
        agent._build_messages("hello", history=[{"role": "user"}])

    with pytest.raises(ValueError):
        agent._build_messages("hello", history=["not a dict"])
