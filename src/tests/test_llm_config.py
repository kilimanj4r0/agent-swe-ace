"""
Simple LLM Configuration Test

Tests that config.yaml settings work for both agent and ace LLMs.
Makes at least 1 real API call to verify connectivity.

Run: uv run python src/tests/test_llm_config.py
"""

import sys
from pathlib import Path

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config():
    """Load config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_config_loading():
    """Test that config.yaml can be loaded."""
    config = load_config()

    assert "llm" in config, "Missing 'llm' section in config.yaml"
    assert "agent" in config["llm"], "Missing 'agent' section in llm config"
    assert "ace" in config["llm"], "Missing 'ace' section in llm config"

    print("✓ config.yaml loaded successfully")


def test_agent_model_creation():
    """Test that agent model can be created from config.yaml."""
    from config.llm import create_model_from_yaml, LLMConfig

    config = load_config()
    agent_config = config["llm"]["agent"]

    print(f"\n[Agent] Provider: {agent_config['provider']}")
    print(f"[Agent] Model: {agent_config['model']}")

    # Create LLMConfig to verify model string
    llm_config = LLMConfig.from_dict(agent_config)
    print(f"[Agent] Model string: {llm_config.get_model_string()}")

    # Create model
    model = create_model_from_yaml(agent_config)
    print(f"[Agent] Model class: {type(model).__name__}")
    print("[Agent] ✓ OK")


def test_ace_client_creation():
    """Test that ACE client can be created from config.yaml."""
    from config.llm import create_ace_client, LLMConfig

    config = load_config()
    ace_config = config["llm"]["ace"]

    print(f"\n[ACE] Provider: {ace_config['provider']}")
    print(f"[ACE] Model: {ace_config['model']}")

    # Create LLMConfig to verify model string
    llm_config = LLMConfig.from_dict(ace_config)
    print(f"[ACE] Model string: {llm_config.get_model_string()}")

    # Create client
    client = create_ace_client(ace_config)
    print(f"[ACE] Client class: {type(client).__name__}")
    print("[ACE] ✓ OK")


def test_agent_llm_call():
    """Test agent LLM with a real API call."""
    from config.llm import create_model_from_yaml

    config = load_config()
    agent_config = config["llm"]["agent"]

    print(f"\n[Agent API Test] Provider: {agent_config['provider']}")
    print(f"[Agent API Test] Model: {agent_config['model']}")

    # Create model
    model = create_model_from_yaml(agent_config)
    print(f"[Agent API Test] Model class: {type(model).__name__}")

    # Make a simple API call
    messages = [{"role": "user", "content": "Say exactly 'OK' and three-words wish for the day."}]
    print("[Agent API Test] Making API call...")
    response = model.query(messages)

    # Extract response content
    content = response.get("content", str(response)) if isinstance(response, dict) else str(response)
    print(f"[Agent API Test] Response: {content[:100]}...")

    assert response, "Agent LLM returned empty response"
    assert "OK" in str(content).upper() or len(content) > 0, "Agent LLM returned unexpected response"
    print("[Agent API Test] ✓ OK")


def test_ace_llm_call():
    """Test ACE LLM with a real API call."""
    from config.llm import create_model_from_yaml

    config = load_config()
    ace_config = config["llm"]["ace"]

    print(f"\n[ACE API Test] Provider: {ace_config['provider']}")
    print(f"[ACE API Test] Model: {ace_config['model']}")

    # Create model
    model = create_model_from_yaml(ace_config)
    print(f"[ACE API Test] Model class: {type(model).__name__}")

    # Make a simple API call
    messages = [{"role": "user", "content": "Say exactly 'OK' and three-words wish for the day."}]
    print("[ACE API Test] Making API call...")
    response = model.query(messages)

    # Extract response content
    content = response.get("content", str(response)) if isinstance(response, dict) else str(response)
    print(f"[ACE API Test] Response: {content[:100]}...")

    assert response, "ACE LLM returned empty response"
    assert "OK" in str(content).upper() or len(content) > 0, "ACE LLM returned unexpected response"
    print("[ACE API Test] ✓ OK")


def main():
    print("=" * 50)
    print("LLM Configuration Test")
    print("=" * 50)

    try:
        test_config_loading()
        test_agent_model_creation()
        test_ace_client_creation()
        test_agent_llm_call()
        test_ace_llm_call()
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
        return 0
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
