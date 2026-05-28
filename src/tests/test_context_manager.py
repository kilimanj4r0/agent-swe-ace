# src/tests/test_context_manager.py
"""Tests for context_manager — 5-level message truncation algorithm."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.context_manager import (
    estimate_tokens,
    truncate_message_content,
    is_format_error,
    truncate_messages,
)


class TestEstimateTokens:
    """Tests for estimate_tokens."""

    def test_empty_messages(self):
        assert estimate_tokens([]) == 0

    def test_empty_content(self):
        assert estimate_tokens([{"role": "user", "content": ""}]) == 4  # overhead only

    def test_basic_estimation(self):
        msg = {"role": "user", "content": "a" * 300}
        tokens = estimate_tokens([msg])
        assert tokens == 104  # 300/3 + 4

    def test_multiple_messages(self):
        msgs = [
            {"role": "user", "content": "a" * 90},
            {"role": "assistant", "content": "b" * 90},
        ]
        tokens = estimate_tokens(msgs)
        # 90/3 + 4 = 34 each, total 68
        assert tokens == 68


class TestTruncateMessageContent:
    """Tests for truncate_message_content."""

    def test_short_content_unchanged(self):
        content = "short"
        assert truncate_message_content(content, max_chars=100) == content

    def test_empty_content_unchanged(self):
        assert truncate_message_content("", max_chars=100) == ""

    def test_none_content_unchanged(self):
        assert truncate_message_content(None, max_chars=100) is None

    def test_long_content_truncated(self):
        content = "x" * 1000
        result = truncate_message_content(content, max_chars=200)
        assert len(result) < len(content)
        assert "chars truncated" in result
        # Should have head and tail
        assert result.startswith("x")
        assert result.endswith("x")

    def test_head_tail_balance(self):
        content = "a" * 500 + "b" * 500
        result = truncate_message_content(content, max_chars=200)
        # Head should be first 100 chars, tail last 100
        assert result.startswith("a")
        assert result.endswith("b")


class TestIsFormatError:
    """Tests for is_format_error."""

    def test_format_error_exact_match(self):
        assert is_format_error("Please always provide EXACTLY ONE action") is True

    def test_format_error_in_context(self):
        msg = "Error: Please always provide EXACTLY ONE action per turn."
        assert is_format_error(msg) is True

    def test_backtick_format_error(self):
        msg = "Please format your action in triple backticks"
        assert is_format_error(msg) is True

    def test_normal_message_not_error(self):
        assert is_format_error("The file has been modified successfully") is False

    def test_empty_string(self):
        assert is_format_error("") is False


class TestTruncateMessages:
    """Tests for truncate_messages — the 5-level truncation algorithm."""

    def _make_messages(self, count, content_len=1000, role_pattern=None):
        """Create a list of messages for testing.

        Args:
            count: Total messages (first 2 are system+instance)
            content_len: Length of content for non-protected messages
            role_pattern: Optional list of roles to cycle through
        """
        if role_pattern is None:
            role_pattern = ["user", "assistant"]

        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Instance description"},
        ]
        for i in range(count - 2):
            role = role_pattern[i % len(role_pattern)]
            msgs.append({"role": role, "content": chr(65 + (i % 26)) * content_len})
        return msgs

    def test_two_messages_unchanged(self):
        """Messages with ≤2 entries should not be truncated."""
        msgs = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "instance"},
        ]
        result = truncate_messages(msgs, max_tokens=10)
        assert result == msgs

    def test_format_errors_dropped_first(self):
        """Level 1: format error messages should be dropped from middle."""
        # Design: total tokens exceed threshold, but dropping format error
        # brings them below — proving format errors are the first to go.
        msgs = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "instance"},
            {"role": "user", "content": "Please always provide EXACTLY ONE action"},
            {"role": "user", "content": "x" * 2000},  # ~670 tokens
            {"role": "user", "content": "final"},
        ]
        # With format error: ~710 tokens, without: ~692 tokens
        result = truncate_messages(msgs, max_tokens=700, keep_recent=2)
        contents = [m["content"] for m in result]
        assert "Please always provide EXACTLY ONE action" not in contents

    def test_long_observations_truncated(self):
        """Level 2: old observations (user messages) should be truncated."""
        long_content = "x" * 5000
        msgs = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "instance"},
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": "recent"},
        ]
        result = truncate_messages(msgs, max_tokens=500, keep_recent=1)
        # The long user message in the middle should be truncated
        middle = result[2]
        assert len(middle["content"]) < len(long_content)

    def test_assistant_thought_truncated_keeps_bash(self):
        """Level 3: assistant THOUGHT truncated but bash commands kept."""
        content = "Let me analyze this.\n\nTHOUGHT\nI need to check the file.\n\n```bash\ncat /etc/hosts\n```\n"
        msgs = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "instance"},
            {"role": "assistant", "content": content * 10},
            {"role": "user", "content": "recent"},
        ]
        result = truncate_messages(msgs, max_tokens=200, keep_recent=1)
        assistant_msg = result[2]
        # Should keep bash command
        if "```bash" in assistant_msg["content"]:
            assert "cat /etc/hosts" in assistant_msg["content"]

    def test_protected_messages_always_kept(self):
        """System and instance messages (indices 0,1) are never dropped."""
        msgs = self._make_messages(20, content_len=500)
        # Use generous max_tokens to avoid infinite loop — just enough to
        # trigger truncation but not so small it can never succeed
        result = truncate_messages(msgs, max_tokens=5000, keep_recent=2)
        assert result[0]["content"] == "System prompt"
        assert result[1]["content"] == "Instance description"

    def test_keep_recent_preserved(self):
        """Last keep_recent messages should be preserved intact."""
        msgs = self._make_messages(10, content_len=200)
        result = truncate_messages(msgs, max_tokens=5000, keep_recent=3)
        # Last 3 messages should match the original last 3
        for i in range(3):
            assert result[-(i + 1)]["role"] == msgs[-(i + 1)]["role"]


# --- I2: TestContextAwareDefaultAgent ---


class TestContextAwareDefaultAgent:
    """Tests for ContextAwareDefaultAgent wrapper."""

    @patch("minisweagent.agents.default.DefaultAgent")
    def test_no_truncation_below_threshold(self, MockDefaultAgent):
        """Messages below threshold should pass through unchanged."""
        from agents.context_manager import ContextAwareDefaultAgent

        # Configure mock agent instance
        mock_agent_instance = MagicMock()
        mock_agent_instance.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "short message"},
        ]
        mock_agent_instance.query = MagicMock(return_value={"exit_status": "submitted"})
        MockDefaultAgent.return_value = mock_agent_instance

        # Create wrapper with small max_input_tokens so threshold is low but
        # messages are still well below it
        wrapper = ContextAwareDefaultAgent(
            model="test-model",
            env=MagicMock(),
            config_class=MagicMock(),
            max_input_tokens=10000,
        )

        original_messages = list(mock_agent_instance.messages)
        wrapper._query_with_context_management()

        # Messages should be unchanged — no truncation triggered
        assert mock_agent_instance.messages == original_messages
        assert wrapper._truncation_count == 0

    @patch("minisweagent.agents.default.DefaultAgent")
    def test_truncation_above_threshold(self, MockDefaultAgent):
        """Messages above threshold should be truncated."""
        from agents.context_manager import ContextAwareDefaultAgent

        # Configure mock agent instance
        mock_agent_instance = MagicMock()
        # Create enough long messages to exceed 850 tokens (1000 * 0.85)
        mock_agent_instance.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "instance"},
        ]
        # Add many long messages to push past 850 tokens
        for i in range(20):
            mock_agent_instance.messages.append(
                {"role": "user", "content": "x" * 300}
            )
        mock_agent_instance.query = MagicMock(return_value={"exit_status": "submitted"})
        MockDefaultAgent.return_value = mock_agent_instance

        wrapper = ContextAwareDefaultAgent(
            model="test-model",
            env=MagicMock(),
            config_class=MagicMock(),
            max_input_tokens=1000,
        )

        tokens_before = estimate_tokens(mock_agent_instance.messages)
        assert tokens_before > 850  # Sanity: above threshold

        wrapper._query_with_context_management()

        tokens_after = estimate_tokens(mock_agent_instance.messages)
        assert wrapper._truncation_count == 1
        assert tokens_after < tokens_before


# --- M5: TestTruncateMessagesEdgeCases ---


class TestTruncateMessagesEdgeCases:
    """Edge-case tests for truncate_messages."""

    def test_all_format_errors_dropped(self):
        """When ALL middle messages are format errors, none remain after truncation."""
        # Use large padding on format error messages so dropping them actually
        # brings the token count below the threshold.
        padding = "x" * 3000  # ~1000 tokens each
        msgs = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "instance"},
            {"role": "user", "content": "Please always provide EXACTLY ONE action " + padding},
            {"role": "user", "content": "Please format your action in triple backticks " + padding},
            {"role": "user", "content": "Please always provide EXACTLY ONE action " + padding},
            {"role": "user", "content": "recent"},
        ]
        # With format errors: ~3000+ tokens. Without (just protected + recent): ~20 tokens.
        # Set threshold so dropping errors gets us under.
        result = truncate_messages(msgs, max_tokens=1000, keep_recent=1)
        contents = [m["content"] for m in result]
        # Only system, instance, and recent should remain
        assert result[0]["content"] == "system"
        assert result[1]["content"] == "instance"
        assert "Please always provide EXACTLY ONE action" not in contents
        assert "Please format your action in triple backticks" not in contents

    def test_zero_length_content(self):
        """Messages with empty string content don't crash."""
        msgs = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "instance"},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "final"},
        ]
        # Should not raise
        result = truncate_messages(msgs, max_tokens=100, keep_recent=1)
        assert isinstance(result, list)

    def test_messages_without_content_key(self):
        """Messages missing the 'content' key are handled gracefully."""
        msgs = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "instance"},
            {"role": "user"},  # No content key
            {"role": "assistant"},  # No content key
            {"role": "user", "content": "final"},
        ]
        # Should not raise KeyError
        result = truncate_messages(msgs, max_tokens=100, keep_recent=1)
        assert isinstance(result, list)

    def test_max_tokens_small(self):
        """Very small max_tokens still returns protected messages (first 2)."""
        msgs = [
            {"role": "system", "content": "system prompt here"},
            {"role": "user", "content": "instance description here"},
            {"role": "user", "content": "x" * 3000},
            {"role": "assistant", "content": "y" * 3000},
            {"role": "user", "content": "z" * 3000},
        ]
        result = truncate_messages(msgs, max_tokens=1, keep_recent=1)
        # Protected messages (first 2) should always be present
        assert len(result) >= 2
        assert result[0]["content"] == "system prompt here"
        assert result[1]["content"] == "instance description here"
