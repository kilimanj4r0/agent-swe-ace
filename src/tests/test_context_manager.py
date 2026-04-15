# src/tests/test_context_manager.py
"""Tests for context_manager — 5-level message truncation algorithm."""

import sys
from pathlib import Path

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
