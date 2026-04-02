"""Context window management for mini-swe-agent's DefaultAgent.

Prevents ContextWindowExceededError by proactively truncating message history
before it exceeds the model's context window limit.
"""

import re
from loguru import logger


def estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count for a list of messages.

    Uses chars/3.0 ratio — conservative estimate to account for
    tokenizer inefficiency on code-heavy content and special tokens.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if content:
            total += int(len(content) / 3.0)
        total += 4  # overhead per message (role, formatting, etc.)
    return total


def truncate_message_content(content: str, max_chars: int = 2000) -> str:
    """Truncate message content using head/tail pattern."""
    if not content or len(content) <= max_chars:
        return content

    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    omitted = len(content) - max_chars

    return (
        content[:head_chars]
        + f"\n\n[{omitted} chars truncated]\n\n"
        + content[-tail_chars:]
    )


def is_format_error(content: str) -> bool:
    """Check if a message is a format error (low-value, repetitive)."""
    return (
        "Please always provide EXACTLY ONE action" in content
        or "Please format your action in triple backticks" in content
    )


def truncate_messages(
    messages: list[dict],
    max_tokens: int,
    keep_recent: int = 6,
) -> list[dict]:
    """Truncate message history to fit within token limit.

    Priority order:
    1. Always keep system message (index 0) and instance/task message (index 1)
    2. Always keep last `keep_recent` messages intact
    3. Drop format error messages in the middle
    4. Truncate old observation content (user messages with command output)
    5. Truncate old assistant THOUGHT sections (keep bash commands)
    6. Drop oldest non-essential messages entirely
    7. Reduce keep_recent and repeat truncation
    """
    if len(messages) <= 2:
        return messages

    # Try progressively smaller keep_recent values if needed
    effective_keep = keep_recent
    while effective_keep >= 2:
        result = _truncate_with_keep_recent(messages, max_tokens, effective_keep)
        if estimate_tokens(result) <= max_tokens:
            return result
        # Reduce keep_recent by half and try again
        effective_keep = max(2, effective_keep // 2)
        logger.debug(f"Context: reducing keep_recent to {effective_keep}")

    # Absolute last resort: truncate everything
    for msg in messages:
        content = msg.get("content", "")
        if len(content) > 500:
            msg["content"] = truncate_message_content(content, max_chars=500)
    logger.warning("Context: extreme truncation applied to all messages")
    return messages


def _truncate_with_keep_recent(
    messages: list[dict],
    max_tokens: int,
    keep_recent: int,
) -> list[dict]:
    """Truncate messages with a specific keep_recent value."""
    if len(messages) <= 2 + keep_recent:
        return messages

    # Split messages into: protected, middle, recent
    protected = messages[:2]  # system + instance
    recent = messages[-keep_recent:] if keep_recent > 0 else []
    middle = messages[2 : len(messages) - keep_recent] if keep_recent > 0 else messages[2:]

    result = list(protected) + list(middle) + list(recent)

    # Level 1: Drop format errors from middle
    if estimate_tokens(result) > max_tokens:
        middle_filtered = [m for m in middle if not is_format_error(m.get("content", ""))]
        result = list(protected) + middle_filtered + list(recent)
        dropped = len(middle) - len(middle_filtered)
        if dropped:
            logger.debug(f"Context: dropped {dropped} format error messages")

    # Level 2: Truncate old observation content (user messages)
    if estimate_tokens(result) > max_tokens:
        current_middle = result[len(protected) : len(result) - len(recent)]
        truncated_count = 0
        for msg in current_middle:
            if msg.get("role") == "user" and not is_format_error(msg.get("content", "")):
                content = msg["content"]
                if len(content) > 1000:
                    msg["content"] = truncate_message_content(content, max_chars=1000)
                    truncated_count += 1
        if truncated_count:
            logger.debug(f"Context: truncated {truncated_count} old observations")

    # Level 3: Truncate old assistant messages (keep bash commands, shorten THOUGHT)
    if estimate_tokens(result) > max_tokens:
        current_middle = result[len(protected) : len(result) - len(recent)]
        truncated_count = 0
        for msg in current_middle:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if len(content) > 500:
                    bash_blocks = re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL)
                    if bash_blocks:
                        last_cmd = bash_blocks[-1].strip()
                        msg["content"] = f"[earlier reasoning truncated]\n\n```bash\n{last_cmd}\n```"
                        truncated_count += 1
                    else:
                        msg["content"] = truncate_message_content(content, max_chars=500)
                        truncated_count += 1
        if truncated_count:
            logger.debug(f"Context: truncated {truncated_count} old assistant messages")

    # Level 4: Drop oldest middle messages entirely
    if estimate_tokens(result) > max_tokens:
        current_middle = result[len(protected) : len(result) - len(recent)]
        while current_middle and estimate_tokens(list(protected) + current_middle + list(recent)) > max_tokens:
            dropped_msg = current_middle.pop(0)
            logger.debug(f"Context: dropped oldest message ({dropped_msg.get('role')}, {len(dropped_msg.get('content', ''))} chars)")
        result = list(protected) + current_middle + list(recent)

    # Level 5: Truncate recent messages content
    if estimate_tokens(result) > max_tokens:
        for msg in result[2:]:  # skip protected
            content = msg.get("content", "")
            if len(content) > 500:
                msg["content"] = truncate_message_content(content, max_chars=500)
        logger.warning("Context: truncated message content (last resort)")

    return result


class ContextAwareDefaultAgent:
    """Wrapper around DefaultAgent with proactive context window management.

    Subclasses DefaultAgent to override query(), checking token count
    and truncating messages before each LLM call.
    """

    def __init__(
        self,
        model,
        env,
        *,
        config_class,
        max_input_tokens: int = 60000,
        keep_recent_messages: int = 6,
        truncate_threshold: float = 0.85,
        max_tokens: int = 4096,
        **kwargs,
    ):
        from minisweagent.agents.default import DefaultAgent

        # Create the real DefaultAgent instance
        self._agent = DefaultAgent(
            model=model,
            env=env,
            config_class=config_class,
            **kwargs,
        )
        # Monkey-patch query to add context management
        self._agent.query = self._query_with_context_management
        self._max_input_tokens = max_input_tokens
        self._keep_recent_messages = keep_recent_messages
        self._truncate_threshold = truncate_threshold
        self._max_tokens = max_tokens
        self._truncation_count = 0

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped agent."""
        return getattr(self._agent, name)

    def _query_with_context_management(self) -> dict:
        """Query with proactive context window management."""
        threshold = int(self._max_input_tokens * self._truncate_threshold)
        estimated = estimate_tokens(self._agent.messages)

        if estimated > threshold:
            before = estimated
            # Use threshold as the truncation target, not max_input_tokens.
            # This ensures we actually reduce tokens when the check triggers.
            self._agent.messages = truncate_messages(
                self._agent.messages,
                max_tokens=threshold,
                keep_recent=self._keep_recent_messages,
            )
            after = estimate_tokens(self._agent.messages)
            self._truncation_count += 1
            logger.warning(
                f"Context window management triggered: "
                f"{before} -> {after} tokens (target: {threshold}, "
                f"saved {before - after} tokens, truncation #{self._truncation_count})"
            )

        # Call the original query
        from minisweagent.agents.default import DefaultAgent
        return DefaultAgent.query(self._agent)
