#!/usr/bin/env python3
# Harness: planning -- keeping the model on course without scripting the route.
"""
s03_todo_write.py - TodoWrite

The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets.

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

Key insight: "The agent can track its own progress -- and I can see it."
"""

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
client = Anthropic(base_url=BASE_URL)
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def create_langfuse_client() -> Any | None:
    if not langfuse_enabled():
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        return None

    kwargs: dict[str, str] = {
        "public_key": os.environ["LANGFUSE_PUBLIC_KEY"],
        "secret_key": os.environ["LANGFUSE_SECRET_KEY"],
    }
    langfuse_host = os.getenv("LANGFUSE_HOST")
    if langfuse_host:
        kwargs["host"] = langfuse_host
    return Langfuse(**kwargs) # type: ignore


LANGFUSE = create_langfuse_client()


def start_langfuse_observation(
    langfuse_client: Any,
    name: str,
    *,
    as_type: str = "span",
    **kwargs: Any,
) -> Any:
    if hasattr(langfuse_client, "start_as_current_observation"):
        return langfuse_client.start_as_current_observation(name=name, as_type=as_type, **kwargs)

    if as_type == "generation" and hasattr(langfuse_client, "start_as_current_generation"):
        return langfuse_client.start_as_current_generation(name=name, **kwargs)

    if as_type == "span" and hasattr(langfuse_client, "start_as_current_span"):
        return langfuse_client.start_as_current_span(name=name, **kwargs)

    raise AttributeError(f"Unsupported Langfuse client API for observation type: {as_type}")


def serialize_response_blocks(blocks: list[Any]) -> list[Any]:
    output: list[Any] = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            output.append(block.text)
        elif block_type == "tool_use":
            output.append({"tool_use": block.name})
        elif block_type == "thinking":
            output.append({"thinking": getattr(block, "thinking", "")})
        else:
            output.append({"type": block_type or "unknown"})
    return output


@contextmanager
def trace_generation(
    langfuse_client: Any,
    span_name: str,
    trace_input: list[Any],
) -> Iterator[Any | None]:
    if not langfuse_client:
        yield None
        return

    with start_langfuse_observation(
        langfuse_client,
        name=span_name,
        as_type="span",
        input=trace_input,
        metadata={"agent": "s03_todo_write"},
    ):
        with start_langfuse_observation(
            langfuse_client,
            name="anthropic.messages.create",
            as_type="generation",
            model=MODEL,
            input=trace_input,
        ) as generation:
            yield generation
    langfuse_client.flush()


# -- TodoManager: structured state the LLM writes to --
class TodoManager:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def update(self, items: list[dict[str, Any]]) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated: list[dict[str, str]] = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines: list[str] = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS: dict[str, Any] = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo":       lambda **kw: TODO.update(kw["items"]),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
]


# -- Agent loop with nag reminder injection --
def agent_loop(messages: list[Any]) -> None:
    rounds_since_todo = 0
    while True:
        # Nag reminder is injected below, alongside tool results
        with trace_generation(LANGFUSE, "s03-user-turn", messages) as generation:
            response = client.messages.create(
                model=MODEL, system=SYSTEM, messages=messages,
                tools=TOOLS, max_tokens=8000,
            )
            if generation:
                generation.update(
                    output=serialize_response_blocks(response.content),
                    metadata={"stop_reason": response.stop_reason},
                )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results: list[dict[str, str]] = []
        used_todo = False
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}: {str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                if block.name == "todo":
                    used_todo = True
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.insert(0, {"type": "text", "text": "<reminder>Update your todos.</reminder>"})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    if LANGFUSE:
        print("Langfuse tracing: enabled")
    history: list[dict[str, Any]] = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
