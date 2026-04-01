from __future__ import annotations

import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator


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
    return Langfuse(**kwargs)


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
    *,
    metadata: dict[str, Any] | None = None,
    model: str,
) -> Iterator[Any | None]:
    if not langfuse_client:
        yield None
        return

    with start_langfuse_observation(
        langfuse_client,
        name=span_name,
        as_type="span",
        input=trace_input,
        metadata=metadata or {},
    ):
        with start_langfuse_observation(
            langfuse_client,
            name="anthropic.messages.create",
            as_type="generation",
            model=model,
            input=trace_input,
        ) as generation:
            yield generation
    langfuse_client.flush()


def traced_model_call(
    get_langfuse_client: Callable[[], Any | None],
    *,
    model: str,
    span_name_fn: Callable[..., str],
    input_fn: Callable[..., list[Any]],
    metadata_fn: Callable[..., dict[str, Any]],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            langfuse_client = get_langfuse_client()
            trace_input = input_fn(*args, **kwargs)
            span_name = span_name_fn(*args, **kwargs)
            metadata = metadata_fn(*args, **kwargs)
            with trace_generation(
                langfuse_client,
                span_name,
                trace_input,
                metadata=metadata,
                model=model,
            ) as generation:
                response = func(*args, **kwargs)
                if generation:
                    generation.update(
                        output=serialize_response_blocks(response.content),
                        metadata={"stop_reason": response.stop_reason},
                    )
                return response

        return wrapper

    return decorator
