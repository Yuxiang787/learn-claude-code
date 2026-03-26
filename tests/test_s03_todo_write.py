import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch


MODULE_NAME = "agents.s03_todo_write"


def load_module():
    os.environ.setdefault("MODEL_ID", "test-model")
    sys.modules.pop(MODULE_NAME, None)
    fake_anthropic = types.SimpleNamespace(Anthropic=lambda **kwargs: types.SimpleNamespace(kwargs=kwargs))
    fake_dotenv = types.SimpleNamespace(load_dotenv=lambda **kwargs: None)
    with patch.dict(
        sys.modules,
        {
            "anthropic": fake_anthropic,
            "dotenv": fake_dotenv,
        },
    ):
        return importlib.import_module(MODULE_NAME)


class FakeObservation:
    def __init__(self, bucket, kind, name, kwargs):
        self.bucket = bucket
        self.kind = kind
        self.name = name
        self.kwargs = kwargs
        self.updated = []

    def update(self, **kwargs):
        self.updated.append(kwargs)

    def __enter__(self):
        self.bucket.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeLangfuseClient:
    def __init__(self):
        self.observations = []
        self.flush_called = False

    def start_as_current_span(self, name, **kwargs):
        return FakeObservation(self.observations, "span", name, kwargs)

    def start_as_current_generation(self, name, **kwargs):
        return FakeObservation(self.observations, "generation", name, kwargs)

    def flush(self):
        self.flush_called = True


@contextmanager
def temp_env(**updates):
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class LangfuseHelpersTest(unittest.TestCase):
    def test_langfuse_disabled_without_required_env(self):
        with temp_env(
            LANGFUSE_PUBLIC_KEY=None,
            LANGFUSE_SECRET_KEY=None,
            LANGFUSE_HOST=None,
        ):
            mod = load_module()

        self.assertFalse(mod.langfuse_enabled())
        self.assertIsNone(mod.create_langfuse_client())

    def test_langfuse_client_uses_env_when_available(self):
        fake_langfuse_module = types.SimpleNamespace(Langfuse=lambda **kwargs: kwargs)

        with temp_env(
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
            LANGFUSE_HOST="https://cloud.langfuse.com",
        ):
            mod = load_module()
            with patch.dict(sys.modules, {"langfuse": fake_langfuse_module}):
                client = mod.create_langfuse_client()

        self.assertEqual(
            client,
            {
                "public_key": "pk-test",
                "secret_key": "sk-test",
                "host": "https://cloud.langfuse.com",
            },
        )

    def test_trace_generation_records_input_output_and_flushes(self):
        mod = load_module()
        fake_client = FakeLangfuseClient()
        trace_input = [{"role": "user", "content": "Write a todo list"}]
        response_text = "Done."

        with mod.trace_generation(fake_client, "s03-turn", trace_input) as generation:
            generation.update(output=response_text, metadata={"stop_reason": "end_turn"})

        self.assertEqual(len(fake_client.observations), 2)
        span, generation = fake_client.observations
        self.assertEqual(span.kind, "span")
        self.assertEqual(span.name, "s03-turn")
        self.assertEqual(span.kwargs["input"], trace_input)
        self.assertEqual(generation.kind, "generation")
        self.assertEqual(generation.name, "anthropic.messages.create")
        self.assertEqual(generation.kwargs["model"], "test-model")
        self.assertEqual(generation.updated[-1]["output"], response_text)
        self.assertTrue(fake_client.flush_called)


if __name__ == "__main__":
    unittest.main()
