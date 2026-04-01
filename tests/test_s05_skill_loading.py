import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch


MODULE_NAME = "agents.s05_skill_loading"


def load_module():
    os.environ.setdefault("MODEL_ID", "test-model")
    sys.modules.pop(MODULE_NAME, None)
    fake_anthropic = types.SimpleNamespace(
        Anthropic=lambda **kwargs: types.SimpleNamespace(
            kwargs=kwargs,
            messages=types.SimpleNamespace(create=lambda **create_kwargs: None),
        )
    )
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


class FakeLangfuseClientV3:
    def __init__(self):
        self.observations = []
        self.flush_called = False

    def start_as_current_observation(self, name, *, as_type="span", **kwargs):
        return FakeObservation(self.observations, as_type, name, kwargs)

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


class S05LangfuseTracingTest(unittest.TestCase):
    def test_call_model_traces_skill_loading_response(self):
        fake_langfuse_client = FakeLangfuseClientV3()
        fake_langfuse_module = types.SimpleNamespace(Langfuse=lambda **kwargs: fake_langfuse_client)

        with temp_env(
            MODEL_ID="test-model",
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
            LANGFUSE_HOST="https://cloud.langfuse.com",
        ):
            with patch.dict(sys.modules, {"langfuse": fake_langfuse_module}):
                mod = load_module()

        messages = [{"role": "user", "content": "Load the code-review skill"}]
        fake_response = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="I can load that skill.")],
            stop_reason="end_turn",
        )
        mod.client.messages.create = lambda **kwargs: fake_response

        response = mod.call_model(messages)

        self.assertIs(response, fake_response)
        self.assertEqual(len(fake_langfuse_client.observations), 2)
        span, generation = fake_langfuse_client.observations
        self.assertEqual(span.kind, "span")
        self.assertEqual(span.name, "s05-user-turn")
        self.assertEqual(span.kwargs["input"], messages)
        self.assertEqual(span.kwargs["metadata"], {"agent": "s05_skill_loading"})
        self.assertEqual(generation.kind, "generation")
        self.assertEqual(generation.kwargs["model"], "test-model")
        self.assertEqual(generation.updated[-1]["output"], ["I can load that skill."])
        self.assertEqual(generation.updated[-1]["metadata"], {"stop_reason": "end_turn"})
        self.assertTrue(fake_langfuse_client.flush_called)


if __name__ == "__main__":
    unittest.main()
