import importlib
import os
import runpy
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


MODULE_NAME = "agents.s04_subagent"


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


class S04LangfuseTracingTest(unittest.TestCase):
    def test_script_can_run_with_local_langfuse_helper_import(self):
        script_path = Path("/Users/slumber/Downloads/learn-claude-code/agents/s04_subagent.py")
        script_dir = str(script_path.parent)
        fake_anthropic = types.SimpleNamespace(
            Anthropic=lambda **kwargs: types.SimpleNamespace(
                kwargs=kwargs,
                messages=types.SimpleNamespace(create=lambda **create_kwargs: None),
            )
        )
        fake_dotenv = types.SimpleNamespace(load_dotenv=lambda **kwargs: None)
        fake_tracing = types.SimpleNamespace(
            create_langfuse_client=lambda: None,
            traced_model_call=lambda *args, **kwargs: (lambda func: func),
        )

        with temp_env(MODEL_ID="test-model"):
            with patch.dict(
                sys.modules,
                {
                    "anthropic": fake_anthropic,
                    "dotenv": fake_dotenv,
                    "langfuse_tracing": fake_tracing,
                },
                clear=False,
            ):
                original_sys_path = sys.path[:]
                try:
                    sys.path = [script_dir]
                    exec(script_path.read_text(), {"__name__": "__test__", "__file__": str(script_path)})
                finally:
                    sys.path = original_sys_path

    def test_call_parent_model_traces_main_agent_response(self):
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

        messages = [{"role": "user", "content": "Use a subtask to inspect tests"}]
        fake_response = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="I can do that.")],
            stop_reason="end_turn",
        )
        mod.client.messages.create = lambda **kwargs: fake_response

        response = mod.call_parent_model(messages)

        self.assertIs(response, fake_response)
        self.assertEqual(len(fake_langfuse_client.observations), 2)
        span, generation = fake_langfuse_client.observations
        self.assertEqual(span.kind, "span")
        self.assertEqual(span.name, "s04-parent-turn")
        self.assertEqual(span.kwargs["input"], messages)
        self.assertEqual(span.kwargs["metadata"], {"agent": "s04_parent"})
        self.assertEqual(generation.kind, "generation")
        self.assertEqual(generation.kwargs["model"], "test-model")
        self.assertEqual(generation.updated[-1]["output"], ["I can do that."])
        self.assertEqual(generation.updated[-1]["metadata"], {"stop_reason": "end_turn"})
        self.assertTrue(fake_langfuse_client.flush_called)

    def test_call_subagent_model_traces_subagent_response(self):
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

        messages = [{"role": "user", "content": "Find the test framework"}]
        fake_response = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="The repo uses unittest.")],
            stop_reason="end_turn",
        )
        mod.client.messages.create = lambda **kwargs: fake_response

        response = mod.call_subagent_model(messages, prompt="Find the test framework")

        self.assertIs(response, fake_response)
        self.assertEqual(len(fake_langfuse_client.observations), 2)
        span, generation = fake_langfuse_client.observations
        self.assertEqual(span.kind, "span")
        self.assertEqual(span.name, "s04-subagent: Find the test framework")
        self.assertEqual(span.kwargs["input"], messages)
        self.assertEqual(span.kwargs["metadata"], {"agent": "s04_subagent", "prompt": "Find the test framework"})
        self.assertEqual(generation.kind, "generation")
        self.assertEqual(generation.updated[-1]["output"], ["The repo uses unittest."])
        self.assertEqual(generation.updated[-1]["metadata"], {"stop_reason": "end_turn"})
        self.assertTrue(fake_langfuse_client.flush_called)


if __name__ == "__main__":
    unittest.main()
