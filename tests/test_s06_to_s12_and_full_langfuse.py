import importlib
import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch


def load_module(module_name: str):
    os.environ.setdefault("MODEL_ID", "test-model")
    sys.modules.pop(module_name, None)
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
        return importlib.import_module(module_name)


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


class RemainingTutorialLangfuseTracingTest(unittest.TestCase):
    def import_with_langfuse(self, module_name: str):
        fake_langfuse_client = FakeLangfuseClientV3()
        fake_langfuse_module = types.SimpleNamespace(
            Langfuse=lambda **kwargs: fake_langfuse_client
        )
        with temp_env(
            MODEL_ID="test-model",
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
            LANGFUSE_HOST="https://cloud.langfuse.com",
        ):
            with patch.dict(sys.modules, {"langfuse": fake_langfuse_module}):
                mod = load_module(module_name)
        return mod, fake_langfuse_client

    def assert_traced_call(
        self,
        module_name: str,
        call_name: str,
        args: tuple,
        span_name: str,
        metadata: dict,
        output_text: str,
    ):
        mod, fake_langfuse_client = self.import_with_langfuse(module_name)
        fake_response = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=output_text)],
            stop_reason="end_turn",
        )
        mod.client.messages.create = lambda **kwargs: fake_response

        response = getattr(mod, call_name)(*args)

        self.assertIs(response, fake_response)
        self.assertEqual(len(fake_langfuse_client.observations), 2)
        span, generation = fake_langfuse_client.observations
        self.assertEqual(span.kind, "span")
        self.assertEqual(span.name, span_name)
        self.assertEqual(span.kwargs["metadata"], metadata)
        self.assertEqual(generation.kind, "generation")
        self.assertEqual(generation.kwargs["model"], "test-model")
        self.assertEqual(generation.updated[-1]["output"], [output_text])
        self.assertEqual(generation.updated[-1]["metadata"], {"stop_reason": "end_turn"})
        self.assertTrue(fake_langfuse_client.flush_called)

    def test_s06_call_model_is_traced(self):
        messages = [{"role": "user", "content": "Compact the transcript if needed"}]
        self.assert_traced_call(
            "agents.s06_context_compact",
            "call_model",
            (messages,),
            "s06-user-turn",
            {"agent": "s06_context_compact"},
            "Working on it.",
        )

    def test_s06_compactor_is_traced(self):
        self.assert_traced_call(
            "agents.s06_context_compact",
            "call_compactor_model",
            ("chat history",),
            "s06-auto-compact",
            {"agent": "s06_compactor"},
            "Summary text.",
        )

    def test_s07_call_model_is_traced(self):
        self.assert_traced_call(
            "agents.s07_task_system",
            "call_model",
            (([{"role": "user", "content": "List tasks"}]),),
            "s07-user-turn",
            {"agent": "s07_task_system"},
            "No tasks yet.",
        )

    def test_s08_call_model_is_traced(self):
        self.assert_traced_call(
            "agents.s08_background_tasks",
            "call_model",
            (([{"role": "user", "content": "Start a background task"}]),),
            "s08-user-turn",
            {"agent": "s08_background_tasks"},
            "Started.",
        )

    def test_s09_lead_and_teammate_calls_are_traced(self):
        self.assert_traced_call(
            "agents.s09_agent_teams",
            "call_lead_model",
            (([{"role": "user", "content": "Spawn alice"}]),),
            "s09-lead-turn",
            {"agent": "s09_lead"},
            "Lead response.",
        )
        self.assert_traced_call(
            "agents.s09_agent_teams",
            "call_teammate_model",
            (([{"role": "user", "content": "Investigate tests"}]), "sys", [], "alice"),
            "s09-teammate:alice",
            {"agent": "s09_teammate", "teammate": "alice"},
            "Teammate response.",
        )

    def test_s10_lead_and_teammate_calls_are_traced(self):
        self.assert_traced_call(
            "agents.s10_team_protocols",
            "call_lead_model",
            (([{"role": "user", "content": "Review plan"}]),),
            "s10-lead-turn",
            {"agent": "s10_lead"},
            "Lead response.",
        )
        self.assert_traced_call(
            "agents.s10_team_protocols",
            "call_teammate_model",
            (([{"role": "user", "content": "Submit a plan"}]), "sys", [], "bob"),
            "s10-teammate:bob",
            {"agent": "s10_teammate", "teammate": "bob"},
            "Teammate response.",
        )

    def test_s11_lead_and_teammate_calls_are_traced(self):
        self.assert_traced_call(
            "agents.s11_autonomous_agents",
            "call_lead_model",
            (([{"role": "user", "content": "Check inbox"}]),),
            "s11-lead-turn",
            {"agent": "s11_lead"},
            "Lead response.",
        )
        self.assert_traced_call(
            "agents.s11_autonomous_agents",
            "call_teammate_model",
            (([{"role": "user", "content": "Auto-claim work"}]), "sys", [], "carol"),
            "s11-teammate:carol",
            {"agent": "s11_teammate", "teammate": "carol"},
            "Teammate response.",
        )

    def test_s12_call_model_is_traced(self):
        self.assert_traced_call(
            "agents.s12_worktree_task_isolation",
            "call_model",
            (([{"role": "user", "content": "Create a worktree"}]),),
            "s12-user-turn",
            {"agent": "s12_worktree_task_isolation"},
            "Created.",
        )

    def test_s_full_calls_are_traced(self):
        self.assert_traced_call(
            "agents.s_full",
            "call_main_model",
            (([{"role": "user", "content": "Use the full agent"}]),),
            "s_full-user-turn",
            {"agent": "s_full"},
            "Main response.",
        )
        self.assert_traced_call(
            "agents.s_full",
            "call_subagent_model",
            (([{"role": "user", "content": "Explore"}]), [], "Explore"),
            "s_full-subagent:Explore",
            {"agent": "s_full_subagent", "agent_type": "Explore"},
            "Subagent response.",
        )
        self.assert_traced_call(
            "agents.s_full",
            "call_teammate_model",
            (([{"role": "user", "content": "Continue work"}]), "sys", [], "dana"),
            "s_full-teammate:dana",
            {"agent": "s_full_teammate", "teammate": "dana"},
            "Teammate response.",
        )
        self.assert_traced_call(
            "agents.s_full",
            "call_compactor_model",
            ("long chat",),
            "s_full-auto-compact",
            {"agent": "s_full_compactor"},
            "Compact summary.",
        )


if __name__ == "__main__":
    unittest.main()
