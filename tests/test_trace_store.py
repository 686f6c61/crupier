import json

from crupier import Crupier
from crupier.cli import main
from crupier.config import CrupierConfig, write_default_project


def make_config(tmp_path):
    config = CrupierConfig.from_dict(
        {
            "project": {"name": "trace-test", "default_profile": "agentic"},
            "providers": {"openai": {"enabled": True, "env_key": "OPENAI_API_KEY"}},
            "models": {"allow": ["openai:gpt-5.4-mini"]},
            "routing": {"default_strategy": "single"},
        }
    )
    config.root = tmp_path
    return config


def test_trace_store_is_opt_in(tmp_path):
    client = Crupier(make_config(tmp_path))

    client.deal("Plan this", dry_run=True, trace=False)

    assert client.traces.list() == []


def test_trace_store_metadata_does_not_store_prompt_response_or_secret(tmp_path):
    client = Crupier(make_config(tmp_path))
    fake_secret = "s" + "k-test-secret-value"

    result = client.deal(
        f"Plan this with {fake_secret}",
        input={"token": fake_secret},
        constraints={"store_trace": True},
        dry_run=True,
        trace=False,
    )

    refs = client.traces.list()
    assert len(refs) == 1
    record = client.traces.read(refs[0].trace_id)
    serialized = json.dumps(record)
    assert result.trace is None
    assert "task" not in record["request"]
    assert "input" not in record["request"]
    assert "output_text" not in record["result"]
    assert fake_secret not in serialized
    assert "[redacted]" in serialized
    assert record["replayable"] is False


def test_trace_store_replay_requires_prompt_storage(tmp_path):
    client = Crupier(make_config(tmp_path))
    original = client.deal(
        "Replay this exact route",
        constraints={"store_trace": True, "store_prompt": True, "store_response": True},
        dry_run=True,
        trace="summary",
    )
    trace_id = original.trace.trace_id

    record = client.traces.read(trace_id)
    replay = client.traces.replay(trace_id, client, dry_run=True, trace="summary")

    assert record["replayable"] is True
    assert record["request"]["task"] == "Replay this exact route"
    assert record["result"]["output_text"].startswith("Crupier dry-run planned")
    assert replay.route.strategy == original.route.strategy


def test_cli_trace_commands(tmp_path, capsys):
    write_default_project(tmp_path)
    assert main(["--project", str(tmp_path), "deal", "Trace me", "--store-prompt", "--store-response"]) == 0
    capsys.readouterr()

    assert main(["--project", str(tmp_path), "trace", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    trace_id = listed[0]["trace_id"]
    assert listed[0]["replayable"] is True

    assert main(["--project", str(tmp_path), "trace", "show", trace_id, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["request"]["task"] == "Trace me"

    assert main(["--project", str(tmp_path), "trace", "replay", trace_id, "--json"]) == 0
    replayed = json.loads(capsys.readouterr().out)
    assert replayed["route"]["strategy"] == shown["result"]["route"]["strategy"]

    assert main(["--project", str(tmp_path), "trace", "delete", trace_id]) == 0
    capsys.readouterr()
    assert main(["--project", str(tmp_path), "trace", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
