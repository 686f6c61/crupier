import subprocess
import sys
from pathlib import Path


def test_sdk_dry_run_example_runs_without_provider_keys(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "sdk_dry_run.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "strategy=" in result.stdout
    assert "models=" in result.stdout
    assert "summary=" in result.stdout
    assert not (tmp_path / ".crupier").exists()
