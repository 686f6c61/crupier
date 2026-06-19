import subprocess
import sys
from pathlib import Path


def test_public_examples_run_without_provider_keys(tmp_path):
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    scripts = sorted(path for path in examples_dir.glob("*.py") if not path.name.startswith("_"))

    assert scripts

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, f"{script.name}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert "strategy=" in result.stdout, script.name
        assert "models=" in result.stdout, script.name
        assert not (tmp_path / ".crupier").exists(), script.name
