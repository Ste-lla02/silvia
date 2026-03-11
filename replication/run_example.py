import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd: Path):
    """Run a command and fail fast with a clear error message."""
    print("\n>>> Running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(cwd), text=True)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the SoftwareX replication example end-to-end."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/configuration.ini",
        help="Path to the example configuration file (relative to replication/).",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default="..",
        help="Path to the project root (where your original main script lives).",
    )
    parser.add_argument(
        "--entrypoint",
        type=str,
        default="core_functions.py",
        help="Entrypoint script (relative to project root).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="build,fusion",
        help="Comma-separated steps to run. Default: clean,build,fusion",
    )
    args = parser.parse_args()

    replication_dir = Path(__file__).resolve().parent
    project_root = (replication_dir / args.project_root).resolve()

    config_path = (replication_dir / args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    entrypoint_path = (project_root / args.entrypoint).resolve()
    if not entrypoint_path.exists():
        raise SystemExit(f"Entrypoint not found: {entrypoint_path}")

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if not steps:
        raise SystemExit("No steps selected. Use --steps clean,build,...")

    for step in steps:
        run([sys.executable, str(entrypoint_path), str(config_path), step], cwd=project_root)

    print("\nReplication run completed.")
    print("Check outputs/logs as specified in config (ideally under replication/output/).")


if __name__ == "__main__":
    main()
