"""Allow `python -m support_agent ...` (delegates to the CLI)."""

from support_agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
