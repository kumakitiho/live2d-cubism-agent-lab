from __future__ import annotations

import argparse
from pathlib import Path

from tools.action_plan import load_action_plan, validate_action_plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Cubism action plan")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    try:
        plan = load_action_plan(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"NG: {exc}")
        return 1

    issues = validate_action_plan(plan)
    if issues:
        print("NG: action plan is invalid")
        for issue in issues:
            print(f"- {issue.format()}")
        return 1

    print("OK: action plan is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
