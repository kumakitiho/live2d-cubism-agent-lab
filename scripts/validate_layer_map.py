from __future__ import annotations

import argparse
from pathlib import Path

from tools.artifact_validation import load_yaml_mapping, validate_layer_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate layer_map.yaml")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    try:
        data = load_yaml_mapping(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"NG: {exc}")
        return 1

    issues = validate_layer_map(data)
    if issues:
        print("NG: layer map is invalid")
        for issue in issues:
            print(f"- {issue.format()}")
        return 1
    print("OK: layer map is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
