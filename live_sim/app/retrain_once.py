"""Manual one-shot retrain command for Makefile/Docker use."""

from __future__ import annotations

import json
import sys

from .bot import now_iso
from .config import load_config
from .scheduler import RetrainScheduler
from .store import Store


def main() -> int:
    cfg = load_config()
    store = Store(cfg.db_path)
    store.initialize_account(cfg.starting_cash, now_iso())
    scheduler = RetrainScheduler(cfg, store)
    try:
        scheduler.run_now_sync("make_update_model")
    except Exception as exc:  # noqa: BLE001 - command-line error path.
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "retraining": scheduler.status_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
