"""SQLite persistence for the live paper-trading simulator."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
        self.init_schema()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def init_schema(self) -> None:
        with self.lock, self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS account_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    total_fees REAL NOT NULL,
                    initialized_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS open_position (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    entry_time TEXT NOT NULL,
                    entry_candle_open_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    investment REAL NOT NULL,
                    entry_fee REAL NOT NULL,
                    entry_prob_up REAL NOT NULL,
                    entry_bid REAL NOT NULL,
                    entry_ask REAL NOT NULL,
                    side TEXT NOT NULL DEFAULT 'long'
                );

                CREATE TABLE IF NOT EXISTS account_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    cash REAL NOT NULL,
                    sol_qty REAL NOT NULL,
                    equity REAL NOT NULL,
                    position_status TEXT NOT NULL,
                    entry_price REAL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    total_fees REAL NOT NULL,
                    last_price REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS candle_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    open_time TEXT NOT NULL UNIQUE,
                    close_time TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS book_ticker_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    spread REAL NOT NULL,
                    spread_pct REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    candle_open_time TEXT NOT NULL UNIQUE,
                    prob_up REAL NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    entry_threshold REAL NOT NULL,
                    exit_threshold REAL NOT NULL,
                    cash REAL NOT NULL,
                    sol_qty REAL NOT NULL,
                    equity REAL NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    spread_pct REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    investment REAL NOT NULL,
                    gross_exit_value REAL NOT NULL,
                    entry_fee REAL NOT NULL,
                    exit_fee REAL NOT NULL,
                    net_profit REAL NOT NULL,
                    gross_return REAL NOT NULL,
                    bars_held INTEGER NOT NULL,
                    exit_reason TEXT NOT NULL,
                    side TEXT NOT NULL DEFAULT 'long',
                    borrow_fee REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS server_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    message TEXT NOT NULL,
                    active_model_updated INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS real_trade_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    armed INTEGER NOT NULL DEFAULT 0,
                    armed_at TEXT,
                    bot_sol_qty REAL NOT NULL DEFAULT 0,
                    bot_cost_usd REAL NOT NULL DEFAULT 0,
                    realized_pnl_usd REAL NOT NULL DEFAULT 0,
                    total_fees_usd REAL NOT NULL DEFAULT 0,
                    last_error TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS real_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    candle_open_time TEXT,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    client_order_id TEXT,
                    coinbase_order_id TEXT,
                    requested_usd REAL NOT NULL DEFAULT 0,
                    requested_sol REAL NOT NULL DEFAULT 0,
                    filled_usd REAL NOT NULL DEFAULT 0,
                    filled_sol REAL NOT NULL DEFAULT 0,
                    average_price REAL,
                    fee_usd REAL NOT NULL DEFAULT 0,
                    reason TEXT,
                    error TEXT,
                    raw_response TEXT
                );
                """
            )
            self._ensure_column("open_position", "side", "TEXT NOT NULL DEFAULT 'long'")
            self._ensure_column("trades", "side", "TEXT NOT NULL DEFAULT 'long'")
            self._ensure_column("trades", "borrow_fee", "REAL NOT NULL DEFAULT 0")
            self.conn.execute(
                """
                INSERT OR IGNORE INTO real_trade_state (
                    id, armed, bot_sol_qty, bot_cost_usd, realized_pnl_usd,
                    total_fees_usd, updated_at
                ) VALUES (1, 0, 0, 0, 0, 0, datetime('now'))
                """
            )

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        columns = {str(row["name"]) for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def initialize_account(self, starting_cash: float, now: str) -> None:
        with self.lock, self.conn:
            row = self.conn.execute("SELECT id FROM account_state WHERE id = 1").fetchone()
            if row is None:
                self.conn.execute(
                    """
                    INSERT INTO account_state (id, cash, realized_pnl, total_fees, initialized_at, updated_at)
                    VALUES (1, ?, 0, 0, ?, ?)
                    """,
                    (starting_cash, now, now),
                )

    def reset_all(self, starting_cash: float, now: str) -> None:
        with self.lock, self.conn:
            for table in [
                "account_state",
                "open_position",
                "account_snapshots",
                "candle_snapshots",
                "book_ticker_ticks",
                "model_decisions",
                "trades",
                "server_events",
                "training_runs",
                "real_trade_state",
                "real_orders",
            ]:
                self.conn.execute(f"DELETE FROM {table}")
            self.conn.execute(
                """
                INSERT INTO account_state (id, cash, realized_pnl, total_fees, initialized_at, updated_at)
                VALUES (1, ?, 0, 0, ?, ?)
                """,
                (starting_cash, now, now),
            )
            self.conn.execute(
                """
                INSERT INTO real_trade_state (
                    id, armed, bot_sol_qty, bot_cost_usd, realized_pnl_usd,
                    total_fees_usd, updated_at
                ) VALUES (1, 0, 0, 0, 0, 0, ?)
                """,
                (now,),
            )

    def account_state(self) -> dict[str, Any]:
        with self.lock:
            row = self.conn.execute("SELECT * FROM account_state WHERE id = 1").fetchone()
            if row is None:
                raise RuntimeError("Account state has not been initialized")
            return dict(row)

    def update_account_state(self, *, cash: float, realized_pnl: float, total_fees: float, updated_at: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                UPDATE account_state
                SET cash = ?, realized_pnl = ?, total_fees = ?, updated_at = ?
                WHERE id = 1
                """,
                (cash, realized_pnl, total_fees, updated_at),
            )

    def open_position(self) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM open_position WHERE id = 1").fetchone()
            return dict(row) if row is not None else None

    def set_open_position(self, position: dict[str, Any]) -> None:
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM open_position WHERE id = 1")
            self.conn.execute(
                """
                INSERT INTO open_position (
                    id, entry_time, entry_candle_open_time, entry_price, quantity, investment,
                    entry_fee, entry_prob_up, entry_bid, entry_ask, side
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position["entry_time"],
                    position["entry_candle_open_time"],
                    position["entry_price"],
                    position["quantity"],
                    position["investment"],
                    position["entry_fee"],
                    position["entry_prob_up"],
                    position["entry_bid"],
                    position["entry_ask"],
                    position.get("side", "long"),
                ),
            )

    def clear_open_position(self) -> None:
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM open_position WHERE id = 1")

    def insert_candles(self, candles: list[Any]) -> None:
        with self.lock, self.conn:
            self.conn.executemany(
                """
                INSERT OR IGNORE INTO candle_snapshots
                    (open_time, close_time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (c.open_time, c.close_time, c.open, c.high, c.low, c.close, c.volume)
                    for c in candles
                ],
            )

    def insert_ticker(self, ticker: Any) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO book_ticker_ticks (ts, bid, ask, spread, spread_pct)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ticker.ts, ticker.bid, ticker.ask, ticker.spread, ticker.spread_pct),
            )

    def insert_decision(self, decision: dict[str, Any]) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO model_decisions (
                    ts, candle_open_time, prob_up, action, reason, entry_threshold, exit_threshold,
                    cash, sol_qty, equity, bid, ask, spread_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision["ts"],
                    decision["candle_open_time"],
                    decision["prob_up"],
                    decision["action"],
                    decision["reason"],
                    decision["entry_threshold"],
                    decision["exit_threshold"],
                    decision["cash"],
                    decision["sol_qty"],
                    decision["equity"],
                    decision["bid"],
                    decision["ask"],
                    decision["spread_pct"],
                ),
            )

    def decision_exists(self, candle_open_time: str) -> bool:
        with self.lock:
            row = self.conn.execute(
                "SELECT 1 FROM model_decisions WHERE candle_open_time = ?",
                (candle_open_time,),
            ).fetchone()
            return row is not None

    def insert_trade(self, trade: dict[str, Any]) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO trades (
                    entry_time, exit_time, entry_price, exit_price, quantity, investment,
                    gross_exit_value, entry_fee, exit_fee, net_profit, gross_return, bars_held, exit_reason,
                    side, borrow_fee
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade["entry_time"],
                    trade["exit_time"],
                    trade["entry_price"],
                    trade["exit_price"],
                    trade["quantity"],
                    trade["investment"],
                    trade["gross_exit_value"],
                    trade["entry_fee"],
                    trade["exit_fee"],
                    trade["net_profit"],
                    trade["gross_return"],
                    trade["bars_held"],
                    trade["exit_reason"],
                    trade.get("side", "long"),
                    trade.get("borrow_fee", 0.0),
                ),
            )

    def insert_account_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO account_snapshots (
                    ts, cash, sol_qty, equity, position_status, entry_price, unrealized_pnl,
                    realized_pnl, total_fees, last_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot["ts"],
                    snapshot["cash"],
                    snapshot["sol_qty"],
                    snapshot["equity"],
                    snapshot["position_status"],
                    snapshot.get("entry_price"),
                    snapshot["unrealized_pnl"],
                    snapshot["realized_pnl"],
                    snapshot["total_fees"],
                    snapshot["last_price"],
                ),
            )

    def insert_event(self, ts: str, level: str, message: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO server_events (ts, level, message) VALUES (?, ?, ?)",
                (ts, level, message),
            )

    def real_state(self) -> dict[str, Any]:
        with self.lock:
            row = self.conn.execute("SELECT * FROM real_trade_state WHERE id = 1").fetchone()
            if row is None:
                with self.conn:
                    self.conn.execute(
                        """
                        INSERT INTO real_trade_state (
                            id, armed, bot_sol_qty, bot_cost_usd, realized_pnl_usd,
                            total_fees_usd, updated_at
                        ) VALUES (1, 0, 0, 0, 0, 0, datetime('now'))
                        """
                    )
                row = self.conn.execute("SELECT * FROM real_trade_state WHERE id = 1").fetchone()
            return dict(row)

    def set_real_armed(self, *, armed: bool, ts: str, error: str | None = None) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                UPDATE real_trade_state
                SET armed = ?, armed_at = CASE WHEN ? THEN ? ELSE armed_at END,
                    last_error = ?, updated_at = ?
                WHERE id = 1
                """,
                (int(armed), int(armed), ts, error, ts),
            )

    def update_real_position(
        self,
        *,
        bot_sol_qty: float,
        bot_cost_usd: float,
        realized_pnl_usd: float,
        total_fees_usd: float,
        ts: str,
        error: str | None = None,
    ) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                UPDATE real_trade_state
                SET bot_sol_qty = ?, bot_cost_usd = ?, realized_pnl_usd = ?,
                    total_fees_usd = ?, last_error = ?, updated_at = ?
                WHERE id = 1
                """,
                (bot_sol_qty, bot_cost_usd, realized_pnl_usd, total_fees_usd, error, ts),
            )

    def insert_real_order(self, order: dict[str, Any]) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO real_orders (
                    ts, candle_open_time, action, status, product_id, side,
                    client_order_id, coinbase_order_id, requested_usd, requested_sol,
                    filled_usd, filled_sol, average_price, fee_usd, reason, error, raw_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order["ts"],
                    order.get("candle_open_time"),
                    order["action"],
                    order["status"],
                    order["product_id"],
                    order["side"],
                    order.get("client_order_id"),
                    order.get("coinbase_order_id"),
                    order.get("requested_usd", 0.0),
                    order.get("requested_sol", 0.0),
                    order.get("filled_usd", 0.0),
                    order.get("filled_sol", 0.0),
                    order.get("average_price"),
                    order.get("fee_usd", 0.0),
                    order.get("reason"),
                    order.get("error"),
                    order.get("raw_response"),
                ),
            )

    def latest_real_order(self) -> dict[str, Any] | None:
        return self.latest_row("real_orders")

    def recent_real_orders(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.recent_rows("real_orders", limit)

    def insert_training_run(
        self,
        *,
        run_id: str,
        started_at: str,
        status: str,
        run_dir: str,
        message: str,
        active_model_updated: bool,
    ) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO training_runs (
                    run_id, started_at, status, run_dir, message, active_model_updated
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, started_at, status, run_dir, message, int(active_model_updated)),
            )

    def finish_training_run(
        self,
        *,
        run_id: str,
        ended_at: str,
        status: str,
        message: str,
        active_model_updated: bool,
        metadata: str | None,
    ) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                UPDATE training_runs
                SET ended_at = ?, status = ?, message = ?, active_model_updated = ?, metadata = ?
                WHERE run_id = ?
                """,
                (ended_at, status, message, int(active_model_updated), metadata, run_id),
            )

    def latest_account_snapshot(self) -> dict[str, Any] | None:
        return self.latest_row("account_snapshots")

    def latest_decision(self) -> dict[str, Any] | None:
        return self.latest_row("model_decisions")

    def latest_ticker(self) -> dict[str, Any] | None:
        return self.latest_row("book_ticker_ticks")

    def latest_training_run(self) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM training_runs ORDER BY started_at DESC LIMIT 1").fetchone()
            return dict(row) if row is not None else None

    def recent_training_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM training_runs ORDER BY started_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def latest_row(self, table: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT 1").fetchone()
            return dict(row) if row is not None else None

    def recent_rows(self, table: str, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 5000))
        with self.lock:
            rows = self.conn.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (safe_limit,)).fetchall()
            return [dict(row) for row in reversed(rows)]
