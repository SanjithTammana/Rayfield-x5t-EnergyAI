from sqlalchemy import create_engine, text
from datetime import datetime
from ..settings import DB_PATH

_engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

def log_event(user: str, action: str, meta: dict | None = None):
    with _engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE IF NOT EXISTS events (ts TEXT, user TEXT, action TEXT, meta TEXT)")
        )
        conn.execute(
            text("INSERT INTO events (ts, user, action, meta) VALUES (:ts,:u,:a,:m)"),
            {"ts": datetime.utcnow().isoformat(), "u": user, "a": action, "m": str(meta)},
        )