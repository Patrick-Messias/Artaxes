# ═══════════════════════════════════════════════════════════════════════════
# Backend/api/deps.py — Database dependency injection
# ═══════════════════════════════════════════════════════════════════════════
from db.Database import Database # type: ignore

_db : Database | None = None

def get_db() -> Database:
    # Returns the global Database instance, one connection per process
    global _db
    if _db is None:
        _db = Database()
    return _db















