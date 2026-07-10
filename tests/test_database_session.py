from sqlalchemy.exc import OperationalError

from src.database.session import run_db_operation_with_retry


def test_run_db_operation_with_retry_retries_transient_lock() -> None:
    attempts = {"count": 0}

    def flaky_operation() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OperationalError("SELECT 1", None, Exception("database is locked"))
        return "ok"

    result = run_db_operation_with_retry(flaky_operation, max_retries=2, delay_seconds=0)

    assert result == "ok"
    assert attempts["count"] == 2
