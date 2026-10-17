import sys

from sqlalchemy import text

from config.database import get_engine


def test_python_version():
    assert sys.version_info[:2] == (3, 11)


def test_database_connection():
    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT 1")
        ).scalar_one()

    assert result == 1


def test_database_identity():
    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

    assert result == "retailpulse_db"


def test_application_role():
    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_user")
        ).scalar_one()

    assert result == "retailpulse_app"


def test_application_not_superuser():
    engine = get_engine()

    with engine.connect() as connection:
        result = connection.execute(
            text(
                """
                SELECT rolsuper
                FROM pg_roles
                WHERE rolname = current_user
                """
            )
        ).scalar_one()

    assert result is False