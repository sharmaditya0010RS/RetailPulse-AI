import importlib
import platform
import sys
from pathlib import Path

from sqlalchemy import text

from config.database import check_database_connection, get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verify_python():
    print("\n[1] Python Environment")
    print("Python:", platform.python_version())

    assert sys.version_info[:2] == (3, 11), (
        "Python 3.11 is required."
    )

    assert sys.prefix != sys.base_prefix, (
        "Virtual environment is not active."
    )

    print("PASS: Python environment")


def verify_packages():
    print("\n[2] Required Packages")

    packages = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "sqlalchemy",
        "psycopg",
        "dotenv",
        "streamlit",
        "plotly",
        "lightgbm",
        "catboost",
        "shap",
        "mlflow",
        "optuna",
        "pytest",
    ]

    for package in packages:
        importlib.import_module(package)
        print(f"PASS: {package}")


def verify_directories():
    print("\n[3] Project Directories")

    directories = [
        "config",
        "scripts",
        "tests",
        "sql",
        "data",
        "src",
    ]

    for directory in directories:
        path = PROJECT_ROOT / directory

        if not path.is_dir():
            raise RuntimeError(
                f"Missing directory: {directory}"
            )

        print(f"PASS: {directory}")


def verify_database():
    print("\n[4] PostgreSQL Connection")

    result = check_database_connection()

    print("Database:", result["database_name"])
    print("User:", result["database_user"])
    print(
        "PostgreSQL:",
        result["postgres_version"].split(",")[0],
    )

    assert result["database_name"] == "retailpulse_db"
    assert result["database_user"] == "retailpulse_app"

    engine = get_engine()

    with engine.connect() as connection:
        value = connection.execute(
            text("SELECT 1")
        ).scalar_one()

    assert value == 1

    print("PASS: Database connectivity")


def main():
    print("=" * 55)
    print("RETAILPULSE AI - PHASE 0 VERIFICATION")
    print("=" * 55)

    verify_python()
    verify_packages()
    verify_directories()
    verify_database()

    print("\n" + "=" * 55)
    print("PHASE 0 VERIFICATION SUCCESSFUL")
    print("=" * 55)


if __name__ == "__main__":
    main()