"""Nox configuration file for automation of testing, linting, typing and security checks."""

import nox

SOURCE_FILES = ("src/", "tests/")

REPORTS_DIR = "reports"
BAGDES_DIR = ".badges"


@nox.session(python=["3.11"])
def tests(session):
    """Run the test suite."""
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "shell")
    session.run("coverage", "run", "-m", "pytest", "--junit-xml=reports/junit/junit.xml")
    session.run("coverage", "combine")
    session.run("coverage", "report")
    session.run("coverage", "xml", "-o", f"{REPORTS_DIR}/coverage.xml")
    session.run("coverage", "html", "-d", f"{REPORTS_DIR}/coverage")
    session.run(
        "genbadge",
        "coverage",
        "-i",
        f"{REPORTS_DIR}/coverage.xml",
        "-o",
        f"{BAGDES_DIR}/coverage-badge.svg",
    )
    session.run("genbadge", "tests", "-o", f"{BAGDES_DIR}/tests-badge.svg")


@nox.session(python=["3.11"])
def lint(session):
    """Lint the codebase."""
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "shell")
    session.run("pylint", *SOURCE_FILES, "--exit-zero")
    # session.run("black", "--check", ".")
    session.run("python", "-c", "import os; os.remove('reports/flake8/flake8stats.txt')")
    session.run(
        "flake8",
        ".",
        "--color",
        "never",
        "--config",
        ".flake8",
        "--exit-zero",
        "--statistics",
        f"--output-file={REPORTS_DIR}/flake8/flake8stats.txt",
    )
    session.run("genbadge", "flake8", "-o", f"{BAGDES_DIR}/flake8-badge.svg")
    session.run("flynt", *SOURCE_FILES)
    session.run("isort", *SOURCE_FILES)


@nox.session(python=["3.11"])
def typing(session):
    """Run the type checker."""
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "shell")
    session.run("mypy", ".")


@nox.session(python=["3.11"])
def security(session):
    """Run the security checks."""
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "shell")
    session.run("bandit", "--exit-zero", "-c", "bandit.yaml", "-r", *SOURCE_FILES)
