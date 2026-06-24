"""Nox configuration for automation of testing, linting, typing and security checks.

Sessions run inside the uv-managed project environment (``uv sync`` + ``uv run``)
rather than building their own virtualenvs, so the toolchain matches local runs.
"""

import nox

SOURCE_FILES = ("abba/", "tests/")

REPORTS_DIR = "reports"
BADGES_DIR = ".badges"


@nox.session(python=False)
def tests(session):
    """Run the test suite with coverage and refresh the badges."""
    session.run("uv", "sync", external=True)
    session.run(
        "uv",
        "run",
        "coverage",
        "run",
        "-m",
        "pytest",
        f"--junit-xml={REPORTS_DIR}/junit/junit.xml",
        external=True,
    )
    session.run("uv", "run", "coverage", "report", external=True)
    session.run("uv", "run", "coverage", "xml", "-o", f"{REPORTS_DIR}/coverage.xml", external=True)
    session.run("uv", "run", "coverage", "html", "-d", f"{REPORTS_DIR}/coverage", external=True)
    session.run(
        "uv",
        "run",
        "genbadge",
        "coverage",
        "-i",
        f"{REPORTS_DIR}/coverage.xml",
        "-o",
        f"{BADGES_DIR}/coverage-badge.svg",
        external=True,
    )
    session.run("uv", "run", "genbadge", "tests", "-o", f"{BADGES_DIR}/tests-badge.svg", external=True)


@nox.session(python=False)
def lint(session):
    """Lint the codebase with ruff (format check + lint, replaces black/isort/flake8/pylint)."""
    session.run("uv", "sync", external=True)
    # Verify formatting (ruff format replaces black + isort).
    session.run("uv", "run", "ruff", "format", "--check", *SOURCE_FILES, external=True)
    # Lint (replaces flake8 + pylint + bandit; rules configured in pyproject.toml).
    session.run("uv", "run", "ruff", "check", *SOURCE_FILES, external=True)


@nox.session(python=False)
def typing(session):
    """Run the type checker with pyright (replaces mypy)."""
    session.run("uv", "sync", external=True)
    session.run("uv", "run", "pyright", "abba/", external=True)


@nox.session(python=False)
def security(session):
    """Run the security checks with ruff's flake8-bandit (S) rules (replaces bandit).

    S101 (assert) and S608 (parameterized SQL built via f-strings) are accepted
    project-wide; case-by-case exceptions use inline ``# noqa: S...`` comments.
    """
    session.run("uv", "sync", external=True)
    session.run("uv", "run", "ruff", "check", "--select", "S", "--ignore", "S101,S608", "abba/", external=True)


@nox.session(python=False)
def format(session):
    """Auto-format code and apply safe lint fixes with ruff."""
    session.run("uv", "sync", external=True)
    session.run("uv", "run", "ruff", "format", *SOURCE_FILES, external=True)
    session.run("uv", "run", "ruff", "check", "--fix", *SOURCE_FILES, external=True)
