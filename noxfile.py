"""Nox configuration file for automation of testing, linting, typing and security checks."""

import nox

SOURCE_FILES = ("abba/", "tests/")

REPORTS_DIR = "reports"
BADGES_DIR = ".badges"


@nox.session(python=["3.11"])
def tests(session):
    """Run the test suite."""
    session.install("poetry")
    session.run("poetry", "install", "--no-interaction")
    session.run(
        "poetry",
        "run",
        "coverage",
        "run",
        "-m",
        "pytest",
        f"--junit-xml={REPORTS_DIR}/junit/junit.xml",
    )
    session.run("poetry", "run", "coverage", "report")
    session.run("poetry", "run", "coverage", "xml", "-o", f"{REPORTS_DIR}/coverage.xml")
    session.run("poetry", "run", "coverage", "html", "-d", f"{REPORTS_DIR}/coverage")
    session.run(
        "poetry",
        "run",
        "genbadge",
        "coverage",
        "-i",
        f"{REPORTS_DIR}/coverage.xml",
        "-o",
        f"{BADGES_DIR}/coverage-badge.svg",
    )
    session.run("poetry", "run", "genbadge", "tests", "-o", f"{BADGES_DIR}/tests-badge.svg")


@nox.session(python=["3.11"])
def lint(session):
    """Lint the codebase with black, isort, flake8, and pylint."""
    session.install("poetry")
    session.run("poetry", "install", "--no-interaction")

    # Check formatting with black (line length 120)
    session.run("poetry", "run", "black", "--check", "--line-length", "120", *SOURCE_FILES)

    # Check import sorting with isort (black-compatible profile)
    session.run("poetry", "run", "isort", "--check-only", "--profile", "black", "--line-length", "120", *SOURCE_FILES)

    # Run flake8 for style enforcement
    session.run("poetry", "run", "flake8", *SOURCE_FILES, "--config", ".flake8")

    # Run pylint for code analysis
    session.run("poetry", "run", "pylint", *SOURCE_FILES)


@nox.session(python=["3.11"])
def typing(session):
    """Run the type checker with mypy."""
    session.install("poetry")
    session.run("poetry", "install", "--no-interaction")
    session.run("poetry", "run", "mypy", *SOURCE_FILES)


@nox.session(python=["3.11"])
def security(session):
    """Run the security checks with bandit."""
    session.install("poetry")
    session.run("poetry", "install", "--no-interaction")
    session.run("poetry", "run", "bandit", "-c", "bandit.yml", "-r", "abba/")


@nox.session(python=["3.11"])
def format(session):
    """Auto-format code with black and isort."""
    session.install("poetry")
    session.run("poetry", "install", "--no-interaction")
    session.run("poetry", "run", "black", "--line-length", "120", *SOURCE_FILES)
    session.run("poetry", "run", "isort", "--profile", "black", "--line-length", "120", *SOURCE_FILES)
