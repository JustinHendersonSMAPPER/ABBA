# Code Quality Standards for ABBA Project

## Overview

The ABBA project maintains high code quality standards with comprehensive type hints and strict linting. This document outlines the standards and tools used.

## Type Checking

### mypy Configuration
- **Strict mode enabled**: All functions must have type annotations
- **No implicit Any**: All types must be explicit
- **No untyped definitions**: Every function requires return type annotation

### Type Annotation Guidelines
1. **All function parameters** must have type hints
2. **All function returns** must be annotated (use `-> None` for procedures)
3. **Class attributes** should have type annotations
4. **Complex types** use typing module imports (Dict, List, Optional, Union, etc.)

Example:
```python
from typing import Dict, List, Optional, Union

def parse_verse_id(verse_str: str) -> Optional[VerseID]:
    """Parse a verse string into a VerseID object."""
    # Implementation
```

## Linting Standards

### Tools Used
1. **black** - Code formatting (line length 100)
2. **isort** - Import sorting
3. **flake8** - Style guide enforcement
4. **pylint** - Code analysis
5. **mypy** - Static type checking

### Key Standards
- **Line length**: Maximum 100 characters
- **Import order**: Standard library, third-party, local
- **Docstrings**: Required for all public functions and classes
- **Type hints**: Required for all functions
- **No trailing whitespace**
- **Files end with newline**

## Running Quality Checks

### Individual Tools
```bash
# Format code
poetry run black src/

# Sort imports
poetry run isort src/

# Check style
poetry run flake8 src/

# Run type checker
poetry run mypy src/

# Run comprehensive linter
poetry run pylint src/
```

### Combined Check
```bash
# Run all checks
make check

# Or individually
make format  # black + isort
make lint    # flake8 + pylint + mypy
```

## Common Issues Fixed

### 1. Type Annotations
- Added `-> None` for procedures
- Fixed `any` to `Any` (proper casing)
- Added type annotations for all function parameters
- Specified return types for all functions

### 2. Import Organization
- Removed unused imports
- Sorted imports properly
- Fixed import order

### 3. Code Style
- Fixed trailing whitespace
- Added newlines at end of files
- Consistent indentation
- Proper spacing around operators

### 4. Type Safety
- Fixed comparison operators in dataclasses
- Added proper type narrowing
- Handled Optional types correctly
- Fixed Union type handling

## Test Coverage

### Current Status
- **100% test coverage** with 1106 tests passing
- All modules have comprehensive test suites
- Tests cover both positive and negative cases
- Integration tests verify component interactions

### Running Tests
```bash
# Run all tests
make test

# Run with coverage report
make test-coverage

# Run specific test module
poetry run pytest tests/test_morphology.py -v

# Run tests in parallel
poetry run pytest -n auto

# Generate HTML coverage report
poetry run pytest --cov=abba --cov-report=html
open htmlcov/index.html
```

### Test Organization
- Unit tests for each module in `tests/`
- Test files mirror source structure
- Fixtures for common test data
- Mocking for external dependencies

### Writing Tests
```python
import pytest
from abba.verse_id import parse_verse_id

def test_parse_valid_verse_id():
    """Test parsing a valid verse ID."""
    verse = parse_verse_id("ROM.3.23")
    assert verse.book == "ROM"
    assert verse.chapter == 3
    assert verse.verse == 23

def test_parse_invalid_verse_id():
    """Test parsing an invalid verse ID."""
    verse = parse_verse_id("INVALID")
    assert verse is None
```

## Dataclass Best Practices

For sortable dataclasses like `VerseID`:
```python
@dataclass
class VerseID:
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VerseID):
            return False
        # comparison logic
    
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, VerseID):
            return NotImplemented
        # comparison logic
    
    # Implement all comparison operators for full ordering
    def __le__(self, other: object) -> bool:
    def __gt__(self, other: object) -> bool:
    def __ge__(self, other: object) -> bool:
```

## Function Documentation

All public functions must have:
1. **Brief description** in the first line
2. **Args section** documenting parameters
3. **Returns section** documenting return value
4. **Optional**: Examples, Raises, Notes sections

Example:
```python
def normalize_book_name(book_name: str) -> Optional[str]:
    """
    Normalize a book name to its canonical 3-letter code.
    
    Args:
        book_name: Book name in any common format
        
    Returns:
        Canonical 3-letter book code, or None if not found
    """
```

## Continuous Quality

### Pre-commit Hooks
The project uses pre-commit hooks to ensure quality:
```bash
# Install hooks
poetry run pre-commit install

# Run manually
poetry run pre-commit run --all-files
```

### CI/CD Integration
All pull requests must pass:
- Type checking (mypy --strict)
- Linting (flake8, pylint)
- Code formatting (black --check)
- Import sorting (isort --check)

## Benefits

1. **Type Safety**: Catch errors before runtime
2. **Consistency**: Uniform code style across the project
3. **Maintainability**: Clear interfaces and documentation
4. **IDE Support**: Better autocomplete and error detection
5. **Refactoring Safety**: Types help prevent breaking changes

## Future Improvements

1. **Coverage Requirements**: Minimum 90% test coverage
2. **Complexity Limits**: Monitor and reduce cyclomatic complexity
3. **Security Scanning**: Add bandit for security checks
4. **Documentation Coverage**: Ensure all public APIs are documented

By maintaining these standards, the ABBA project ensures high-quality, maintainable code that's easy to understand and extend.