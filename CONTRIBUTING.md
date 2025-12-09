# Contributing to Hypothesis Forge

Thank you for your interest in contributing to Hypothesis Forge! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/Hypothesis-Forge.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Install dev dependencies: `pip install -e ".[dev]"`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Write or update tests
4. Ensure all tests pass: `pytest tests/ -v`
5. Check code style: `flake8 .` and `black . --check`
6. Commit your changes: `git commit -m "Add feature: description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a pull request

## Code Style

- Follow PEP 8
- Use Black for formatting (max line length 88)
- Use type hints where possible
- Write docstrings for all public functions and classes
- Keep functions focused and small

## Testing

- Write tests for all new features
- Aim for >80% code coverage
- Run tests before submitting: `pytest tests/ -v --cov=.`
- Include both unit and integration tests

## Pull Request Process

1. Update README.md if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Request review from maintainers

## Questions?

Open an issue or contact the maintainers.

