# Contributing to Agentic AI Framework

First off, thank you for considering contributing to the Agentic AI Framework! It's people like you that make this project such a great tool.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and expected**
- **Include Python version and dependency versions**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Provide specific examples to demonstrate the enhancement**
- **Describe the current behavior and expected behavior**
- **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the style guidelines
6. Issue that pull request!

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/agentic-community.git
cd agentic-community
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

## Style Guidelines

### Python Style

We use [Black](https://black.readthedocs.io/) for code formatting and [isort](https://pycqa.github.io/isort/) for import sorting:

```bash
# Format code
black .

# Sort imports
isort .

# Check style
flake8 . --max-line-length=88 --extend-ignore=E203
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### Documentation

- Use docstrings for all public functions, classes, and modules
- Follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings
- Update the README.md if you change functionality

## Testing

We use pytest for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentic_community

# Run specific test file
pytest tests/test_agents.py
```

### Writing Tests

- Write tests for any new functionality
- Ensure tests are independent and can run in any order
- Use descriptive test names that explain what is being tested
- Mock external dependencies (like API calls)

## Project Structure

```
agentic_community/
├── agents/          # Agent implementations
├── tools/           # Tool implementations
├── core/            # Core components
│   ├── base/       # Base classes
│   ├── reasoning/  # Reasoning engine
│   └── state/      # State management
├── examples/        # Example scripts
└── tests/          # Test files
```

## Making Changes

1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes**: Follow the style guidelines
3. **Add tests**: Ensure your changes are tested
4. **Run tests**: `pytest`
5. **Check style**: `black . && isort . && flake8 .`
6. **Commit changes**: Use clear commit messages
7. **Push to your fork**: `git push origin feature/your-feature-name`
8. **Create Pull Request**: Use the PR template

## Community Edition vs Enterprise Edition

Please note that some features are reserved for the Enterprise Edition:
- Advanced reasoning capabilities
- Multi-agent orchestration
- Additional LLM providers
- Enterprise integrations

If your contribution adds enterprise-level features, we may need to discuss how to handle it appropriately.

## Questions?

Feel free to ask questions in:
- [GitHub Discussions](https://github.com/zahere/agentic-community/discussions)
- [Discord Server](https://discord.gg/agentic)
- [GitHub Issues](https://github.com/zahere/agentic-community/issues)

## Recognition

Contributors will be recognized in our:
- README.md contributors section
- Release notes
- Project website

Thank you for contributing to the Agentic AI Framework! 🎉
