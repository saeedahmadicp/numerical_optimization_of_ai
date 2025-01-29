# Contributing Guidelines

Thank you for considering contributing to our Numerical Optimization repository! This document outlines the process and guidelines for contributing.

## Ways to Contribute

You can contribute in several ways:
- Implementing new optimization algorithms
- Improving existing implementations
- Adding examples and use cases
- Enhancing documentation
- Fixing bugs
- Adding tests
- Suggesting improvements

## Getting Started

1. **Fork the Repository**
   ```bash
   # Clone your fork
   git clone https://github.com/your-username/repository-name.git
   cd repository-name

   # Add upstream remote
   git remote add upstream https://github.com/original-owner/repository-name.git
   ```

2. **Create a Branch**
   ```bash
   # Create and switch to a new branch
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style
- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Include type hints where appropriate

### Documentation
Each contribution should include:
- Clear docstrings explaining functionality
- Mathematical background (if applicable)
- Usage examples
- References to relevant papers/resources

### Implementation Requirements
- Write clean, readable code
- Include necessary tests
- Provide performance benchmarks (if applicable)
- Add example usage in docstrings

### Testing
- Add unit tests for new functionality
- Ensure all tests pass before submitting
- Include test cases with edge conditions

## Submitting Changes

1. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

2. **Update Your Fork**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

3. **Push Changes**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**
   - Go to the repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template

## Pull Request Guidelines

Your PR should include:
- Clear description of changes
- Reference to any related issues
- List of new dependencies (if any)
- Updates to documentation
- Test results

## Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Best Practices

- Keep PRs focused and reasonably sized
- Write clear commit messages
- Update documentation as needed
- Test thoroughly before submitting
- Be responsive to feedback

## Questions?

If you have questions, feel free to:
- Open an issue
- Ask in the discussions section
- Contact the maintainers

---

Thank you for contributing to making numerical optimization more accessible to everyone! 🚀