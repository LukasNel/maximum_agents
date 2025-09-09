# Publishing Guide for Maximum Agents

This guide explains how to publish the maximum-agents package to PyPI using the automated publishing scripts.

## Prerequisites

1. **Python 3.12+** installed
2. **Git** configured with your credentials
3. **PyPI account** (create at https://pypi.org/account/register/)
4. **TestPyPI account** (optional, for testing: https://test.pypi.org/account/register/)

## Setup

### 1. Install Publishing Dependencies

```bash
pip install -r publish_requirements.txt
```

### 2. Configure PyPI Credentials

Create a `.pypirc` file in your home directory:

```ini
[distutils]
index-servers = pypi testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

**To get API tokens:**
- Go to https://pypi.org/manage/account/token/ (for PyPI)
- Go to https://test.pypi.org/manage/account/token/ (for TestPyPI)
- Create a new token and copy it to your `.pypirc` file

### 3. Update GitHub Repository URL

Before publishing, update the GitHub repository URL in the publishing script or use the `--github-url` parameter:

```bash
python publish_to_pypi.py --github-url https://github.com/LukasNel/maximum_agents
```

## Publishing Workflow

### Quick Start

```bash
# Test publish to TestPyPI first
./publish.sh --test --version patch

# If successful, publish to PyPI
./publish.sh --version patch
```

### Detailed Steps

#### 1. Test Publishing (Recommended)

```bash
# Publish to TestPyPI for testing
python publish_to_pypi.py --test --version patch

# Test the published package
pip install --index-url https://test.pypi.org/simple/ maximum-agents
```

#### 2. Production Publishing

```bash
# Publish to PyPI
python publish_to_pypi.py --version patch
```

#### 3. Verify Publication

```bash
# Install from PyPI
pip install maximum-agents

# Verify installation
python -c "import maximum_agents; print(maximum_agents.__version__)"
```

## Version Management

### Version Types

- **patch**: Bug fixes (0.1.0 → 0.1.1)
- **minor**: New features (0.1.0 → 0.2.0)
- **major**: Breaking changes (0.1.0 → 1.0.0)

### Examples

```bash
# Bug fix release
./publish.sh --version patch

# New feature release
./publish.sh --version minor

# Major release
./publish.sh --version major
```

## Advanced Usage

### Dry Run

Test the publishing process without actually publishing:

```bash
python publish_to_pypi.py --dry-run --version patch
```

### Custom GitHub URL

```bash
python publish_to_pypi.py --github-url https://github.com/LukasNel/maximum_agents --version patch
```

### TestPyPI Only

```bash
python publish_to_pypi.py --test --version patch
```

## What the Script Does

1. **Prerequisites Check**: Verifies required tools are installed
2. **Version Bumping**: Updates version in `pyproject.toml`
3. **GitHub URL Update**: Updates repository URLs in metadata
4. **Cleanup**: Removes old build artifacts
5. **Testing**: Runs tests if available
6. **Building**: Creates distribution packages
7. **Validation**: Checks package integrity
8. **Publishing**: Uploads to PyPI/TestPyPI
9. **Git Tagging**: Creates version tags

## Troubleshooting

### Common Issues

#### 1. Authentication Errors

```
HTTPError: 403 Client Error: Invalid or non-existent authentication information
```

**Solution**: Check your `.pypirc` file and API tokens

#### 2. Package Already Exists

```
HTTPError: 400 Client Error: File already exists
```

**Solution**: Bump the version number

#### 3. Build Errors

```
ERROR: Failed to build package
```

**Solution**: Check `pyproject.toml` syntax and dependencies

#### 4. Test Failures

```
ERROR: Tests failed
```

**Solution**: Fix failing tests before publishing

### Manual Publishing

If the automated script fails, you can publish manually:

```bash
# Clean and build
rm -rf dist/ build/ *.egg-info/
python -m build

# Check package
twine check dist/*

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Post-Publishing

### 1. Verify Publication

- Check PyPI: https://pypi.org/project/maximum-agents/
- Test installation: `pip install maximum-agents`

### 2. Update Documentation

- Update GitHub README if needed
- Update any external documentation

### 3. Announce Release

- Create GitHub release notes
- Update changelog
- Notify users if applicable

## Security Notes

- Never commit API tokens to version control
- Use environment variables for sensitive data in CI/CD
- Regularly rotate API tokens
- Use TestPyPI for testing before production releases

## CI/CD Integration

For automated publishing, you can integrate this script into your CI/CD pipeline:

```yaml
# GitHub Actions example
name: Publish to PyPI
on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -r publish_requirements.txt
      - run: python publish_to_pypi.py --version patch
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

## Support

If you encounter issues with publishing:

1. Check this guide for common solutions
2. Review the script output for specific error messages
3. Test with TestPyPI first
4. Open an issue on GitHub with error details
