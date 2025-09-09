#!/bin/bash

# Maximum Agents PyPI Publishing Script
# Simple wrapper for the Python publishing script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Maximum Agents PyPI Publisher"
    echo "============================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --version TYPE     Version bump type: major, minor, patch (default: patch)"
    echo "  --test            Publish to TestPyPI instead of PyPI"
    echo "  --dry-run         Show what would be done without actually publishing"
    echo "  --github-url URL  GitHub repository URL"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --version patch --test"
    echo "  $0 --version minor --github-url https://github.com/your-org/maximum-agents"
    echo "  $0 --dry-run"
    echo ""
}

# Check if Python script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/publish_to_pypi.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python publishing script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

# Parse arguments
VERSION_TYPE="patch"
TEST_ONLY=false
DRY_RUN=false
GITHUB_URL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION_TYPE="$2"
            shift 2
            ;;
        --test)
            TEST_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --github-url)
            GITHUB_URL="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate version type
if [[ ! "$VERSION_TYPE" =~ ^(major|minor|patch)$ ]]; then
    print_error "Invalid version type: $VERSION_TYPE. Must be major, minor, or patch"
    exit 1
fi

# Build command
CMD="python3 $PYTHON_SCRIPT --version $VERSION_TYPE"

if [ "$TEST_ONLY" = true ]; then
    CMD="$CMD --test"
fi

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
fi

if [ -n "$GITHUB_URL" ]; then
    CMD="$CMD --github-url \"$GITHUB_URL\""
fi

# Show what we're about to do
print_status "Starting Maximum Agents PyPI Publisher"
print_status "Version bump: $VERSION_TYPE"
if [ "$TEST_ONLY" = true ]; then
    print_status "Target: TestPyPI"
else
    print_status "Target: PyPI"
fi
if [ "$DRY_RUN" = true ]; then
    print_status "Mode: Dry run"
fi
if [ -n "$GITHUB_URL" ]; then
    print_status "GitHub URL: $GITHUB_URL"
fi
echo ""

# Execute the Python script
print_status "Executing: $CMD"
echo ""

if eval $CMD; then
    print_success "Publishing completed successfully!"
else
    print_error "Publishing failed!"
    exit 1
fi
