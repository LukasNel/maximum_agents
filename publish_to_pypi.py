#!/usr/bin/env python3
"""
Maximum Agents PyPI Publishing Script

This script automates the process of publishing the maximum-agents package to PyPI.
It handles version bumping, building, testing, and publishing with proper GitHub integration.

Usage:
    python publish_to_pypi.py [--version TYPE] [--test] [--dry-run] [--github-url URL]

Examples:
    python publish_to_pypi.py --version patch --test
    python publish_to_pypi.py --version minor --github-url https://github.com/your-org/maximum-agents
    python publish_to_pypi.py --dry-run
"""

import os
import sys
import subprocess
import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import shutil

class PyPIPublisher:
    def __init__(self, project_root: Path, github_url: Optional[str] = None):
        self.project_root = project_root
        self.github_url = github_url or "https://github.com/LukasNel/maximum_agents"
        self.pyproject_path = project_root / "pyproject.toml"
        self.readme_path = project_root / "README.md"
        
    def check_prerequisites(self) -> bool:
        """Check if all required tools are available."""
        print("🔍 Checking prerequisites...")
        
        required_tools = ["python", "pip", "twine"]
        missing_tools = []
        subprocess.run(["pip3", "install", "twine"], capture_output=True, check=True)
        
        for tool in required_tools:
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
                print(f"✅ {tool} is available")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"❌ {tool} is missing: {e}")
                missing_tools.append(tool)
                print(f"❌ {tool} is missing")
        
        if missing_tools:
            print(f"\n❌ Missing required tools: {', '.join(missing_tools)}")
            print("Install them with: pip install twine build")
            return False
        
        return True
    
    def get_current_version(self) -> str:
        """Get current version from pyproject.toml."""
        with open(self.pyproject_path, 'r') as f:
            content = f.read()
            match = re.search(r'version = "([^"]+)"', content)
            if match:
                return match.group(1)
            raise ValueError("Could not find version in pyproject.toml")
    
    def bump_version(self, version_type: str) -> str:
        """Bump version number."""
        current_version = self.get_current_version()
        print(f"📦 Current version: {current_version}")
        
        # Parse version
        parts = current_version.split('.')
        major, minor, patch = map(int, parts)
        
        if version_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        elif version_type == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid version type: {version_type}")
        
        new_version = f"{major}.{minor}.{patch}"
        print(f"🚀 New version: {new_version}")
        
        # Update pyproject.toml
        with open(self.pyproject_path, 'r') as f:
            content = f.read()
        
        content = re.sub(
            r'version = "[^"]+"',
            f'version = "{new_version}"',
            content
        )
        
        with open(self.pyproject_path, 'w') as f:
            f.write(content)
        
        return new_version
    
    def update_github_urls(self, github_url: str):
        """Update GitHub URLs in pyproject.toml and README.md."""
        print(f"🔗 Updating GitHub URLs to: {github_url}")
        
        # Update pyproject.toml
        with open(self.pyproject_path, 'r') as f:
            content = f.read()
        
        # Replace GitHub URLs
        content = re.sub(
            r'https://github\.com/[^"]+',
            github_url,
            content
        )
        
        with open(self.pyproject_path, 'w') as f:
            f.write(content)
        
        # Update README.md
        with open(self.readme_path, 'r') as f:
            content = f.read()
        
        # Replace GitHub URLs
        content = re.sub(
            r'https://github\.com/[^"]+',
            github_url,
            content
        )
        
        with open(self.readme_path, 'w') as f:
            f.write(content)
    
    def clean_build_directories(self):
        """Clean build directories."""
        print("🧹 Cleaning build directories...")
        
        dirs_to_clean = ["dist", "build", "*.egg-info"]
        
        for pattern in dirs_to_clean:
            for path in self.project_root.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"   Removed {path}")
    
    def run_tests(self) -> bool:
        """Run tests if they exist."""
        print("🧪 Running tests...")
        
        test_dirs = ["tests", "test"]
        test_files = ["test_*.py", "*_test.py"]
        
        has_tests = False
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                has_tests = True
                break
        
        if not has_tests:
            print("ℹ️  No tests found, skipping test execution")
            return True
        
        try:
            # Try to run tests with pytest
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-v"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Tests passed")
                return True
            else:
                print("❌ Tests failed")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except FileNotFoundError:
            print("ℹ️  pytest not available, trying unittest")
            try:
                result = subprocess.run(
                    ["python", "-m", "unittest", "discover", "tests"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print("✅ Tests passed")
                    return True
                else:
                    print("❌ Tests failed")
                    print(result.stdout)
                    print(result.stderr)
                    return False
                    
            except Exception as e:
                print(f"❌ Error running tests: {e}")
                return False
    
    def build_package(self) -> bool:
        """Build the package."""
        print("🔨 Building package...")
        
        try:
            # Build the package
            result = subprocess.run(
                ["python", "-m", "build"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Package built successfully")
                
                # List built files
                dist_dir = self.project_root / "dist"
                if dist_dir.exists():
                    files = list(dist_dir.glob("*"))
                    print(f"📦 Built files: {[f.name for f in files]}")
                
                return True
            else:
                print("❌ Build failed")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error building package: {e}")
            return False
    
    def check_package(self) -> bool:
        """Check the built package."""
        print("🔍 Checking package...")
        
        try:
            dist_dir = self.project_root / "dist"
            if not dist_dir.exists():
                print("❌ No dist directory found")
                return False
            
            # Check with twine
            result = subprocess.run(
                ["twine", "check", "dist/*"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Package check passed")
                return True
            else:
                print("❌ Package check failed")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error checking package: {e}")
            return False
    
    def publish_to_testpypi(self) -> bool:
        """Publish to TestPyPI."""
        print("🚀 Publishing to TestPyPI...")
        
        try:
            result = subprocess.run(
                ["twine", "upload", "--repository", "testpypi", "dist/*"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Successfully published to TestPyPI")
                print("🔗 You can test the package with:")
                print("   pip install --index-url https://test.pypi.org/simple/ maximum-agents")
                return True
            else:
                print("❌ Failed to publish to TestPyPI")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error publishing to TestPyPI: {e}")
            return False
    
    def publish_to_pypi(self) -> bool:
        """Publish to PyPI."""
        print("🚀 Publishing to PyPI...")
        
        try:
            result = subprocess.run(
                ["twine", "upload", "dist/*"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Successfully published to PyPI!")
                print("🔗 Package available at: https://pypi.org/project/maximum-agents/")
                return True
            else:
                print("❌ Failed to publish to PyPI")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error publishing to PyPI: {e}")
            return False
    
    def create_git_tag(self, version: str) -> bool:
        """Create a git tag for the version."""
        print(f"🏷️  Creating git tag: v{version}")
        
        try:
            # Add and commit changes
            subprocess.run(["git", "add", "pyproject.toml", "README.md"], cwd=self.project_root, check=True)
            subprocess.run(["git", "commit", "-m", f"Release v{version}"], cwd=self.project_root, check=True)
            
            # Create tag
            subprocess.run(["git", "tag", f"v{version}"], cwd=self.project_root, check=True)
            
            print(f"✅ Git tag v{version} created")
            print("💡 Don't forget to push: git push origin main --tags")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating git tag: {e}")
            return False
    
    def publish(self, version_type: str, test_only: bool = False, dry_run: bool = False, github_url: Optional[str] = None):
        """Main publishing workflow."""
        print("🚀 Maximum Agents PyPI Publisher")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_prerequisites():
            sys.exit(1)
        
        # Update GitHub URL if provided
        if github_url:
            self.update_github_urls(github_url)
        
        # Bump version
        try:
            new_version = self.bump_version(version_type)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        
        if dry_run:
            print("🔍 Dry run mode - no actual publishing will occur")
            print(f"📦 Would publish version: {new_version}")
            return
        
        # Clean build directories
        self.clean_build_directories()
       
        
        # Build package
        if not self.build_package():
            print("❌ Build failed, aborting publish")
            sys.exit(1)
        
        # Check package
        if not self.check_package():
            print("❌ Package check failed, aborting publish")
            sys.exit(1)
        
        # Publish to TestPyPI or PyPI
        if test_only:
            if not self.publish_to_testpypi():
                print("❌ TestPyPI publish failed")
                sys.exit(1)
        else:
            if not self.publish_to_pypi():
                print("❌ PyPI publish failed")
                sys.exit(1)
        
        # Create git tag
        self.create_git_tag(new_version)
        
        print("\n🎉 Publishing completed successfully!")
        print(f"📦 Version: {new_version}")
        if test_only:
            print("🔗 TestPyPI: https://test.pypi.org/project/maximum-agents/")
        else:
            print("🔗 PyPI: https://pypi.org/project/maximum-agents/")
        print(f"🔗 GitHub: {self.github_url}")

def main():
    parser = argparse.ArgumentParser(description="Publish maximum-agents to PyPI")
    parser.add_argument(
        "--version",
        choices=["major", "minor", "patch"],
        default="patch",
        help="Version bump type (default: patch)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Publish to TestPyPI instead of PyPI"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually publishing"
    )
    parser.add_argument(
        "--github-url",
        help="GitHub repository URL (e.g., https://github.com/LukasNel/maximum_agents)"
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Create publisher
    publisher = PyPIPublisher(project_root, args.github_url)
    
    # Publish
    publisher.publish(
        version_type=args.version,
        test_only=args.test,
        dry_run=args.dry_run,
        github_url=args.github_url
    )

if __name__ == "__main__":
    main()
