#!/usr/bin/env python3
"""
Coverage Analysis Summary Script
Shows percent coverage for every file.
"""

import json
import os
import subprocess
from pathlib import Path


def get_coverage_data():
    """Get coverage data from coverage.py"""
    try:
        # Use the coverage file in the .coverage-tools directory
        coverage_file = Path("coverage.json")
        if coverage_file.exists():
            with open(coverage_file) as f:
                return json.load(f)
        else:
            # If file doesn't exist, try to generate it from the project root
            os.chdir("..")
            subprocess.run(["coverage", "json", "-o", ".coverage-tools/coverage.json"], check=True)
            os.chdir(".coverage-tools")
            if coverage_file.exists():
                with open(coverage_file) as f:
                    return json.load(f)
        return None
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return None


def main():
    """Show coverage percentage for every file"""
    coverage_data = get_coverage_data()
    if not coverage_data:
        print("❌ Could not get coverage data")
        return

    # Get all files and their coverage percentages
    files_coverage = []
    for filepath, data in coverage_data["files"].items():
        coverage_pct = data["summary"]["percent_covered"]
        filename = Path(filepath).name
        files_coverage.append((filename, coverage_pct))

    # Sort by filename
    files_coverage.sort(key=lambda x: x[0])

    # Print results
    for filename, coverage_pct in files_coverage:
        print(f"{filename}: {coverage_pct:.1f}%")


if __name__ == "__main__":
    main()
