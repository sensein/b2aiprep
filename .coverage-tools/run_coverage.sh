#!/bin/bash

# Comprehensive Coverage Analysis Script
# Run this from the project root directory

echo "🔍 Running comprehensive code coverage analysis..."

# Activate the conda environment
source activate b2ai 2>/dev/null || conda activate b2ai

# Clean previous coverage data
echo "📝 Cleaning previous coverage data..."
coverage erase

# Run tests with coverage (including subprocess coverage for CLI tests)
echo "🧪 Running tests with coverage tracking..."
pytest tests/ \
    --cov=src/b2aiprep \
    --cov-config=.coverage-tools/.coveragerc \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=xml \
    --cov-branch \
    -v

# Generate additional detailed reports
echo "📊 Generating detailed coverage reports..."

# Terminal summary with missing lines
echo "
═══════════════════════════════════════════════
📈 COVERAGE SUMMARY WITH MISSING LINES
═══════════════════════════════════════════════"
coverage report --show-missing --precision=2

# HTML report
echo "
🌐 HTML coverage report generated at: htmlcov/index.html
   Open in browser: open htmlcov/index.html

📄 XML coverage report generated at: coverage.xml
"

# Coverage by module breakdown
echo "
═══════════════════════════════════════════════
📂 COVERAGE BY MODULE
═══════════════════════════════════════════════"
coverage report --show-missing --precision=2 --format=markdown

echo "✅ Coverage analysis complete!"
echo "📊 View detailed HTML report: open htmlcov/index.html"
