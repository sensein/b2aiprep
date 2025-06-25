# Coverage Analysis Tools

This directory contains all the code coverage analysis tools and configuration for the B2AI-Prep project.

## Files

- **`.coveragerc`** - Coverage configuration file
- **`run_coverage.sh`** - Comprehensive coverage analysis script
- **`coverage_summary.py`** - Detailed coverage analysis tool
- **`COVERAGE.md`** - Complete documentation

## Quick Usage

From the project root directory:

```bash
# Using Make (recommended)
make coverage-report

# Using the shell script directly
./.coverage-tools/run_coverage.sh

# Using Python script for analysis
python .coverage-tools/coverage_summary.py
```

## Configuration

The `.coveragerc` file configures:
- Source code tracking (`src/b2aiprep`)
- Subprocess coverage for CLI testing
- Branch coverage analysis
- HTML and XML report generation

## Reports Generated

- **HTML Report**: `htmlcov/index.html` (interactive)
- **XML Report**: `coverage.xml` (CI/CD integration)
- **Terminal Report**: Immediate feedback

See `COVERAGE.md` for complete documentation.
