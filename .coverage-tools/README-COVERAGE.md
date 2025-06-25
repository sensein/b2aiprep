# Code Coverage Analysis

Code coverage analysis tools are available in the `.coverage-tools/` directory.

## Quick Start

```bash
# Activate environment and run coverage analysis
conda activate b2ai
make coverage-report
```

## Available Commands
$$
- `make coverage` - Run tests with coverage analysis
- `make coverage-html` - Generate HTML coverage report
- `make coverage-report` - Show detailed coverage analysis
- `make test` - Run all tests without coverage
- `make test-cli` - Run only CLI tests
- `make clean-coverage` - Clean coverage data

## Documentation

- **`.coverage-tools/README.md`** - Quick reference
- **`.coverage-tools/COVERAGE.md`** - Complete documentation

## Current Status

- **Overall Coverage**: ~48%
- **CLI Interface**: 100% ✅
- **Core Commands**: 77% ✅
- **BIDS Processing**: 92% ✅

The CLI tests provide excellent integration coverage across both `cli.py` and `commands.py` modules.
