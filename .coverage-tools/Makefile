.PHONY: coverage coverage-html coverage-report test test-cli clean-coverage help

# Default target
help:
	@echo "Available commands:"
	@echo "  coverage       - Run all tests with coverage analysis"
	@echo "  coverage-html  - Generate HTML coverage report"
	@echo "  coverage-report- Show file coverage percentages"
	@echo "  test           - Run all tests"
	@echo "  test-cli       - Run only CLI tests"
	@echo "  clean-coverage - Clean coverage data"

# Run tests with coverage
coverage:
	@cd .. && pytest tests/ \
		--cov=src/b2aiprep \
		--cov-config=.coverage-tools/.coveragerc \
		--cov-report=html:.coverage-tools/htmlcov \
		--cov-report=xml:.coverage-tools/coverage.xml \
		--cov-branch \
		-q > /dev/null 2>&1 || true

# Generate HTML coverage report
coverage-html: coverage
	@echo "HTML coverage report: .coverage-tools/htmlcov/index.html"

# Run file coverage analysis
coverage-report: coverage
	@python coverage_summary.py

# Run all tests without coverage
test:
	@cd .. && pytest tests/ -q

# Run only CLI tests
test-cli:
	@cd .. && pytest tests/test_cli.py -q

# Clean coverage data
clean-coverage:
	@cd .. && coverage erase
	@rm -rf htmlcov/
	@rm -f coverage.xml coverage.json .coverage*
