---
name: glplot-test-runner
type: agent
description: Run and analyze GLPlot test suite with comprehensive reporting
---

# GLPlot Test Runner Agent

## Capabilities

This agent specializes in running, analyzing, and reporting on the GLPlot test suite.

## Functions

### Run All Tests
Executes the complete test suite with coverage reporting:
- Runs all 65+ tests across all modules
- Generates HTML coverage reports
- Identifies flaky or slow tests
- Reports test performance metrics

### Run Specific Test Categories
- **Unit Tests**: Core functionality tests
- **API Tests**: Matplotlib API compatibility
- **Edge Case Tests**: Boundary condition handling
- **Performance Tests**: Scaling and benchmarks
- **Regression Tests**: Previously fixed bug verification

### Generate Test Reports
Creates detailed test analysis including:
- Test coverage by module
- Performance benchmarks
- Failed test diagnosis
- Coverage gaps and recommendations

## Usage

```bash
/glplot-test-runner
```

## Example Tasks

- "Run all tests and generate coverage report"
- "Find slow tests in the test suite"
- "Check test coverage for pyplot module"
- "Run regression tests to verify bug fixes"
- "Benchmark performance on large datasets"

## Output

Produces structured test reports with:
- Test pass/fail status
- Execution times
- Coverage percentages
- Performance metrics
- Recommended improvements
