# GLPlot Claude Skills and Agents Manifest

This directory contains Claude AI skills and custom agents for GLPlot development, testing, and documentation.

## Available Agents

### Test and Quality Assurance

- **glplot-test-runner** - Run and analyze the comprehensive test suite
  - Execute all 65+ tests with coverage reporting
  - Run specific test categories (unit, API, edge cases, performance, regression)
  - Generate performance benchmarks and test analysis
  - Identify flaky or slow tests

- **glplot-api-analyzer** - Verify API completeness and consistency
  - Audit Matplotlib API compatibility
  - Check function signature consistency
  - Verify docstring completeness
  - Identify missing features and gaps

### Performance and Optimization

- **glplot-performance-profiler** - Profile and optimize performance
  - Run comprehensive benchmarks (1k to 1M data points)
  - Identify performance bottlenecks
  - Analyze scaling behavior
  - Compare performance vs. Matplotlib, Plotly, VisPy

### Documentation and Publishing

- **glplot-documentation-auditor** - Audit and improve documentation
  - Check documentation completeness
  - Verify docstring coverage
  - Analyze example distribution
  - Validate scientific accuracy
  - Prepare publication materials

- **glplot-release-manager** - Manage releases and versioning
  - Pre-release checklist automation
  - Version management and semver
  - Changelog and release notes generation
  - Publication workflow (Zenodo, PyPI, journals)

### Support and Bug Tracking

- **glplot-bug-reporter** - Help report and track issues
  - Analyze error messages
  - Generate minimal reproducible examples
  - Classify and prioritize issues
  - Detect regressions
  - Suggest workarounds

## Usage

### Invoke a skill from Claude Code:

```bash
/glplot-test-runner
```

### Example tasks:

- "Run all tests and generate coverage report" → `/glplot-test-runner`
- "Audit API for Matplotlib compatibility" → `/glplot-api-analyzer`
- "Profile performance bottlenecks" → `/glplot-performance-profiler`
- "Prepare version 0.2.0 release" → `/glplot-release-manager`
- "Help me report this bug" → `/glplot-bug-reporter`

## Integration with Testing

The test suite includes:
- **test_api_consistency.py** - 40+ tests for API compliance
- **test_edge_cases.py** - 30+ tests for boundary conditions
- **test_performance_benchmarks.py** - 20+ performance tests
- **test_regression.py** - 50+ regression tests

Total: 130+ tests covering unit, integration, performance, and regression scenarios.

## File Structure

```
.claude/
├── MANIFEST.md (this file)
└── agents/
    ├── glplot-test-runner.md
    ├── glplot-api-analyzer.md
    ├── glplot-performance-profiler.md
    ├── glplot-documentation-auditor.md
    ├── glplot-release-manager.md
    └── glplot-bug-reporter.md
```

## Contributing

When adding new skills:
1. Create a new `.md` file in `.claude/agents/`
2. Follow the skill template format
3. Document capabilities and functions
4. Include usage examples
5. List integration points
6. Update this MANIFEST.md

## See Also

- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Contribution guidelines
- [PUBLICATION_CHECKLIST.md](../../PUBLICATION_CHECKLIST.md) - Publication requirements
- [tests/](../../tests/) - Complete test suite documentation
