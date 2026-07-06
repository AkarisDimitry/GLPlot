---
name: glplot-api-analyzer
type: agent
description: Analyze and verify GLPlot API for consistency and completeness
---

# GLPlot API Analyzer Agent

## Capabilities

Specializes in analyzing GLPlot's public API for consistency, completeness, and Matplotlib compatibility.

## Functions

### API Completeness Check
- Identifies missing public functions compared to Matplotlib
- Lists implemented vs. planned features
- Reports API gaps and coverage
- Suggests priority improvements

### Consistency Analysis
- Verifies parameter naming conventions
- Checks function signatures match Matplotlib
- Identifies inconsistent behavior
- Reports deviation from conventions

### Documentation Review
- Audits docstring coverage
- Validates parameter documentation
- Checks return value documentation
- Reports missing examples

### Matplotlib Compatibility Report
- Lists implemented Matplotlib functions
- Identifies unsupported features
- Reports workarounds and alternatives
- Suggests compatibility improvements

## Usage

```bash
/glplot-api-analyzer
```

## Example Tasks

- "Audit API for Matplotlib compatibility"
- "Check which pyplot functions are missing"
- "Verify all public functions have docstrings"
- "Analyze parameter consistency"
- "Generate API completeness report"

## Output

Produces detailed API analysis including:
- Completeness percentage
- Compatibility assessment
- Documentation coverage
- Specific gaps and recommendations
- Priority action items

## Integration

Works with:
- Test suite for behavioral verification
- Documentation generator
- Release preparation checklist
