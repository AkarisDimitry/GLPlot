---
name: glplot-performance-profiler
type: agent
description: Profile and optimize GLPlot performance, identify bottlenecks
---

# GLPlot Performance Profiler Agent

## Capabilities

Specializes in profiling, benchmarking, and optimizing GLPlot performance.

## Functions

### Benchmark Suite Execution
- Runs standardized performance benchmarks
- Tests with various data sizes (1k to 1M points)
- Measures rendering time and GPU usage
- Reports throughput and efficiency metrics

### Bottleneck Identification
- Profiles CPU usage during operations
- Identifies slow code paths
- Reports memory allocation patterns
- Suggests optimization opportunities

### Scaling Analysis
- Tests linear vs. superlinear scaling
- Analyzes performance with dataset size
- Reports performance degradation points
- Recommends scaling improvements

### Comparative Benchmarking
- Compares GLPlot performance vs. alternatives
- Benchmarks against Matplotlib, Plotly, VisPy
- Reports relative strengths and weaknesses
- Generates performance comparison reports

## Usage

```bash
/glplot-performance-profiler
```

## Example Tasks

- "Run performance benchmarks on all operations"
- "Find performance bottlenecks in rendering"
- "Analyze scaling behavior with increasing data"
- "Compare performance vs. Matplotlib"
- "Generate performance optimization report"

## Output

Produces performance analysis including:
- Benchmark results with timing
- Throughput metrics (points/second)
- Memory usage patterns
- GPU efficiency metrics
- Specific optimization recommendations

## Metrics Tracked

- Plotting latency
- Rendering FPS
- Memory peak usage
- GPU utilization
- Scaling efficiency
- Dataset throughput

## Integration

Works with:
- Test suite for regression detection
- Documentation for performance claims
- Release notes for performance metrics
