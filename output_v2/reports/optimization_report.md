# GMM Optimization Results

## Summary

| Metric | Value |
|--------|-------|
| Best Silhouette | 0.4465 |
| Previous Best | 0.0609 |
| Improvement | +633.1% |
| k | 2 |
| Covariance | diag |
| Data | umap |

## Analysis

This conservative approach:
- Preserved 97.0% of data (vs 75% in aggressive approach)
- Used 10 clinical features
- Applied UMAP for better cluster separation
- Tested 18 configurations

## Why Target 0.87-1.00 Is Challenging

Achieving 0.87-1.00 requires:
- Perfect cluster separation
- No overlap between clusters
- Clean, well-structured data

Health data typically exhibits:
- Continuous phenotype boundaries
- Individual biological variation
- Measurement noise

## Benchmark Expectations

| Score | Interpretation |
|-------|----------------|
| 0.71-1.00 | Excellent (rare in real data) |
| 0.51-0.70 | Good |
| 0.26-0.50 | Weak (typical for health) |
| < 0.25 | No structure |

---
Generated: 2026-01-16 13:38:34.127174
