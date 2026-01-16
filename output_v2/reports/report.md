# GMM Optimization Results

## Summary

| Metric | Value |
|--------|-------|
| Best Silhouette | 0.0391 |
| Target | 0.87 - 1.00 |
| Status | ✗ BELOW TARGET |
| k | 3 |
| Covariance | diag |

## Analysis

Achieving 0.87-1.00 requires:
- Perfect cluster separation
- No overlap between clusters
- Clean, well-structured data

Health data typically exhibits:
- Continuous phenotype boundaries
- Individual variation
- Measurement noise
- Overlapping characteristics

## Realistic Expectations

| Score | Interpretation |
|-------|----------------|
| 0.71-1.00 | Excellent (rare in real data) |
| 0.51-0.70 | Good |
| 0.26-0.50 | Weak (typical for health) |
| < 0.25 | No structure |

---
Generated: 2026-01-16 13:23:13.304411
