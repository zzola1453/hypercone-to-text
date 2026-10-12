args: {'datasets': ['clinc'], 'ratios': [0.75], 'seeds': [0], 'methods': ['msp', 'energy', 'maha', 'knn'], 'epochs': 30, 'delta': 0.1, 'q': 0.95, 'space': 'feature', 'beta': 1.0, 'gamma': 0.0, 'model': 'sentence-transformers/all-mpnet-base-v2', 'tag': 'smoke2'}

### clinc, known 75%

| method | acc | f1_all | f1_known | f1_unknown | auroc | fpr95 |
|---|---|---|---|---|---|---|
| msp | 77.36 ± 0.00 | 84.36 ± 0.00 | 84.52 ± 0.00 | 65.50 ± 0.00 | 90.69 ± 0.00 | 46.50 ± 0.00 |
| energy | 78.20 ± 0.00 | 84.16 ± 0.00 | 84.30 ± 0.00 | 68.74 ± 0.00 | 89.46 ± 0.00 | 42.99 ± 0.00 |
| maha | 80.96 ± 0.00 | 86.67 ± 0.00 | 86.79 ± 0.00 | 73.44 ± 0.00 | 92.83 ± 0.00 | 37.48 ± 0.00 |
| knn | 77.82 ± 0.00 | 83.91 ± 0.00 | 84.05 ± 0.00 | 67.79 ± 0.00 | 91.97 ± 0.00 | 41.31 ± 0.00 |
