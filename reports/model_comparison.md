# Model Evaluation Results

K = 10; relevant = test rating >= 4.0; evaluated users = 796; test ratings = 11,223; avg relevant items per user = 4.81.

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Coverage | Diversity | Novelty | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `popularity` | 0.0730 | 0.1635 | 0.1403 | 0.4837 | 0.0867 | 0.8986 | 7.1708 | 0.8021 | 0.6370 |
| `item_knn_cf` | 0.0732 | 0.1608 | 0.1401 | 0.4799 | 0.1133 | 0.8965 | 7.2098 | 0.8285 | 0.6568 |
| `matrix_factorization` | 0.0237 | 0.0497 | 0.0366 | 0.1935 | 0.1217 | 0.8634 | 9.6880 | 0.6927 | 0.5525 |
| `implicit_als` | 0.0995 | 0.2287 | 0.1955 | 0.5678 | 0.5050 | 0.8451 | 7.6101 | 0.8962 | 0.6973 |
| `hybrid_als_content` | 0.1222 | 0.2778 | 0.2390 | 0.6281 | 0.5967 | 0.6486 | 8.0891 | 0.6927 | 0.5525 |

Best model by NDCG@10: **hybrid_als_content**.

Higher is better for all metrics except RMSE and MAE. RMSE/MAE for the popularity baseline use the item's Bayesian-average rating as the prediction.
