# Screenshots

Drop screenshots into this folder using the exact filenames below — the main
[README](../../README.md) and the [GitHub Pages docs site](../index.html) both reference
these paths directly.

## Required (submission minimum)

| Filename | What to capture |
|---|---|
| `mlflow-experiments.png` | DagsHub **Experiments** tab, all 3 runs (logistic_regression, random_forest, xgboost) visible with their params/metrics columns |
| `model-registry.png` | DagsHub **Models** tab, `loan-default-classifier`, showing the version in **Production** stage |
| `dvc-tracking.png` | Terminal output of `dvc status` (clean) or the DagsHub **Data** tab showing the DVC-tracked dataset/model |
| `gh-actions-build.png` | GitHub **Actions** tab, a green "Build" workflow run |
| `gh-actions-test.png` | GitHub **Actions** tab, a green "Test" workflow run |
| `gh-actions-docker-build.png` | GitHub **Actions** tab, a green "Docker Build" workflow run |
| `fastapi-docs-swagger.png` | `/docs` Swagger UI with a successful `/predict` call expanded (request + response) |

## Recommended (extra polish)

| Filename | What to capture |
|---|---|
| `prediction-ui-empty.png` | The web UI at `/` with the form filled with defaults, before submitting |
| `prediction-ui-result.png` | The web UI after submitting, showing the prediction, probability, and risk pill |
| `docker-running.png` | `docker ps` showing the healthy container, or a `curl` hitting it |
| `dvc-dag.png` | Terminal output of `dvc dag` |
| `huggingface-model.png` | The pushed model's page on huggingface.co (see `HF_REPO_ID` in `.env`) |
| `github-pages-site.png` | The deployed GitHub Pages documentation site |

`reports/shap_summary.png` already exists at the repo root and is reused as-is — no need to
duplicate it here.
