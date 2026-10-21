# Contributing

Create a focused branch, keep credentials and generated data out of Git, and add or update tests for behavior changes. Before opening a pull request, run:

```bash
uv run ruff check .
uv run pytest
```

Model changes must record the Git commit, DVC data version, seed, configuration, and evaluation artifacts in MLflow. Do not compare models on the test split while tuning decisions are still being made.
