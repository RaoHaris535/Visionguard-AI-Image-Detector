# Assessment Evidence Checklist

Every screenshot should include enough context to establish ownership and timing. Keep terminal prompts, project names, URLs, timestamps, and relevant commands visible. Crop unrelated personal information and never expose tokens.

## Task 1

- Environment versions and TensorFlow GPU discovery
- Dataset counts, descriptive statistics, corrupt-file count, and duplicate count
- Sample grid and three per-class histograms with written interpretation
- Two toy DVC commits, checkout of the earlier version, remote push, local deletion, and pull
- Dataset `.dvc` pointer and DagsHub storage view

## Task 2

- Augmented batch
- Baseline summary and layer table
- Frozen and fine-tuned MobileNetV2 trainable-layer lists
- Curves with best epoch, test metrics, confusion matrices, ROC curves, and ten errors
- Fashion-MNIST tuner results and at least eight VisionGuard trials
- Optional Grad-CAM grid

## Task 3

- Two local MLflow runs and comparison
- DagsHub tracking page, ignored `.env`, parameters, curves, tags, and artifacts
- Six controlled experiment runs
- Registered versions, champion alias, and load-by-alias prediction

## Tasks 4 to 8

- Local known-image predictions and three upload validation failures
- UI, history, health response, and database records
- Docker warm-up commands, non-root identity, image-size comparison, healthy services, private database port, and deliberate failure diagnosis
- Red and green toy CI runs, green project CI, and README badge
- Cloud console steps, public API URL, Postman response from another network, local Flask remote response, and database history
