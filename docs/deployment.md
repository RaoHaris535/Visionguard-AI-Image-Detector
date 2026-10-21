# Cloud API Deployment

The recommended deployment target is Google Cloud Run. The local Flask interface remains outside the cloud and sends images to the deployed inference API, matching Tasks 7 and 8. The API container loads the champion model once during process startup and requires an `X-API-Key` header for `/predict`.

## Required resources

- One Google Cloud project with billing enabled
- Artifact Registry repository named `visionguard`
- Cloud Build and Cloud Run APIs
- A generated inference API key stored as a Cloud Run secret or environment variable

## Deployment outline

1. Authenticate with `gcloud auth login` and select the dedicated project.
2. Enable Cloud Build, Artifact Registry, and Cloud Run.
3. Create a Docker-format Artifact Registry repository in `asia-south1`.
4. Download the MLflow champion artifact to `models/champion.keras` before building.
5. Submit `cloudbuild.yaml` and deploy the resulting image to Cloud Run.
6. Configure `MODEL_PATH`, `MODEL_VERSION`, `DECISION_THRESHOLD`, and `INFERENCE_API_KEY`.
7. Permit public invocation while retaining application-level API-key authentication.
8. Set maximum instances and a billing budget alert.
9. Test `/health` and `/predict` from Postman on a second network.

Exact commands and screenshots are recorded during the deployment session because project IDs, service URLs, current quotas, and billing terms are account-specific.
