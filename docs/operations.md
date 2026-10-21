# Operations and Troubleshooting

## Local stack

Copy `.env.example` to `.env`, replace every placeholder, ensure `models/champion.keras` exists, then start the services:

```bash
docker compose up --build -d
docker compose ps
curl --fail http://localhost:8000/health
```

The database has no host port. The web service reaches PostgreSQL through the Compose service name `db` on the private `backend` network.

## Routine commands

```bash
docker compose logs --follow web
docker compose logs --follow db
docker compose restart web
docker compose exec web id
docker compose exec db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
docker compose down
```

`docker compose down` preserves the named volume. Removing the volume is destructive and is not part of routine shutdown.

## Deliberate failure exercise

Temporarily point `MODEL_PATH` to a nonexistent file, start the web service, and capture the startup error from `docker compose logs web`. Restore the correct path and recreate the service. Never use a real credential as the intentionally broken value.
