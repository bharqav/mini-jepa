# Deployment

Two supported shapes:

1. **Compose (recommended for a single host):** API container plus nginx TLS
   termination, edge rate limiting and internal-only metrics.
2. **systemd (bare metal / GPU host):** the API on loopback behind an existing
   nginx or load balancer.

Neither shape makes this a multi-tenant service. One key grants inference; there
is no user database, per-tenant isolation, or cluster-wide quota. See
[`../docs/production.md`](../docs/production.md) for the honest gap list.

## Compose

```bash
cp deploy/.env.example deploy/.env
mkdir -p deploy/secrets deploy/certs
python scripts/generate_api_keys.py > deploy/secrets/api_keys.json
chmod 600 deploy/secrets/api_keys.json

# place fullchain.pem and privkey.pem in deploy/certs/
docker compose -f deploy/docker-compose.yml up -d --build
curl -k https://localhost:8443/health
```

The API is not published directly; only nginx binds host ports `8080` (HTTP
redirect) and `8443` (TLS). Put a checkpoint under `checkpoints/checkpoint` (or
set `JEV_MODEL_DIR`) and keep it read-only.

### GPU

The default image is CPU-only. Build a CUDA 12.8 image with:

```bash
docker build -f deploy/Dockerfile \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  --build-arg TORCH_VERSION=2.11.0 \
  -t jev-qwen:cuda .
```

Then change `image:` in the compose file and give the container access to the
GPU (`gpus: all` with the NVIDIA container toolkit). The model must fit the
card; `JEV_MAX_CONCURRENCY=1` is deliberate for a single 8 GB GPU.

## systemd

```bash
sudo useradd --system --home /opt/mini-jepa --shell /usr/sbin/nologin jev
sudo mkdir -p /etc/jev-qwen /var/lib/jev-qwen /models
sudo cp deploy/jev-qwen.service /etc/systemd/system/
sudo cp deploy/jev-qwen.env.example /etc/jev-qwen/jev-qwen.env
sudo chmod 600 /etc/jev-qwen/jev-qwen.env
sudo python scripts/generate_api_keys.py --file /etc/jev-qwen/keys.json --key-id prod
sudo chown -R jev:jev /var/lib/jev-qwen
sudo systemctl daemon-reload
sudo systemctl enable --now jev-qwen
```

The unit binds `127.0.0.1:8000`; terminate TLS in front of it. If nginx runs on
the same host, set `JEV_TRUSTED_PROXIES=127.0.0.1/32` and keep the default nginx
example's upstream pointed at `127.0.0.1:8000`.

## Key rotation

Add a second key, deploy it, move clients over, then remove the old entry and
restart (or reload the process). Both keys are accepted while present:

```bash
python scripts/generate_api_keys.py --file deploy/secrets/api_keys.json --key-id new
# update the JSON, remove the old key, restart the api service
```

`JEV_API_KEYS_FILE` is read at startup. A restart is required to pick up
changes; there is no hot reload.

## Metrics

`/metrics` is unauthenticated and intended for a private scrape network. The
nginx example allows only loopback and the compose subnet. Scrape
`http://<api>:8000/metrics` from Prometheus, or route through nginx from an
allowed address. Useful series:

- `jev_requests_total{method,path,status}`
- `jev_request_duration_seconds{method,path}`
- `jev_in_flight_requests`
- `jev_rate_limited_total`
- `jev_auth_failures_total{reason}`
- `jev_inference_errors_total{error_type}`
- `jev_model_ready`

## Load/soak test

Run against staging, not production:

```bash
JEV_API_TOKEN=... python scripts/load_test.py \
  --url https://staging.example/v1/decide \
  --concurrency 8 --duration 60 --rps 2
```

The script reports p50/p95/p99, throughput and status counts; it exits non-zero
on connection errors or 5xx. 429s are reported as expected backpressure.
