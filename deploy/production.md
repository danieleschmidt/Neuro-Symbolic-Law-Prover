# Production Deployment Guide

## 🚀 Neuro-Symbolic Law Prover - Production Deployment

### Prerequisites

- Docker and Docker Compose
- Kubernetes cluster (for K8s deployment)
- Python 3.9+ (for local development)

### Deployment Options

#### 1. Docker Compose (Recommended for Small-Medium Scale)

```bash
# Basic deployment
docker-compose up -d

# With monitoring
docker-compose --profile monitoring up -d

# Production setup with nginx
docker-compose --profile production up -d

# Development setup
docker-compose --profile dev up -d
```

#### 2. Kubernetes (Recommended for Large Scale)

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/

# Using Helm (recommended)
helm install neuro-law deploy/helm/

# Check deployment status
kubectl get pods -l app=neuro-law-api
kubectl get services
```

#### 3. Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
make test-all

# Start API server
python -m uvicorn api.main:app --reload

# Use CLI
python -m neuro_symbolic_law.cli --help
```

### Configuration

#### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHONPATH` | Python module path | `/app/src` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_WORKERS` | Max parallel workers | `4` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `DATABASE_URL` | Database connection URL | - |

#### API Configuration

The API server supports the following endpoints:

- `POST /analyze` - Start contract analysis
- `GET /analyze/{id}/status` - Get analysis status
- `GET /analyze/{id}/results` - Get analysis results
- `POST /analyze/sync` - Synchronous analysis
- `POST /analyze/upload` - Upload file for analysis
- `GET /regulations` - List available regulations
- `GET /health` - Health check

### Scaling and Performance

#### Horizontal Scaling

```bash
# Scale API pods
kubectl scale deployment neuro-law-api --replicas=5

# Auto-scaling is configured with HPA:
# - Min replicas: 2
# - Max replicas: 10
# - CPU target: 70%
# - Memory target: 80%
```

#### Performance Tuning

1. **Caching**: Redis caching is enabled for compliance results
2. **Parallel Processing**: Configurable worker threads
3. **Resource Limits**: Set appropriate CPU/memory limits
4. **Load Balancing**: Nginx or K8s ingress for load distribution

### Monitoring and Observability

#### Health Checks

- Application health: `GET /health`
- Container health: Built-in Docker healthchecks
- K8s probes: Liveness and readiness probes configured

#### Monitoring Stack

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

#### Logging

- Structured logging with configurable levels
- Container logs available via `docker logs`
- Kubernetes logs via `kubectl logs`

### Security Considerations

#### Authentication & Authorization

```python
# Add authentication middleware to API
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.middleware("http")
async def authenticate(request: Request, call_next):
    # Implement your authentication logic
    pass
```

#### Network Security

- Use HTTPS in production (TLS certificates)
- Configure proper CORS policies
- Network policies in Kubernetes
- Rate limiting (configured in ingress)

#### Data Protection

- Encrypt data at rest and in transit
- Secure Redis with authentication
- Regular security scans with Bandit
- Dependency vulnerability checks with Safety

### Backup and Recovery

#### Data Backup

```bash
# Backup Redis data
docker exec neuro-law_redis_1 redis-cli BGSAVE

# Backup analysis results
kubectl exec -it deployment/neuro-law-api -- python -c "
from api.main import analysis_results
import json
with open('/backup/results.json', 'w') as f:
    json.dump(analysis_results, f)
"
```

#### Disaster Recovery

1. Multi-region deployment
2. Database replication
3. Automated backups
4. Infrastructure as Code

### Troubleshooting

#### Common Issues

1. **Memory Issues**: Increase container memory limits
2. **High CPU**: Scale horizontally or increase CPU limits
3. **Slow Analysis**: Enable parallel processing, check cache
4. **API Errors**: Check logs, verify dependencies

#### Debug Commands

```bash
# Check API health
curl http://localhost:8000/health

# View logs
docker-compose logs neuro-law-api
kubectl logs -f deployment/neuro-law-api

# Test core functionality
python test_minimal.py

# Run with debug logging
LOG_LEVEL=DEBUG docker-compose up
```

### Maintenance

#### Updates

```bash
# Update container image
docker-compose pull
docker-compose up -d

# Kubernetes rolling update
kubectl set image deployment/neuro-law-api api=neuro-law:v0.2.0
```

#### Database Migrations

```bash
# Add migration logic as needed
python manage.py migrate
```

### Performance Benchmarks

Expected performance characteristics:

- **Contract Parsing**: ~100ms for typical contracts
- **GDPR Compliance Check**: ~500ms for full verification
- **API Response Time**: <2s for sync analysis
- **Throughput**: ~50 concurrent analyses with 4 workers
- **Memory Usage**: ~500MB per worker process

### Support

For production support:

1. Check the troubleshooting guide
2. Review application logs
3. Monitor system metrics
4. Contact: daniel@terragonlabs.ai

---

**Note**: This is a production-ready deployment guide. Always test thoroughly in a staging environment before deploying to production.