# Deployment Guide

## Neuro-Symbolic Law Prover - Production Deployment

This guide covers deployment of the Neuro-Symbolic Law Prover system with all three generations of enhancements.

### 🏗️ Architecture Overview

The system implements a three-generation architecture:

- **Generation 1**: Basic functionality with core compliance verification
- **Generation 2**: Enhanced robustness with error handling and monitoring  
- **Generation 3**: Scalable architecture with adaptive caching and auto-scaling

### 📋 System Requirements

#### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2 CPU cores
- 10GB disk space

#### Recommended Requirements (Generation 3)
- Python 3.10+
- 16GB RAM
- 8 CPU cores
- 50GB disk space
- Redis (for distributed caching)

#### Optional Dependencies
```bash
# Core functionality
pip install transformers torch sentence-transformers

# SMT solving
pip install z3-solver

# Performance monitoring
pip install psutil

# Distributed caching
pip install redis

# Advanced caching
pip install cryptography
```

### 🚀 Deployment Options

#### Option 1: Basic Deployment (Generation 1)
```python
from neuro_symbolic_law.core.legal_prover import LegalProver
from neuro_symbolic_law.parsing.contract_parser import ContractParser
from neuro_symbolic_law.regulations import GDPR

# Initialize basic components
parser = ContractParser(model='basic')
prover = LegalProver(debug=False)
gdpr = GDPR()

# Basic usage
contract = parser.parse(contract_text)
results = prover.verify_compliance(contract, gdpr)
```

#### Option 2: Enhanced Deployment (Generation 2)
```python
from neuro_symbolic_law.core.enhanced_prover import EnhancedLegalProver
from neuro_symbolic_law.core.monitoring import setup_monitoring

# Setup monitoring
setup_monitoring(retention_hours=24)

# Initialize enhanced prover with robustness features
prover = EnhancedLegalProver(
    cache_enabled=True,
    max_cache_size=10000,
    enable_security_logging=True,
    verification_timeout=300
)

# Enhanced usage with error handling
try:
    results = prover.verify_compliance(contract, regulation, 
                                     focus_areas=['security', 'data_subject_rights'])
    report = prover.generate_compliance_report(results, contract)
except Exception as e:
    # Comprehensive error handling built-in
    logger.error(f"Verification failed: {e}")
```

#### Option 3: Scalable Deployment (Generation 3)
```python
from neuro_symbolic_law.core.scalable_prover import ScalableLegalProver
import asyncio

# Initialize scalable prover with auto-scaling
prover = ScalableLegalProver(
    initial_cache_size=1000,
    max_cache_size=50000,
    max_workers=8,
    enable_adaptive_caching=True,
    enable_concurrent_processing=True,
    memory_threshold=0.8
)

# Concurrent processing of multiple contracts
async def verify_multiple_contracts(contracts, regulation):
    results = await prover.verify_compliance_concurrent(
        contracts=contracts,
        regulation=regulation,
        max_concurrent=4
    )
    return results

# Batch processing
async def batch_verification(requests):
    return await prover.batch_verify_compliance(
        batch_requests=requests,
        batch_size=10
    )
```

### 🐳 Docker Deployment

#### Basic Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY examples/ ./examples/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port for web interface (if applicable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from neuro_symbolic_law.core.legal_prover import LegalProver; LegalProver()"

CMD ["python", "examples/basic_usage.py"]
```

#### Docker Compose (Generation 3 with Redis)
```yaml
version: '3.8'

services:
  neuro-symbolic-law:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
      - CACHE_SIZE=10000
    depends_on:
      - redis
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:
```

### ☸️ Kubernetes Deployment

#### Basic Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuro-symbolic-law
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neuro-symbolic-law
  template:
    metadata:
      labels:
        app: neuro-symbolic-law
    spec:
      containers:
      - name: app
        image: neuro-symbolic-law:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "8Gi" 
            cpu: "2"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30

---
apiVersion: v1
kind: Service
metadata:
  name: neuro-symbolic-law-service
spec:
  selector:
    app: neuro-symbolic-law
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 🔧 Configuration

#### Environment Variables
```bash
# Core settings
PYTHONPATH=/app/src
LOG_LEVEL=INFO

# Generation 2 settings
CACHE_ENABLED=true
MAX_CACHE_SIZE=10000
VERIFICATION_TIMEOUT=300
SECURITY_LOGGING=true

# Generation 3 settings
ADAPTIVE_CACHE=true
CONCURRENT_PROCESSING=true
MAX_WORKERS=8
REDIS_URL=redis://localhost:6379
MEMORY_THRESHOLD=0.8

# Optional ML models
TRANSFORMERS_CACHE=/app/cache/transformers
SENTENCE_TRANSFORMERS_HOME=/app/cache/sentence-transformers
```

#### Configuration File (config.yaml)
```yaml
app:
  name: "Neuro-Symbolic Law Prover"
  version: "1.0.0"
  environment: "production"

parser:
  model: "enhanced"
  max_clause_length: 5000
  enable_neural_parsing: true

prover:
  generation: 3
  cache:
    enabled: true
    initial_size: 1000
    max_size: 50000
    adaptive: true
  
  concurrency:
    enabled: true
    max_workers: 8
    memory_threshold: 0.8
  
  monitoring:
    enabled: true
    retention_hours: 24
    health_checks: true

security:
  logging: true
  input_validation: true
  rate_limiting: true

redis:
  url: "redis://localhost:6379"
  timeout: 5
  max_connections: 20
```

### 📊 Monitoring and Observability

#### Health Checks
The system provides comprehensive health checks:

```python
from neuro_symbolic_law.core.monitoring import get_health_checker

# Check system health
health_checker = get_health_checker()
health_report = health_checker.get_health_report()

print(f"Overall status: {health_report['overall_status']}")
for check_name, check_result in health_report['checks'].items():
    print(f"{check_name}: {check_result['status']} - {check_result['message']}")
```

#### Metrics Collection
```python
from neuro_symbolic_law.core.monitoring import get_metrics_collector

# Get performance metrics
metrics = get_metrics_collector()
summary = metrics.get_all_metrics_summary()

print(f"Uptime: {summary['uptime_seconds']}s")
print(f"Total verifications: {summary['counters'].get('compliance_verification.completed', 0)}")
```

#### Generation 3 Performance Metrics
```python
from neuro_symbolic_law.core.scalable_prover import ScalableLegalProver

prover = ScalableLegalProver()
performance_metrics = prover.get_performance_metrics()

print(f"Cache hit rate: {performance_metrics['adaptive_cache']['hit_rate']:.2%}")
print(f"Current workers: {performance_metrics['resource_manager']['current_workers']}")
print(f"Avg verification time: {performance_metrics['performance_stats']['avg_verification_time']:.3f}s")
```

### 🔒 Security Considerations

#### Production Security Checklist
- [ ] Enable security logging
- [ ] Implement input validation and sanitization
- [ ] Set up rate limiting
- [ ] Use HTTPS for all communications
- [ ] Secure Redis with authentication
- [ ] Implement proper error handling without information leakage
- [ ] Regular security audits and dependency updates
- [ ] Monitor for suspicious compliance queries

#### Example Security Configuration
```python
from neuro_symbolic_law.core.enhanced_prover import EnhancedLegalProver

prover = EnhancedLegalProver(
    enable_security_logging=True,
    strict_input_validation=True,
    max_contract_size=1024*1024,  # 1MB limit
    rate_limit_per_minute=100,
    enable_audit_trail=True
)
```

### 🚀 Performance Tuning

#### Generation 3 Optimization
```python
# Optimize for high throughput
prover = ScalableLegalProver(
    initial_cache_size=5000,      # Larger initial cache
    max_cache_size=100000,        # Higher cache limit
    max_workers=16,               # More concurrent workers
    memory_threshold=0.75,        # Aggressive scaling
    enable_adaptive_caching=True  # Learn access patterns
)

# Manual optimization
optimization_results = prover.optimize_system()
print(f"Cache optimized: {optimization_results['cache_optimization']['size_changed']}")
print(f"Memory freed: {optimization_results['garbage_collection']['memory_freed']}")
```

#### Resource Scaling
```python
# Check and adjust resources
resource_info = prover.resource_manager.check_resources()

if resource_info['resource_status'] == 'high':
    print("High resource usage detected - consider scaling")
    # Auto-scaling will handle this automatically

# Get detailed performance stats
stats = prover.get_performance_metrics()
print(f"Requests per second: {stats['performance_stats']['requests_per_second']:.2f}")
```

### 📚 API Integration Examples

#### REST API Wrapper (Example)
```python
from flask import Flask, request, jsonify
from neuro_symbolic_law.core.scalable_prover import ScalableLegalProver
from neuro_symbolic_law.parsing.contract_parser import ContractParser
from neuro_symbolic_law.regulations import GDPR, AIAct

app = Flask(__name__)
parser = ContractParser(model='enhanced')
prover = ScalableLegalProver()

@app.route('/verify', methods=['POST'])
def verify_compliance():
    try:
        data = request.json
        contract_text = data['contract_text']
        regulation_type = data.get('regulation', 'gdpr')
        focus_areas = data.get('focus_areas', [])
        
        # Parse contract
        contract = parser.parse(contract_text)
        
        # Select regulation
        regulation = GDPR() if regulation_type.lower() == 'gdpr' else AIAct()
        
        # Verify compliance
        results = prover.verify_compliance(
            contract, 
            regulation,
            focus_areas=focus_areas if focus_areas else None
        )
        
        # Format response
        return jsonify({
            'status': 'success',
            'contract_id': contract.id,
            'regulation': regulation.name,
            'results': {req_id: {
                'compliant': result.compliant,
                'confidence': result.confidence,
                'status': result.status.value,
                'issue': result.issue
            } for req_id, result in results.items()}
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 400

@app.route('/health')
def health_check():
    health_checker = get_health_checker()
    health_report = health_checker.get_health_report()
    
    status_code = 200 if health_report['overall_status'] == 'healthy' else 503
    return jsonify(health_report), status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### 📈 Scaling Strategies

#### Horizontal Scaling
- Deploy multiple instances behind a load balancer
- Use Redis for shared caching across instances
- Implement sticky sessions if needed for cache efficiency

#### Vertical Scaling
- Increase memory for larger caches
- Add more CPU cores for concurrent processing
- Use SSD storage for faster I/O

#### Auto-scaling Configuration
```python
# Configure auto-scaling parameters
prover = ScalableLegalProver(
    max_workers=32,                # Maximum worker threads
    memory_threshold=0.8,          # Scale up threshold
    enable_concurrent_processing=True,
    initial_cache_size=2000,       # Start with larger cache
    max_cache_size=100000          # Allow large cache growth
)

# Monitor scaling metrics
metrics = prover.get_performance_metrics()
if metrics['resource_manager']['resource_status'] == 'critical':
    # Alert or trigger additional scaling
    pass
```

### 🔄 Maintenance and Updates

#### Regular Maintenance Tasks
- Monitor cache hit rates and adjust sizes
- Review error logs and security events
- Update regulation definitions as laws change
- Performance testing and optimization
- Security audits and dependency updates

#### Update Procedures
1. Test updates in staging environment
2. Backup current configuration and data
3. Deploy with blue-green or rolling updates
4. Monitor metrics during rollout
5. Rollback procedure if issues detected

### 📞 Support and Troubleshooting

#### Common Issues

**High Memory Usage**
- Reduce cache sizes
- Check for memory leaks
- Monitor garbage collection

**Low Cache Hit Rates**
- Enable adaptive caching
- Increase cache size
- Check cache eviction policies

**Performance Issues**
- Enable concurrent processing
- Increase worker threads
- Optimize contract parsing

#### Logging Configuration
```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log'),
        logging.StreamHandler()
    ]
)

# Enable debug logging for specific components
logging.getLogger('neuro_symbolic_law.core.scalable_prover').setLevel(logging.DEBUG)
logging.getLogger('neuro_symbolic_law.reasoning.solver').setLevel(logging.INFO)
```

---

## Summary

The Neuro-Symbolic Law Prover provides a comprehensive solution for automated legal compliance verification with three generations of progressive enhancement:

1. **Generation 1**: Core functionality for basic compliance checking
2. **Generation 2**: Enhanced robustness with monitoring and error handling
3. **Generation 3**: Scalable architecture with adaptive optimization

Choose the appropriate generation based on your performance and scalability requirements. All generations maintain compatibility and can be deployed using the configurations provided in this guide.