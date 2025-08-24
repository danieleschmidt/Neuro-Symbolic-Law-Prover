#!/bin/bash
set -e

# Neuro-Symbolic Legal Reasoning System - Docker Entry Point
# Handles service initialization, configuration, and startup

echo "🚀 Starting Neuro-Symbolic Legal Reasoning System"
echo "Version: ${NSL_VERSION:-latest}"
echo "Environment: ${NSL_ENV:-production}"

# Function to log with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check required environment variables
check_env() {
    local var_name=$1
    local var_value=${!var_name}
    
    if [[ -z "$var_value" ]]; then
        log "ERROR: Required environment variable $var_name is not set"
        exit 1
    fi
}

# Function to wait for Redis availability
wait_for_redis() {
    if [[ -n "$NSL_REDIS_URLS" ]]; then
        log "Checking Redis connectivity..."
        
        # Parse Redis URLs and check connectivity
        IFS=',' read -ra REDIS_URLS <<< "$NSL_REDIS_URLS"
        for redis_url in "${REDIS_URLS[@]}"; do
            # Extract host and port from Redis URL
            redis_host=$(echo "$redis_url" | sed -n 's|redis://\([^:]*\).*|\1|p')
            redis_port=$(echo "$redis_url" | sed -n 's|redis://[^:]*:\([0-9]*\).*|\1|p')
            
            if [[ -z "$redis_port" ]]; then
                redis_port=6379
            fi
            
            log "Checking Redis at $redis_host:$redis_port"
            
            # Wait up to 30 seconds for Redis to be available
            timeout=30
            while ! timeout 1 bash -c "echo > /dev/tcp/$redis_host/$redis_port" 2>/dev/null; do
                timeout=$((timeout - 1))
                if [[ $timeout -eq 0 ]]; then
                    log "ERROR: Redis at $redis_host:$redis_port is not available after 30 seconds"
                    exit 1
                fi
                log "Waiting for Redis at $redis_host:$redis_port... ($timeout seconds remaining)"
                sleep 1
            done
            
            log "✅ Redis at $redis_host:$redis_port is available"
        done
    else
        log "No Redis URLs configured, skipping Redis connectivity check"
    fi
}

# Function to initialize application
initialize_app() {
    log "Initializing application..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/data /app/cache /app/metrics
    
    # Set permissions
    chmod 755 /app/logs /app/data /app/cache /app/metrics
    
    # Validate Python environment
    log "Validating Python environment..."
    python -c "
import sys
print(f'Python version: {sys.version}')

try:
    from neuro_symbolic_law import __version__
    print(f'NSL version: {__version__}')
except ImportError as e:
    print(f'ERROR: Failed to import NSL: {e}')
    sys.exit(1)

# Test core components
try:
    from neuro_symbolic_law.core.scalable_prover import ScalableLegalProver
    from neuro_symbolic_law.parsing.contract_parser import ContractParser
    from neuro_symbolic_law.regulations.gdpr import GDPR
    print('✅ Core components loaded successfully')
except ImportError as e:
    print(f'ERROR: Failed to load core components: {e}')
    sys.exit(1)
    
# Test system health
try:
    from neuro_symbolic_law.core.monitoring import get_health_checker
    health_checker = get_health_checker()
    health = health_checker.get_health_report()
    print(f'System health: {health[\"overall_status\"]}')
except Exception as e:
    print(f'WARNING: Health check failed: {e}')
"
    
    if [[ $? -ne 0 ]]; then
        log "ERROR: Application initialization failed"
        exit 1
    fi
    
    log "✅ Application initialized successfully"
}

# Function to configure logging
setup_logging() {
    log "Setting up logging configuration..."
    
    # Create logging configuration
    cat > /app/logging.conf << EOF
[loggers]
keys=root,nsl

[handlers]
keys=console,file

[formatters]
keys=detailed,simple

[logger_root]
level=${NSL_LOG_LEVEL:-INFO}
handlers=console,file

[logger_nsl]
level=${NSL_LOG_LEVEL:-INFO}
handlers=console,file
qualname=neuro_symbolic_law
propagate=0

[handler_console]
class=StreamHandler
level=${NSL_LOG_LEVEL:-INFO}
formatter=simple
args=(sys.stdout,)

[handler_file]
class=handlers.RotatingFileHandler
level=${NSL_LOG_LEVEL:-INFO}
formatter=detailed
args=('/app/logs/nsl.log', 'a', 10485760, 5)

[formatter_simple]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s

[formatter_detailed]
format=%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s
EOF

    export PYTHONPATH="/app/src:$PYTHONPATH"
    export NSL_LOGGING_CONFIG="/app/logging.conf"
    
    log "✅ Logging configured"
}

# Function to start the main service
start_service() {
    log "Starting Neuro-Symbolic Legal Reasoning Service..."
    
    # Build command based on configuration
    cmd_args=(
        "--host" "${NSL_BIND_HOST:-0.0.0.0}"
        "--port" "${NSL_BIND_PORT:-8000}"
        "--workers" "${NSL_WORKERS:-8}"
        "--max-workers" "${NSL_MAX_WORKERS:-32}"
        "--cache-size" "${NSL_CACHE_SIZE:-10000}"
        "--log-level" "${NSL_LOG_LEVEL:-info}"
    )
    
    # Add optional configurations
    if [[ "${NSL_ENABLE_AUTO_SCALING:-true}" == "true" ]]; then
        cmd_args+=("--enable-auto-scaling")
    fi
    
    if [[ "${NSL_ENABLE_ADAPTIVE_CACHE:-true}" == "true" ]]; then
        cmd_args+=("--enable-adaptive-cache")
    fi
    
    if [[ "${NSL_ENABLE_MONITORING:-true}" == "true" ]]; then
        cmd_args+=("--enable-monitoring" "--metrics-port" "${NSL_METRICS_PORT:-9090}")
    fi
    
    if [[ -n "$NSL_REDIS_URLS" ]]; then
        cmd_args+=("--redis-urls" "$NSL_REDIS_URLS")
    fi
    
    log "Service command: python -m neuro_symbolic_law.server ${cmd_args[*]}"
    
    # Execute the service
    exec python -m neuro_symbolic_law.server "${cmd_args[@]}"
}

# Function to start worker process
start_worker() {
    log "Starting NSL Worker Process..."
    
    worker_args=(
        "--worker-id" "${NSL_WORKER_ID:-worker-$RANDOM}"
        "--queue-url" "${NSL_QUEUE_URL:-redis://localhost:6379}"
        "--log-level" "${NSL_LOG_LEVEL:-info}"
    )
    
    if [[ -n "$NSL_REDIS_URLS" ]]; then
        worker_args+=("--redis-urls" "$NSL_REDIS_URLS")
    fi
    
    exec python -m neuro_symbolic_law.worker "${worker_args[@]}"
}

# Function to run one-time tasks
run_task() {
    local task_name=$1
    shift
    local task_args=("$@")
    
    log "Running task: $task_name"
    
    case "$task_name" in
        "migrate")
            python -m neuro_symbolic_law.migrations "${task_args[@]}"
            ;;
        "test")
            python -m pytest tests/ "${task_args[@]}"
            ;;
        "validate")
            python -c "
from neuro_symbolic_law.core.scalable_prover import ScalableLegalProver
from neuro_symbolic_law.parsing.contract_parser import ContractParser
from neuro_symbolic_law.regulations.gdpr import GDPR

# Basic validation
parser = ContractParser()
prover = ScalableLegalProver()
gdpr = GDPR()

test_contract = 'Test privacy policy with data processing for service purposes.'
contract = parser.parse(test_contract)
results = prover.verify_compliance(contract, gdpr)

print(f'✅ Validation successful: {len(results)} requirements checked')
"
            ;;
        "health")
            python healthcheck.py
            ;;
        *)
            log "ERROR: Unknown task: $task_name"
            exit 1
            ;;
    esac
}

# Main execution logic
main() {
    # Parse command
    local command=${1:-serve}
    shift || true
    
    # Setup logging first
    setup_logging
    
    case "$command" in
        "serve")
            # Check dependencies
            wait_for_redis
            
            # Initialize application
            initialize_app
            
            # Start main service
            start_service
            ;;
            
        "worker")
            # Check dependencies
            wait_for_redis
            
            # Initialize application
            initialize_app
            
            # Start worker
            start_worker
            ;;
            
        "task")
            # Check dependencies for tasks that need them
            if [[ "$1" != "health" ]]; then
                wait_for_redis
                initialize_app
            fi
            
            # Run specified task
            run_task "$@"
            ;;
            
        "bash")
            # Interactive bash shell
            exec /bin/bash
            ;;
            
        "help"|"--help"|"-h")
            cat << EOF
Neuro-Symbolic Legal Reasoning System - Docker Entry Point

USAGE:
    docker run neuro-symbolic-law:latest [COMMAND] [OPTIONS]

COMMANDS:
    serve       Start the main API service (default)
    worker      Start a worker process
    task        Run a one-time task
    bash        Start interactive bash shell
    help        Show this help message

TASKS:
    migrate     Run database migrations
    test        Run test suite
    validate    Validate system functionality
    health      Run health check

ENVIRONMENT VARIABLES:
    NSL_WORKERS             Number of worker threads (default: 8)
    NSL_MAX_WORKERS         Maximum worker threads (default: 32)
    NSL_BIND_HOST           Bind host (default: 0.0.0.0)
    NSL_BIND_PORT           Bind port (default: 8000)
    NSL_METRICS_PORT        Metrics port (default: 9090)
    NSL_CACHE_SIZE          Initial cache size (default: 10000)
    NSL_MAX_CACHE_SIZE      Maximum cache size (default: 100000)
    NSL_REDIS_URLS          Redis cluster URLs (comma-separated)
    NSL_LOG_LEVEL           Log level (default: INFO)
    NSL_ENABLE_AUTO_SCALING Enable auto-scaling (default: true)
    NSL_ENABLE_MONITORING   Enable monitoring (default: true)

EXAMPLES:
    # Start service with custom configuration
    docker run -e NSL_WORKERS=16 -e NSL_CACHE_SIZE=20000 nsl:latest serve
    
    # Run worker process
    docker run -e NSL_REDIS_URLS=redis://redis:6379 nsl:latest worker
    
    # Run health check
    docker run nsl:latest task health
    
    # Run validation
    docker run nsl:latest task validate
EOF
            ;;
            
        *)
            log "ERROR: Unknown command: $command"
            log "Use 'help' command for usage information"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"