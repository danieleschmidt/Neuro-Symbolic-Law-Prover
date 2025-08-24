#!/usr/bin/env python3
"""
Health check script for Docker container
Validates system components and service availability
"""

import sys
import os
import time
import json
import logging
from typing import Dict, Any

# Add source path
sys.path.insert(0, '/app/src')

def setup_logging():
    """Setup basic logging for health check"""
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors
        format='%(asctime)s - HEALTH - %(levelname)s - %(message)s'
    )

def check_imports() -> Dict[str, Any]:
    """Check if core modules can be imported"""
    result = {
        'status': 'healthy',
        'details': {},
        'issues': []
    }
    
    try:
        # Test core imports
        from neuro_symbolic_law.core.scalable_prover import ScalableLegalProver
        result['details']['core_imports'] = 'success'
    except ImportError as e:
        result['status'] = 'unhealthy'
        result['issues'].append(f'Core import failed: {str(e)}')
        result['details']['core_imports'] = 'failed'
    
    try:
        # Test parsing imports
        from neuro_symbolic_law.parsing.contract_parser import ContractParser
        result['details']['parsing_imports'] = 'success'
    except ImportError as e:
        result['status'] = 'degraded'
        result['issues'].append(f'Parsing import failed: {str(e)}')
        result['details']['parsing_imports'] = 'failed'
    
    try:
        # Test regulation imports
        from neuro_symbolic_law.regulations.gdpr import GDPR
        result['details']['regulation_imports'] = 'success'
    except ImportError as e:
        result['status'] = 'degraded'
        result['issues'].append(f'Regulation import failed: {str(e)}')
        result['details']['regulation_imports'] = 'failed'
    
    return result

def check_system_health() -> Dict[str, Any]:
    """Check system health via monitoring module"""
    result = {
        'status': 'healthy',
        'details': {},
        'issues': []
    }
    
    try:
        from neuro_symbolic_law.core.monitoring import get_health_checker
        
        health_checker = get_health_checker()
        health_report = health_checker.get_health_report()
        
        result['details']['overall_status'] = health_report['overall_status']
        result['details']['check_count'] = len(health_report['checks'])
        
        # Check individual health components
        unhealthy_checks = []
        for check_name, check_result in health_report['checks'].items():
            if check_result['status'] in ['unhealthy', 'degraded']:
                unhealthy_checks.append(f"{check_name}: {check_result['status']}")
        
        if unhealthy_checks:
            result['status'] = 'degraded'
            result['issues'].extend(unhealthy_checks)
            result['details']['unhealthy_checks'] = len(unhealthy_checks)
        else:
            result['details']['unhealthy_checks'] = 0
        
        # Overall system health determines result status
        if health_report['overall_status'] == 'unhealthy':
            result['status'] = 'unhealthy'
        elif health_report['overall_status'] == 'degraded' and result['status'] == 'healthy':
            result['status'] = 'degraded'
        
    except Exception as e:
        result['status'] = 'degraded'
        result['issues'].append(f'Health check system failed: {str(e)}')
        result['details']['health_check_error'] = str(e)
    
    return result

def check_basic_functionality() -> Dict[str, Any]:
    """Test basic system functionality"""
    result = {
        'status': 'healthy',
        'details': {},
        'issues': []
    }
    
    try:
        # Test contract parsing
        from neuro_symbolic_law.parsing.contract_parser import ContractParser
        
        parser = ContractParser()
        test_contract = "Test privacy policy for health check validation."
        
        start_time = time.time()
        contract = parser.parse(test_contract, contract_id='health_check')
        parse_time = time.time() - start_time
        
        result['details']['parsing_time_ms'] = round(parse_time * 1000, 2)
        result['details']['parsed_clauses'] = len(contract.clauses)
        
        if parse_time > 5.0:  # More than 5 seconds is too slow
            result['status'] = 'degraded'
            result['issues'].append(f'Slow parsing performance: {parse_time:.2f}s')
        
    except Exception as e:
        result['status'] = 'unhealthy'
        result['issues'].append(f'Basic parsing functionality failed: {str(e)}')
        result['details']['parsing_error'] = str(e)
        return result
    
    try:
        # Test basic verification (lightweight)
        from neuro_symbolic_law.core.enhanced_prover import EnhancedLegalProver
        from neuro_symbolic_law.regulations.gdpr import GDPR
        
        prover = EnhancedLegalProver(cache_enabled=True)
        gdpr = GDPR()
        
        start_time = time.time()
        # Run a lightweight verification for health check
        results = prover.verify_compliance(contract, gdpr, focus_areas=['data_minimization'])
        verification_time = time.time() - start_time
        
        result['details']['verification_time_ms'] = round(verification_time * 1000, 2)
        result['details']['requirements_checked'] = len(results)
        
        if verification_time > 10.0:  # More than 10 seconds is concerning
            result['status'] = 'degraded'
            result['issues'].append(f'Slow verification performance: {verification_time:.2f}s')
        
        # Check if any results were returned
        if not results:
            result['status'] = 'degraded'
            result['issues'].append('No verification results returned')
        
    except Exception as e:
        result['status'] = 'degraded'  # Not critical for health check
        result['issues'].append(f'Basic verification functionality failed: {str(e)}')
        result['details']['verification_error'] = str(e)
    
    return result

def check_file_system() -> Dict[str, Any]:
    """Check file system access and permissions"""
    result = {
        'status': 'healthy',
        'details': {},
        'issues': []
    }
    
    # Check required directories
    required_dirs = ['/app', '/app/logs', '/app/data', '/app/cache']
    
    for directory in required_dirs:
        try:
            if not os.path.exists(directory):
                result['status'] = 'degraded'
                result['issues'].append(f'Missing directory: {directory}')
            elif not os.access(directory, os.R_OK | os.W_OK):
                result['status'] = 'degraded'
                result['issues'].append(f'No read/write access to: {directory}')
            else:
                result['details'][f'{directory}_access'] = 'ok'
        except Exception as e:
            result['status'] = 'degraded'
            result['issues'].append(f'Directory check failed for {directory}: {str(e)}')
    
    # Test file operations
    try:
        test_file = '/app/logs/healthcheck.tmp'
        with open(test_file, 'w') as f:
            f.write('health check test')
        
        with open(test_file, 'r') as f:
            content = f.read()
        
        os.remove(test_file)
        
        if content != 'health check test':
            result['status'] = 'degraded'
            result['issues'].append('File read/write test failed')
        else:
            result['details']['file_operations'] = 'ok'
            
    except Exception as e:
        result['status'] = 'degraded'
        result['issues'].append(f'File operations test failed: {str(e)}')
    
    return result

def check_network_connectivity() -> Dict[str, Any]:
    """Check network connectivity if Redis is configured"""
    result = {
        'status': 'healthy',
        'details': {},
        'issues': []
    }
    
    redis_urls = os.environ.get('NSL_REDIS_URLS')
    
    if redis_urls:
        # If Redis is configured, try to test connectivity
        try:
            import socket
            
            redis_url_list = redis_urls.split(',')
            connected_instances = 0
            
            for redis_url in redis_url_list:
                try:
                    # Parse Redis URL
                    if '://' in redis_url:
                        host_port = redis_url.split('://')[1]
                    else:
                        host_port = redis_url
                    
                    if ':' in host_port:
                        host, port = host_port.split(':')
                        port = int(port)
                    else:
                        host = host_port
                        port = 6379
                    
                    # Test TCP connection
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)  # 2 second timeout
                    result_code = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result_code == 0:
                        connected_instances += 1
                    else:
                        result['issues'].append(f'Redis connection failed: {host}:{port}')
                        
                except Exception as e:
                    result['issues'].append(f'Redis connectivity test error: {str(e)}')
            
            result['details']['redis_instances_configured'] = len(redis_url_list)
            result['details']['redis_instances_connected'] = connected_instances
            
            # If less than half of Redis instances are accessible, mark as degraded
            if connected_instances < len(redis_url_list) / 2:
                result['status'] = 'degraded'
            
        except Exception as e:
            result['status'] = 'degraded'
            result['issues'].append(f'Redis connectivity check failed: {str(e)}')
    else:
        result['details']['redis_configured'] = False
    
    return result

def main():
    """Main health check function"""
    setup_logging()
    
    print("🏥 Neuro-Symbolic Legal Reasoning System - Health Check")
    print("=" * 60)
    
    # Initialize overall health status
    overall_status = 'healthy'
    all_results = {}
    
    # Run all health checks
    checks = [
        ('imports', check_imports),
        ('system_health', check_system_health),
        ('functionality', check_basic_functionality),
        ('filesystem', check_file_system),
        ('network', check_network_connectivity)
    ]
    
    for check_name, check_func in checks:
        try:
            print(f"Checking {check_name}...")
            result = check_func()
            all_results[check_name] = result
            
            # Update overall status
            if result['status'] == 'unhealthy':
                overall_status = 'unhealthy'
            elif result['status'] == 'degraded' and overall_status == 'healthy':
                overall_status = 'degraded'
            
            # Print immediate feedback
            status_emoji = {
                'healthy': '✅',
                'degraded': '⚠️',
                'unhealthy': '❌'
            }
            print(f"  {status_emoji.get(result['status'], '❓')} {check_name}: {result['status']}")
            
            if result['issues']:
                for issue in result['issues']:
                    print(f"    - {issue}")
            
        except Exception as e:
            print(f"  ❌ {check_name}: failed with exception: {str(e)}")
            overall_status = 'unhealthy'
            all_results[check_name] = {
                'status': 'unhealthy',
                'error': str(e),
                'issues': [f'Health check exception: {str(e)}']
            }
    
    # Print summary
    print("=" * 60)
    print(f"Overall Health Status: {overall_status.upper()}")
    
    # Create health report
    health_report = {
        'timestamp': time.time(),
        'overall_status': overall_status,
        'checks': all_results,
        'container_info': {
            'python_version': sys.version.split()[0],
            'environment': os.environ.get('NSL_ENV', 'unknown'),
            'workers': os.environ.get('NSL_WORKERS', 'unknown'),
            'cache_size': os.environ.get('NSL_CACHE_SIZE', 'unknown')
        }
    }
    
    # Write health report to file for monitoring
    try:
        with open('/app/logs/health-report.json', 'w') as f:
            json.dump(health_report, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not write health report: {e}")
    
    # Exit with appropriate code
    exit_codes = {
        'healthy': 0,
        'degraded': 0,  # Still pass health check but log concerns
        'unhealthy': 1
    }
    
    exit_code = exit_codes.get(overall_status, 1)
    
    if exit_code == 0:
        print("✅ Health check PASSED")
    else:
        print("❌ Health check FAILED")
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()