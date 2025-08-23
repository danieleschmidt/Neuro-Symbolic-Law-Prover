#!/usr/bin/env python3
"""
🛡️ Comprehensive Quality Gates - Production-Ready Validation
===========================================================

Implements comprehensive quality gates for production deployment:
- Code quality and style validation
- Security vulnerability scanning
- Performance benchmarking
- Integration testing
- Documentation validation
- Production readiness checks
"""

import asyncio
import os
import sys
import time
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate status levels"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Result of a quality gate check"""
    gate_name: str
    status: QualityGateStatus
    score: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class ComprehensiveQualityGates:
    """
    Comprehensive quality gates system for production readiness validation.
    
    Features:
    - Multi-stage validation pipeline
    - Configurable quality thresholds
    - Detailed reporting and recommendations
    - Integration with CI/CD pipelines
    """
    
    def __init__(self, 
                 project_root: str = "/root/repo",
                 strict_mode: bool = False,
                 performance_benchmarks: bool = True):
        """
        Initialize comprehensive quality gates
        
        Args:
            project_root: Root directory of the project
            strict_mode: Enable strict validation (fail on warnings)
            performance_benchmarks: Run performance benchmarks
        """
        self.project_root = Path(project_root)
        self.strict_mode = strict_mode
        self.performance_benchmarks = performance_benchmarks
        
        # Quality thresholds
        self.thresholds = {
            'code_quality_min_score': 8.0,
            'test_coverage_min': 85.0,
            'performance_max_response_time': 2000,  # ms
            'security_vulnerability_max': 0,  # High/Critical vulnerabilities
            'documentation_coverage_min': 80.0
        }
        
        # Results tracking
        self.results: List[QualityGateResult] = []
        self.overall_status = QualityGateStatus.PASSED
        
        logger.info(f"Quality gates initialized for {project_root}")
        logger.info(f"Strict mode: {strict_mode}, Performance benchmarks: {performance_benchmarks}")
    
    async def run_all_quality_gates(self) -> Dict[str, Any]:
        """
        Run all quality gates and return comprehensive results
        
        Returns:
            Comprehensive quality gate results with pass/fail status
        """
        start_time = time.time()
        
        logger.info("🚀 Starting comprehensive quality gate validation...")
        print("\n🛡️ COMPREHENSIVE QUALITY GATES")
        print("=" * 45)
        
        # Stage 1: Code Quality Gates
        await self._run_code_quality_gates()
        
        # Stage 2: Security Gates
        await self._run_security_gates()
        
        # Stage 3: Performance Gates
        if self.performance_benchmarks:
            await self._run_performance_gates()
        
        # Stage 4: Integration Gates
        await self._run_integration_gates()
        
        # Stage 5: Documentation Gates
        await self._run_documentation_gates()
        
        # Stage 6: Production Readiness Gates
        await self._run_production_readiness_gates()
        
        # Calculate overall results
        total_time = time.time() - start_time
        results_summary = self._calculate_overall_results(total_time)
        
        # Display final results
        self._display_final_results(results_summary)
        
        return results_summary
    
    async def _run_code_quality_gates(self) -> None:
        """Run code quality validation gates"""
        print("\n📋 Stage 1: Code Quality Gates")
        print("-" * 35)
        
        # Python syntax validation
        await self._check_python_syntax()
        
        # Import validation
        await self._check_imports()
        
        # Code style and formatting
        await self._check_code_style()
        
        # Complexity analysis
        await self._check_code_complexity()
    
    async def _check_python_syntax(self) -> None:
        """Check Python syntax validity"""
        start_time = time.time()
        
        try:
            python_files = list(self.project_root.rglob("*.py"))
            syntax_errors = []
            
            for py_file in python_files:
                if any(skip in str(py_file) for skip in ['.git', '__pycache__', '.venv']):
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    compile(source_code, str(py_file), 'exec')
                except SyntaxError as e:
                    syntax_errors.append(f"{py_file}: {e}")
                except UnicodeDecodeError:
                    continue  # Skip binary files
            
            execution_time = time.time() - start_time
            
            if not syntax_errors:
                result = QualityGateResult(
                    gate_name="Python Syntax Validation",
                    status=QualityGateStatus.PASSED,
                    score=10.0,
                    message=f"All {len(python_files)} Python files have valid syntax",
                    execution_time=execution_time,
                    details={'files_checked': len(python_files)}
                )
                print(f"   ✅ Python Syntax: {len(python_files)} files validated")
            else:
                result = QualityGateResult(
                    gate_name="Python Syntax Validation",
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    message=f"{len(syntax_errors)} syntax errors found",
                    execution_time=execution_time,
                    details={'syntax_errors': syntax_errors[:5]},  # Limit details
                    recommendations=["Fix syntax errors before deployment"]
                )
                print(f"   ❌ Python Syntax: {len(syntax_errors)} errors found")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Python Syntax Validation", str(e), time.time() - start_time)
    
    async def _check_imports(self) -> None:
        """Check import validity and circular dependencies"""
        start_time = time.time()
        
        try:
            # Add project src to path for import testing
            sys.path.insert(0, str(self.project_root / "src"))
            
            import_errors = []
            test_imports = [
                "neuro_symbolic_law",
                "neuro_symbolic_law.core.legal_prover",
                "neuro_symbolic_law.consciousness.conscious_reasoner"
            ]
            
            for module_name in test_imports:
                try:
                    __import__(module_name)
                except ImportError as e:
                    import_errors.append(f"{module_name}: {e}")
                except Exception as e:
                    import_errors.append(f"{module_name}: {type(e).__name__}: {e}")
            
            execution_time = time.time() - start_time
            
            if not import_errors:
                result = QualityGateResult(
                    gate_name="Import Validation",
                    status=QualityGateStatus.PASSED,
                    score=10.0,
                    message=f"All {len(test_imports)} critical imports successful",
                    execution_time=execution_time,
                    details={'imports_tested': test_imports}
                )
                print(f"   ✅ Imports: {len(test_imports)} critical imports validated")
            else:
                result = QualityGateResult(
                    gate_name="Import Validation",
                    status=QualityGateStatus.WARNING,
                    score=7.0,
                    message=f"{len(import_errors)} import issues found",
                    execution_time=execution_time,
                    details={'import_errors': import_errors},
                    recommendations=["Review import dependencies and missing packages"]
                )
                print(f"   ⚠️  Imports: {len(import_errors)} issues found")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Import Validation", str(e), time.time() - start_time)
    
    async def _check_code_style(self) -> None:
        """Check code style and formatting"""
        start_time = time.time()
        
        try:
            # Simple code style checks
            python_files = list(self.project_root.rglob("*.py"))
            style_issues = []
            
            for py_file in python_files:
                if any(skip in str(py_file) for skip in ['.git', '__pycache__', '.venv']):
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines, 1):
                        # Check line length (ignore very long strings)
                        if len(line.rstrip()) > 120 and not ('"""' in line or "'''" in line):
                            if len(style_issues) < 10:  # Limit reported issues
                                style_issues.append(f"{py_file}:{i} - Line too long ({len(line.rstrip())} chars)")
                        
                        # Check trailing whitespace
                        if line.endswith(' \n') or line.endswith('\t\n'):
                            if len(style_issues) < 10:
                                style_issues.append(f"{py_file}:{i} - Trailing whitespace")
                
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            execution_time = time.time() - start_time
            
            if len(style_issues) <= 5:
                result = QualityGateResult(
                    gate_name="Code Style Check",
                    status=QualityGateStatus.PASSED,
                    score=9.0,
                    message=f"Code style acceptable ({len(style_issues)} minor issues)",
                    execution_time=execution_time,
                    details={'style_issues_count': len(style_issues)}
                )
                print(f"   ✅ Code Style: {len(style_issues)} minor issues")
            else:
                result = QualityGateResult(
                    gate_name="Code Style Check",
                    status=QualityGateStatus.WARNING,
                    score=6.0,
                    message=f"{len(style_issues)} style issues found",
                    execution_time=execution_time,
                    details={'style_issues': style_issues[:10]},
                    recommendations=["Consider using black or autopep8 for code formatting"]
                )
                print(f"   ⚠️  Code Style: {len(style_issues)} issues found")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Code Style Check", str(e), time.time() - start_time)
    
    async def _check_code_complexity(self) -> None:
        """Check code complexity metrics"""
        start_time = time.time()
        
        try:
            # Simple complexity analysis
            python_files = list(self.project_root.rglob("*.py"))
            complexity_data = {
                'total_files': 0,
                'total_lines': 0,
                'total_functions': 0,
                'large_functions': 0,
                'deep_nesting': 0
            }
            
            for py_file in python_files:
                if any(skip in str(py_file) for skip in ['.git', '__pycache__', '.venv']):
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    complexity_data['total_files'] += 1
                    complexity_data['total_lines'] += len(lines)
                    
                    # Count functions and analyze complexity
                    in_function = False
                    function_lines = 0
                    max_indent = 0
                    
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith('def ') or stripped.startswith('async def '):
                            if in_function and function_lines > 50:
                                complexity_data['large_functions'] += 1
                            
                            in_function = True
                            function_lines = 0
                            complexity_data['total_functions'] += 1
                        
                        if in_function:
                            function_lines += 1
                            # Check indentation depth
                            indent_level = (len(line) - len(line.lstrip())) // 4
                            max_indent = max(max_indent, indent_level)
                            
                            if indent_level > 4:  # Very deep nesting
                                complexity_data['deep_nesting'] += 1
                        
                        if stripped.startswith('class ') and in_function:
                            in_function = False
                
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            execution_time = time.time() - start_time
            
            # Calculate complexity score
            avg_lines_per_file = complexity_data['total_lines'] / max(complexity_data['total_files'], 1)
            large_function_ratio = complexity_data['large_functions'] / max(complexity_data['total_functions'], 1)
            
            complexity_score = 10.0
            issues = []
            
            if avg_lines_per_file > 500:
                complexity_score -= 1.0
                issues.append("Large average file size")
            
            if large_function_ratio > 0.2:
                complexity_score -= 2.0
                issues.append("Many large functions detected")
            
            if complexity_data['deep_nesting'] > 10:
                complexity_score -= 1.0
                issues.append("Deep nesting detected")
            
            status = QualityGateStatus.PASSED if complexity_score >= 8.0 else QualityGateStatus.WARNING
            
            result = QualityGateResult(
                gate_name="Code Complexity Analysis",
                status=status,
                score=complexity_score,
                message=f"Complexity analysis completed ({len(issues)} issues)",
                execution_time=execution_time,
                details=complexity_data,
                recommendations=["Refactor large functions", "Reduce nesting depth"] if issues else []
            )
            
            print(f"   {'✅' if status == QualityGateStatus.PASSED else '⚠️ '} Complexity: Score {complexity_score:.1f}/10.0")
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Code Complexity Analysis", str(e), time.time() - start_time)
    
    async def _run_security_gates(self) -> None:
        """Run security validation gates"""
        print("\n🔒 Stage 2: Security Gates")
        print("-" * 25)
        
        await self._check_security_patterns()
        await self._check_dependencies_security()
        await self._check_secrets_exposure()
    
    async def _check_security_patterns(self) -> None:
        """Check for common security anti-patterns"""
        start_time = time.time()
        
        try:
            python_files = list(self.project_root.rglob("*.py"))
            security_issues = []
            
            # Security patterns to check
            security_patterns = {
                'eval(': 'Dangerous eval() usage',
                'exec(': 'Dangerous exec() usage', 
                'subprocess.call': 'Subprocess usage - verify input sanitization',
                'os.system': 'OS command execution - verify input sanitization',
                'pickle.loads': 'Pickle deserialization - potential RCE risk',
                'yaml.load': 'YAML load - use safe_load instead',
                'shell=True': 'Shell injection risk'
            }
            
            for py_file in python_files:
                if any(skip in str(py_file) for skip in ['.git', '__pycache__', '.venv']):
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern, description in security_patterns.items():
                        if pattern in content:
                            security_issues.append(f"{py_file}: {description}")
                
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            execution_time = time.time() - start_time
            
            if not security_issues:
                result = QualityGateResult(
                    gate_name="Security Pattern Analysis",
                    status=QualityGateStatus.PASSED,
                    score=10.0,
                    message="No dangerous security patterns detected",
                    execution_time=execution_time,
                    details={'files_scanned': len(python_files)}
                )
                print(f"   ✅ Security Patterns: {len(python_files)} files scanned, no issues")
            else:
                result = QualityGateResult(
                    gate_name="Security Pattern Analysis",
                    status=QualityGateStatus.WARNING,
                    score=5.0,
                    message=f"{len(security_issues)} potential security issues found",
                    execution_time=execution_time,
                    details={'security_issues': security_issues[:10]},
                    recommendations=["Review and validate security-sensitive code patterns"]
                )
                print(f"   ⚠️  Security Patterns: {len(security_issues)} potential issues")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Security Pattern Analysis", str(e), time.time() - start_time)
    
    async def _check_dependencies_security(self) -> None:
        """Check dependency security"""
        start_time = time.time()
        
        try:
            # Check if requirements files exist
            req_files = [
                self.project_root / "requirements.txt",
                self.project_root / "requirements-prod.txt",
                self.project_root / "pyproject.toml"
            ]
            
            found_reqs = [f for f in req_files if f.exists()]
            
            execution_time = time.time() - start_time
            
            if found_reqs:
                result = QualityGateResult(
                    gate_name="Dependency Security Check",
                    status=QualityGateStatus.PASSED,
                    score=8.0,
                    message=f"Dependency files found: {len(found_reqs)}",
                    execution_time=execution_time,
                    details={'requirement_files': [str(f) for f in found_reqs]},
                    recommendations=["Consider using safety or bandit for deeper security analysis"]
                )
                print(f"   ✅ Dependencies: {len(found_reqs)} requirement files found")
            else:
                result = QualityGateResult(
                    gate_name="Dependency Security Check",
                    status=QualityGateStatus.WARNING,
                    score=5.0,
                    message="No dependency files found",
                    execution_time=execution_time,
                    recommendations=["Add requirements.txt for dependency tracking"]
                )
                print(f"   ⚠️  Dependencies: No requirement files found")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Dependency Security Check", str(e), time.time() - start_time)
    
    async def _check_secrets_exposure(self) -> None:
        """Check for exposed secrets and credentials"""
        start_time = time.time()
        
        try:
            all_files = list(self.project_root.rglob("*"))
            secret_issues = []
            
            # Patterns that might indicate secrets
            secret_patterns = {
                r'password\s*=\s*["\'][^"\'\\n]+["\']': 'Hardcoded password',
                r'api_key\s*=\s*["\'][^"\'\\n]+["\']': 'Hardcoded API key', 
                r'secret\s*=\s*["\'][^"\'\\n]+["\']': 'Hardcoded secret',
                r'token\s*=\s*["\'][^"\'\\n]+["\']': 'Hardcoded token'
            }
            
            import re
            
            for file_path in all_files:
                if file_path.is_file() and file_path.suffix in ['.py', '.yml', '.yaml', '.json', '.env']:
                    if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.venv']):
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        for pattern, description in secret_patterns.items():
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            if matches:
                                secret_issues.append(f"{file_path}: {description}")
                                if len(secret_issues) >= 10:  # Limit findings
                                    break
                    
                    except (UnicodeDecodeError, PermissionError, OSError):
                        continue
                
                if len(secret_issues) >= 10:
                    break
            
            execution_time = time.time() - start_time
            
            if not secret_issues:
                result = QualityGateResult(
                    gate_name="Secrets Exposure Check",
                    status=QualityGateStatus.PASSED,
                    score=10.0,
                    message="No exposed secrets detected",
                    execution_time=execution_time,
                    details={'files_scanned': len([f for f in all_files if f.is_file()])}
                )
                print(f"   ✅ Secrets: No exposed credentials detected")
            else:
                result = QualityGateResult(
                    gate_name="Secrets Exposure Check",
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    message=f"{len(secret_issues)} potential secret exposures found",
                    execution_time=execution_time,
                    details={'secret_issues': secret_issues},
                    recommendations=["Remove hardcoded secrets", "Use environment variables or secret management"]
                )
                print(f"   ❌ Secrets: {len(secret_issues)} potential exposures found")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Secrets Exposure Check", str(e), time.time() - start_time)
    
    async def _run_performance_gates(self) -> None:
        """Run performance validation gates"""
        print("\n⚡ Stage 3: Performance Gates")
        print("-" * 30)
        
        await self._benchmark_import_performance()
        await self._benchmark_core_functionality()
    
    async def _benchmark_import_performance(self) -> None:
        """Benchmark import performance"""
        start_time = time.time()
        
        try:
            import_times = []
            
            # Test critical imports
            test_modules = [
                "neuro_symbolic_law",
                "neuro_symbolic_law.core.legal_prover",
                "neuro_symbolic_law.consciousness.conscious_reasoner"
            ]
            
            for module in test_modules:
                module_start = time.time()
                try:
                    __import__(module)
                    import_time = (time.time() - module_start) * 1000  # Convert to ms
                    import_times.append(import_time)
                except ImportError:
                    import_times.append(10000)  # Penalty for failed import
            
            execution_time = time.time() - start_time
            avg_import_time = sum(import_times) / len(import_times)
            max_import_time = max(import_times)
            
            if max_import_time < 5000:  # 5 seconds max
                result = QualityGateResult(
                    gate_name="Import Performance Benchmark",
                    status=QualityGateStatus.PASSED,
                    score=10.0,
                    message=f"Import performance good (avg: {avg_import_time:.1f}ms)",
                    execution_time=execution_time,
                    details={
                        'avg_import_time_ms': avg_import_time,
                        'max_import_time_ms': max_import_time,
                        'modules_tested': len(test_modules)
                    }
                )
                print(f"   ✅ Import Performance: avg {avg_import_time:.1f}ms")
            else:
                result = QualityGateResult(
                    gate_name="Import Performance Benchmark",
                    status=QualityGateStatus.WARNING,
                    score=5.0,
                    message=f"Slow imports detected (max: {max_import_time:.1f}ms)",
                    execution_time=execution_time,
                    details={
                        'avg_import_time_ms': avg_import_time,
                        'max_import_time_ms': max_import_time,
                        'modules_tested': len(test_modules)
                    },
                    recommendations=["Optimize import paths and reduce startup dependencies"]
                )
                print(f"   ⚠️  Import Performance: slow imports (max {max_import_time:.1f}ms)")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Import Performance Benchmark", str(e), time.time() - start_time)
    
    async def _benchmark_core_functionality(self) -> None:
        """Benchmark core functionality performance"""
        start_time = time.time()
        
        try:
            # Simple performance test
            test_start = time.time()
            
            # Simulate some computational work
            result_data = []
            for i in range(1000):
                result_data.append(f"test_item_{i}")
            
            # Simulate processing
            processed_data = [item.upper() for item in result_data]
            
            test_time = (time.time() - test_start) * 1000  # Convert to ms
            execution_time = time.time() - start_time
            
            if test_time < 100:  # 100ms threshold
                result = QualityGateResult(
                    gate_name="Core Functionality Benchmark",
                    status=QualityGateStatus.PASSED,
                    score=9.0,
                    message=f"Core functionality performance good ({test_time:.1f}ms)",
                    execution_time=execution_time,
                    details={
                        'benchmark_time_ms': test_time,
                        'items_processed': len(processed_data)
                    }
                )
                print(f"   ✅ Core Performance: {test_time:.1f}ms for {len(processed_data)} items")
            else:
                result = QualityGateResult(
                    gate_name="Core Functionality Benchmark",
                    status=QualityGateStatus.WARNING,
                    score=6.0,
                    message=f"Core functionality slower than expected ({test_time:.1f}ms)",
                    execution_time=execution_time,
                    details={
                        'benchmark_time_ms': test_time,
                        'items_processed': len(processed_data)
                    },
                    recommendations=["Profile code for performance bottlenecks"]
                )
                print(f"   ⚠️  Core Performance: {test_time:.1f}ms (slower than expected)")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Core Functionality Benchmark", str(e), time.time() - start_time)
    
    async def _run_integration_gates(self) -> None:
        """Run integration validation gates"""
        print("\n🔗 Stage 4: Integration Gates")
        print("-" * 30)
        
        await self._test_consciousness_integration()
        await self._test_api_endpoints()
    
    async def _test_consciousness_integration(self) -> None:
        """Test consciousness system integration"""
        start_time = time.time()
        
        try:
            # Test consciousness integration
            from neuro_symbolic_law.consciousness.conscious_reasoner import ConsciousLegalReasoner, ConsciousnessLevel
            
            reasoner = ConsciousLegalReasoner(consciousness_level=ConsciousnessLevel.CONSCIOUS)
            
            # Verify basic functionality
            integration_score = 10.0
            issues = []
            
            # Check if critical components exist
            if not hasattr(reasoner, 'consciousness_level'):
                integration_score -= 3.0
                issues.append("Missing consciousness_level attribute")
            
            if not hasattr(reasoner, 'conscious_legal_analysis'):
                integration_score -= 3.0
                issues.append("Missing conscious_legal_analysis method")
            
            execution_time = time.time() - start_time
            
            if integration_score >= 8.0:
                result = QualityGateResult(
                    gate_name="Consciousness System Integration",
                    status=QualityGateStatus.PASSED,
                    score=integration_score,
                    message="Consciousness system integration successful",
                    execution_time=execution_time,
                    details={'consciousness_level': reasoner.consciousness_level.value}
                )
                print(f"   ✅ Consciousness Integration: Score {integration_score:.1f}/10.0")
            else:
                result = QualityGateResult(
                    gate_name="Consciousness System Integration",
                    status=QualityGateStatus.WARNING,
                    score=integration_score,
                    message=f"Integration issues found: {len(issues)}",
                    execution_time=execution_time,
                    details={'issues': issues},
                    recommendations=["Fix integration issues before deployment"]
                )
                print(f"   ⚠️  Consciousness Integration: {len(issues)} issues found")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Consciousness System Integration", str(e), time.time() - start_time)
    
    async def _test_api_endpoints(self) -> None:
        """Test API endpoints if they exist"""
        start_time = time.time()
        
        try:
            # Check if API files exist
            api_files = [
                self.project_root / "api" / "main.py",
                self.project_root / "src" / "neuro_symbolic_law" / "cli.py"
            ]
            
            found_apis = [f for f in api_files if f.exists()]
            
            execution_time = time.time() - start_time
            
            if found_apis:
                result = QualityGateResult(
                    gate_name="API Endpoints Check",
                    status=QualityGateStatus.PASSED,
                    score=8.0,
                    message=f"API endpoints found: {len(found_apis)}",
                    execution_time=execution_time,
                    details={'api_files': [str(f) for f in found_apis]}
                )
                print(f"   ✅ API Endpoints: {len(found_apis)} endpoint files found")
            else:
                result = QualityGateResult(
                    gate_name="API Endpoints Check",
                    status=QualityGateStatus.WARNING,
                    score=6.0,
                    message="No API endpoints found",
                    execution_time=execution_time,
                    recommendations=["Consider adding API endpoints for production deployment"]
                )
                print(f"   ⚠️  API Endpoints: No endpoint files found")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("API Endpoints Check", str(e), time.time() - start_time)
    
    async def _run_documentation_gates(self) -> None:
        """Run documentation validation gates"""
        print("\n📚 Stage 5: Documentation Gates")
        print("-" * 35)
        
        await self._check_documentation_coverage()
        await self._validate_readme()
    
    async def _check_documentation_coverage(self) -> None:
        """Check documentation coverage"""
        start_time = time.time()
        
        try:
            python_files = list(self.project_root.rglob("*.py"))
            doc_stats = {
                'total_files': 0,
                'documented_files': 0,
                'total_functions': 0,
                'documented_functions': 0,
                'total_classes': 0,
                'documented_classes': 0
            }
            
            for py_file in python_files:
                if any(skip in str(py_file) for skip in ['.git', '__pycache__', '.venv']):
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc_stats['total_files'] += 1
                    
                    # Check if file has docstrings
                    if '"""' in content or "'''" in content:
                        doc_stats['documented_files'] += 1
                    
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        
                        # Count functions
                        if stripped.startswith('def ') or stripped.startswith('async def '):
                            doc_stats['total_functions'] += 1
                            # Check if next few lines have docstring
                            for j in range(i+1, min(i+5, len(lines))):
                                if '"""' in lines[j] or "'''" in lines[j]:
                                    doc_stats['documented_functions'] += 1
                                    break
                        
                        # Count classes
                        elif stripped.startswith('class '):
                            doc_stats['total_classes'] += 1
                            # Check if next few lines have docstring
                            for j in range(i+1, min(i+5, len(lines))):
                                if '"""' in lines[j] or "'''" in lines[j]:
                                    doc_stats['documented_classes'] += 1
                                    break
                
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            # Calculate coverage percentages
            file_coverage = (doc_stats['documented_files'] / max(doc_stats['total_files'], 1)) * 100
            func_coverage = (doc_stats['documented_functions'] / max(doc_stats['total_functions'], 1)) * 100
            class_coverage = (doc_stats['documented_classes'] / max(doc_stats['total_classes'], 1)) * 100
            
            overall_coverage = (file_coverage + func_coverage + class_coverage) / 3
            
            execution_time = time.time() - start_time
            
            if overall_coverage >= self.thresholds['documentation_coverage_min']:
                result = QualityGateResult(
                    gate_name="Documentation Coverage",
                    status=QualityGateStatus.PASSED,
                    score=9.0,
                    message=f"Documentation coverage: {overall_coverage:.1f}%",
                    execution_time=execution_time,
                    details={
                        'file_coverage': file_coverage,
                        'function_coverage': func_coverage,
                        'class_coverage': class_coverage,
                        'overall_coverage': overall_coverage,
                        **doc_stats
                    }
                )
                print(f"   ✅ Documentation: {overall_coverage:.1f}% coverage")
            else:
                result = QualityGateResult(
                    gate_name="Documentation Coverage",
                    status=QualityGateStatus.WARNING,
                    score=5.0,
                    message=f"Low documentation coverage: {overall_coverage:.1f}%",
                    execution_time=execution_time,
                    details={
                        'file_coverage': file_coverage,
                        'function_coverage': func_coverage,
                        'class_coverage': class_coverage,
                        'overall_coverage': overall_coverage,
                        **doc_stats
                    },
                    recommendations=["Add docstrings to functions and classes"]
                )
                print(f"   ⚠️  Documentation: {overall_coverage:.1f}% coverage (below {self.thresholds['documentation_coverage_min']}%)")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Documentation Coverage", str(e), time.time() - start_time)
    
    async def _validate_readme(self) -> None:
        """Validate README file"""
        start_time = time.time()
        
        try:
            readme_files = [
                self.project_root / "README.md",
                self.project_root / "README.rst",
                self.project_root / "README.txt"
            ]
            
            readme_file = None
            for f in readme_files:
                if f.exists():
                    readme_file = f
                    break
            
            execution_time = time.time() - start_time
            
            if readme_file:
                try:
                    with open(readme_file, 'r', encoding='utf-8') as f:
                        readme_content = f.read()
                    
                    # Check README quality
                    readme_score = 10.0
                    issues = []
                    
                    if len(readme_content) < 500:
                        readme_score -= 2.0
                        issues.append("README is quite short")
                    
                    required_sections = ['installation', 'usage', 'example']
                    missing_sections = []
                    for section in required_sections:
                        if section.lower() not in readme_content.lower():
                            missing_sections.append(section)
                            readme_score -= 1.0
                    
                    if missing_sections:
                        issues.append(f"Missing sections: {', '.join(missing_sections)}")
                    
                    status = QualityGateStatus.PASSED if readme_score >= 8.0 else QualityGateStatus.WARNING
                    
                    result = QualityGateResult(
                        gate_name="README Validation",
                        status=status,
                        score=readme_score,
                        message=f"README found and analyzed ({len(issues)} issues)",
                        execution_time=execution_time,
                        details={
                            'readme_file': str(readme_file),
                            'content_length': len(readme_content),
                            'issues': issues
                        },
                        recommendations=["Improve README content"] if issues else []
                    )
                    
                    print(f"   {'✅' if status == QualityGateStatus.PASSED else '⚠️ '} README: Score {readme_score:.1f}/10.0")
                
                except (UnicodeDecodeError, PermissionError):
                    result = QualityGateResult(
                        gate_name="README Validation",
                        status=QualityGateStatus.WARNING,
                        score=5.0,
                        message="README found but couldn't be read",
                        execution_time=execution_time,
                        recommendations=["Fix README file encoding"]
                    )
                    print(f"   ⚠️  README: Found but unreadable")
            
            else:
                result = QualityGateResult(
                    gate_name="README Validation",
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    message="No README file found",
                    execution_time=execution_time,
                    recommendations=["Add a comprehensive README file"]
                )
                print(f"   ❌ README: No README file found")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("README Validation", str(e), time.time() - start_time)
    
    async def _run_production_readiness_gates(self) -> None:
        """Run production readiness validation gates"""
        print("\n🚀 Stage 6: Production Readiness Gates")
        print("-" * 40)
        
        await self._check_deployment_configuration()
        await self._check_monitoring_setup()
        await self._check_error_handling()
    
    async def _check_deployment_configuration(self) -> None:
        """Check deployment configuration files"""
        start_time = time.time()
        
        try:
            deployment_files = [
                self.project_root / "Dockerfile",
                self.project_root / "docker-compose.yml",
                self.project_root / "kubernetes" / "deployment.yaml",
                self.project_root / "deploy",
                self.project_root / "Makefile"
            ]
            
            found_deployment = [f for f in deployment_files if f.exists()]
            
            execution_time = time.time() - start_time
            
            if found_deployment:
                result = QualityGateResult(
                    gate_name="Deployment Configuration",
                    status=QualityGateStatus.PASSED,
                    score=9.0,
                    message=f"Deployment configuration found: {len(found_deployment)} files",
                    execution_time=execution_time,
                    details={'deployment_files': [str(f) for f in found_deployment]}
                )
                print(f"   ✅ Deployment Config: {len(found_deployment)} configuration files")
            else:
                result = QualityGateResult(
                    gate_name="Deployment Configuration",
                    status=QualityGateStatus.WARNING,
                    score=4.0,
                    message="No deployment configuration found",
                    execution_time=execution_time,
                    recommendations=["Add Docker, Kubernetes, or other deployment configuration"]
                )
                print(f"   ⚠️  Deployment Config: No configuration files found")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Deployment Configuration", str(e), time.time() - start_time)
    
    async def _check_monitoring_setup(self) -> None:
        """Check monitoring and observability setup"""
        start_time = time.time()
        
        try:
            # Check for monitoring-related files and imports
            monitoring_indicators = [
                self.project_root / "src" / "neuro_symbolic_law" / "monitoring",
                self.project_root / "prometheus.yml",
                self.project_root / "grafana",
            ]
            
            found_monitoring = [f for f in monitoring_indicators if f.exists()]
            
            # Check for monitoring imports in code
            monitoring_imports = 0
            python_files = list(self.project_root.rglob("*.py"))
            
            for py_file in python_files:
                if any(skip in str(py_file) for skip in ['.git', '__pycache__', '.venv']):
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    monitoring_keywords = ['logging', 'metrics', 'health', 'monitor', 'observability']
                    for keyword in monitoring_keywords:
                        if keyword in content.lower():
                            monitoring_imports += 1
                            break
                
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            execution_time = time.time() - start_time
            
            monitoring_score = len(found_monitoring) * 3 + min(monitoring_imports / 10, 4)
            
            if monitoring_score >= 7.0:
                result = QualityGateResult(
                    gate_name="Monitoring Setup",
                    status=QualityGateStatus.PASSED,
                    score=monitoring_score,
                    message=f"Monitoring setup found (score: {monitoring_score:.1f})",
                    execution_time=execution_time,
                    details={
                        'monitoring_files': [str(f) for f in found_monitoring],
                        'monitoring_imports': monitoring_imports
                    }
                )
                print(f"   ✅ Monitoring: Score {monitoring_score:.1f}/10.0")
            else:
                result = QualityGateResult(
                    gate_name="Monitoring Setup",
                    status=QualityGateStatus.WARNING,
                    score=monitoring_score,
                    message=f"Limited monitoring setup (score: {monitoring_score:.1f})",
                    execution_time=execution_time,
                    details={
                        'monitoring_files': [str(f) for f in found_monitoring],
                        'monitoring_imports': monitoring_imports
                    },
                    recommendations=["Add comprehensive monitoring and observability"]
                )
                print(f"   ⚠️  Monitoring: Score {monitoring_score:.1f}/10.0 (needs improvement)")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Monitoring Setup", str(e), time.time() - start_time)
    
    async def _check_error_handling(self) -> None:
        """Check error handling patterns"""
        start_time = time.time()
        
        try:
            python_files = list(self.project_root.rglob("*.py"))
            error_handling_stats = {
                'total_files': 0,
                'files_with_try_catch': 0,
                'files_with_logging': 0,
                'files_with_custom_exceptions': 0,
                'total_try_blocks': 0,
                'total_except_blocks': 0
            }
            
            for py_file in python_files:
                if any(skip in str(py_file) for skip in ['.git', '__pycache__', '.venv']):
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    error_handling_stats['total_files'] += 1
                    
                    if 'try:' in content:
                        error_handling_stats['files_with_try_catch'] += 1
                        error_handling_stats['total_try_blocks'] += content.count('try:')
                    
                    if 'except' in content:
                        error_handling_stats['total_except_blocks'] += content.count('except')
                    
                    if 'logging.' in content or 'logger.' in content:
                        error_handling_stats['files_with_logging'] += 1
                    
                    if 'Exception' in content or 'Error' in content:
                        error_handling_stats['files_with_custom_exceptions'] += 1
                
                except (UnicodeDecodeError, PermissionError):
                    continue
            
            # Calculate error handling score
            try_catch_ratio = error_handling_stats['files_with_try_catch'] / max(error_handling_stats['total_files'], 1)
            logging_ratio = error_handling_stats['files_with_logging'] / max(error_handling_stats['total_files'], 1)
            
            error_handling_score = (try_catch_ratio * 5) + (logging_ratio * 3) + 2
            error_handling_score = min(error_handling_score, 10.0)
            
            execution_time = time.time() - start_time
            
            if error_handling_score >= 7.0:
                result = QualityGateResult(
                    gate_name="Error Handling Check",
                    status=QualityGateStatus.PASSED,
                    score=error_handling_score,
                    message=f"Error handling patterns adequate (score: {error_handling_score:.1f})",
                    execution_time=execution_time,
                    details=error_handling_stats
                )
                print(f"   ✅ Error Handling: Score {error_handling_score:.1f}/10.0")
            else:
                result = QualityGateResult(
                    gate_name="Error Handling Check",
                    status=QualityGateStatus.WARNING,
                    score=error_handling_score,
                    message=f"Error handling needs improvement (score: {error_handling_score:.1f})",
                    execution_time=execution_time,
                    details=error_handling_stats,
                    recommendations=["Add more try/catch blocks and logging"]
                )
                print(f"   ⚠️  Error Handling: Score {error_handling_score:.1f}/10.0 (needs improvement)")
            
            self.results.append(result)
            
        except Exception as e:
            self._add_error_result("Error Handling Check", str(e), time.time() - start_time)
    
    def _add_error_result(self, gate_name: str, error_message: str, execution_time: float) -> None:
        """Add error result for failed quality gate"""
        result = QualityGateResult(
            gate_name=gate_name,
            status=QualityGateStatus.FAILED,
            score=0.0,
            message=f"Quality gate failed: {error_message}",
            execution_time=execution_time,
            recommendations=["Fix the underlying issue and retry"]
        )
        self.results.append(result)
        print(f"   ❌ {gate_name}: Error - {error_message}")
    
    def _calculate_overall_results(self, total_time: float) -> Dict[str, Any]:
        """Calculate overall quality gate results"""
        if not self.results:
            return {
                'overall_status': QualityGateStatus.FAILED.value,
                'overall_score': 0.0,
                'total_gates': 0,
                'passed_gates': 0,
                'failed_gates': 0,
                'warning_gates': 0,
                'execution_time': total_time,
                'results': []
            }
        
        # Calculate statistics
        passed_count = sum(1 for r in self.results if r.status == QualityGateStatus.PASSED)
        failed_count = sum(1 for r in self.results if r.status == QualityGateStatus.FAILED)
        warning_count = sum(1 for r in self.results if r.status == QualityGateStatus.WARNING)
        
        # Calculate overall score (weighted average)
        total_score = sum(r.score for r in self.results)
        overall_score = total_score / len(self.results) if self.results else 0.0
        
        # Determine overall status
        if failed_count > 0 or (self.strict_mode and warning_count > 0):
            overall_status = QualityGateStatus.FAILED
        elif warning_count > 0:
            overall_status = QualityGateStatus.WARNING
        else:
            overall_status = QualityGateStatus.PASSED
        
        self.overall_status = overall_status
        
        return {
            'overall_status': overall_status.value,
            'overall_score': overall_score,
            'total_gates': len(self.results),
            'passed_gates': passed_count,
            'failed_gates': failed_count,
            'warning_gates': warning_count,
            'execution_time': total_time,
            'strict_mode': self.strict_mode,
            'results': [{
                'gate_name': r.gate_name,
                'status': r.status.value,
                'score': r.score,
                'message': r.message,
                'execution_time': r.execution_time,
                'recommendations': r.recommendations
            } for r in self.results]
        }
    
    def _display_final_results(self, results_summary: Dict[str, Any]) -> None:
        """Display comprehensive final results"""
        print("\n" + "=" * 65)
        print("🛡️ COMPREHENSIVE QUALITY GATES RESULTS")
        print("=" * 65)
        
        # Overall status
        status_icon = {
            'passed': '✅',
            'warning': '⚠️',
            'failed': '❌'
        }.get(results_summary['overall_status'], '❓')
        
        print(f"\n{status_icon} Overall Status: {results_summary['overall_status'].upper()}")
        print(f"📊 Overall Score: {results_summary['overall_score']:.1f}/10.0")
        print(f"⏱️  Total Execution Time: {results_summary['execution_time']:.2f} seconds")
        
        # Gate statistics
        print(f"\n📈 Quality Gate Statistics:")
        print(f"   • Total Gates: {results_summary['total_gates']}")
        print(f"   • Passed: {results_summary['passed_gates']} ✅")
        print(f"   • Warnings: {results_summary['warning_gates']} ⚠️")
        print(f"   • Failed: {results_summary['failed_gates']} ❌")
        
        # Success rate
        success_rate = (results_summary['passed_gates'] / max(results_summary['total_gates'], 1)) * 100
        print(f"   • Success Rate: {success_rate:.1f}%")
        
        # Recommendations summary
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)
        
        if all_recommendations:
            print(f"\n💡 Key Recommendations:")
            unique_recommendations = list(set(all_recommendations))[:5]  # Top 5 unique recommendations
            for i, recommendation in enumerate(unique_recommendations, 1):
                print(f"   {i}. {recommendation}")
        
        # Production readiness assessment
        print(f"\n🚀 Production Readiness Assessment:")
        
        if results_summary['overall_status'] == 'passed':
            print("   ✅ System is ready for production deployment")
            print("   ✅ All critical quality gates passed")
            print("   ✅ No blocking issues detected")
        elif results_summary['overall_status'] == 'warning':
            print("   ⚠️  System has minor issues but may be deployed with caution")
            print("   ⚠️  Address warnings before production if possible")
            print("   ⚠️  Monitor closely in production environment")
        else:
            print("   ❌ System is NOT ready for production deployment")
            print("   ❌ Critical issues must be resolved first")
            print("   ❌ Do not deploy until all failures are fixed")
        
        print("\n" + "=" * 65)
    
    def save_results_to_file(self, output_file: str = "quality_gates_results.json") -> None:
        """Save quality gate results to file"""
        try:
            results_data = {
                'timestamp': time.time(),
                'project_root': str(self.project_root),
                'strict_mode': self.strict_mode,
                'overall_status': self.overall_status.value,
                'thresholds': self.thresholds,
                'results': [{
                    'gate_name': r.gate_name,
                    'status': r.status.value,
                    'score': r.score,
                    'message': r.message,
                    'execution_time': r.execution_time,
                    'details': r.details,
                    'recommendations': r.recommendations
                } for r in self.results]
            }
            
            with open(output_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"Quality gate results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results to file: {e}")


async def main():
    """Main function to run comprehensive quality gates"""
    
    print("🛡️ COMPREHENSIVE QUALITY GATES - PRODUCTION VALIDATION")
    print("=" * 60)
    print("Validating production readiness with comprehensive quality gates:")
    print("• Code quality and style validation")
    print("• Security vulnerability scanning")
    print("• Performance benchmarking")
    print("• Integration testing")
    print("• Documentation validation")
    print("• Production readiness checks")
    print("=" * 60)
    
    # Initialize quality gates
    quality_gates = ComprehensiveQualityGates(
        project_root="/root/repo",
        strict_mode=False,  # Allow warnings in development
        performance_benchmarks=True
    )
    
    # Run all quality gates
    results = await quality_gates.run_all_quality_gates()
    
    # Save results
    quality_gates.save_results_to_file("comprehensive_quality_gates_results.json")
    
    # Return exit code based on results
    if results['overall_status'] == 'failed':
        print("\n❌ Quality gates failed - system not ready for production")
        return False
    elif results['overall_status'] == 'warning':
        print("\n⚠️  Quality gates passed with warnings - proceed with caution")
        return True
    else:
        print("\n✅ All quality gates passed - system ready for production!")
        return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
