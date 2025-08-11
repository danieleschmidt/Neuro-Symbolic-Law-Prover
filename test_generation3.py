#!/usr/bin/env python3
"""
Test script for Generation 3 scalable features.
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from neuro_symbolic_law.core.scalable_prover import ScalableLegalProver
from neuro_symbolic_law.parsing.contract_parser import ContractParser
from neuro_symbolic_law.regulations import GDPR, AIAct


async def test_scalable_features():
    """Test Generation 3 scalable features."""
    
    print("🚀 Testing Generation 3: Scalable Legal Prover")
    print("=" * 60)
    
    # Initialize scalable prover with Generation 3 features
    print("\n1. 🏗️ Initializing Scalable Prover...")
    scalable_prover = ScalableLegalProver(
        initial_cache_size=500,
        max_cache_size=5000,
        max_workers=4,
        enable_adaptive_caching=True,
        enable_concurrent_processing=True,
        memory_threshold=0.7,
        debug=True
    )
    
    # Initialize parser
    parser = ContractParser(model='basic')
    
    # Sample contracts for testing
    contract_texts = [
        """
        DATA PROCESSING AGREEMENT - Contract A
        
        Between TechCorp Inc. (Controller) and CloudServices Ltd. (Processor).
        
        1. DATA CATEGORIES: Personal data including names, emails, usage analytics
        2. PROCESSING PURPOSES: Service provision, analytics, compliance monitoring
        3. RETENTION: Data retained for 12 months after service termination
        4. SECURITY: AES-256 encryption, access controls, regular audits
        5. DATA SUBJECT RIGHTS: Access, rectification, erasure within 30 days
        6. BREACH NOTIFICATION: Controller notified within 24 hours
        """,
        
        """
        AI SYSTEM SERVICE AGREEMENT - Contract B
        
        Between Innovation Corp (Client) and AI Solutions Inc. (Provider).
        
        1. AI SYSTEM: Machine learning recommendation engine for e-commerce
        2. DATA PROCESSING: Customer behavior, purchase history, preferences
        3. TRANSPARENCY: Users informed of AI-driven recommendations
        4. HUMAN OVERSIGHT: Manual review of high-impact decisions
        5. ACCURACY: Regular model validation and bias testing
        6. RETENTION: Model data retained for 24 months for improvement
        """,
        
        """
        PRIVACY POLICY - Contract C
        
        DataTech Corporation Privacy Policy
        
        1. DATA COLLECTION: We collect personal information when you use our services
        2. USE OF DATA: Analytics, personalization, marketing communications
        3. DATA SHARING: Third-party processors with adequate safeguards
        4. RETENTION PERIOD: 36 months or until account deletion
        5. YOUR RIGHTS: Access, correct, delete your personal information
        6. SECURITY MEASURES: Encryption, firewalls, access controls
        """
    ]
    
    # Parse contracts
    print("\n2. 📄 Parsing Test Contracts...")
    contracts = []
    for i, text in enumerate(contract_texts):
        contract = parser.parse(text, f"test_contract_{chr(65+i)}")
        contracts.append(contract)
        print(f"   ✓ Contract {chr(65+i)}: {len(contract.clauses)} clauses, {len(contract.parties)} parties")
    
    # Test single contract verification with adaptive caching
    print("\n3. ⚖️  Testing Single Contract Verification (with caching)...")
    start_time = time.time()
    
    gdpr = GDPR()
    result_a = scalable_prover.verify_compliance(
        contracts[0], 
        gdpr,
        focus_areas=['data_subject_rights', 'storage_limitation', 'security']
    )
    
    first_time = time.time() - start_time
    print(f"   ✓ First verification: {len(result_a)} requirements in {first_time:.3f}s")
    
    # Test cache hit
    start_time = time.time()
    result_a_cached = scalable_prover.verify_compliance(
        contracts[0], 
        gdpr,
        focus_areas=['data_subject_rights', 'storage_limitation', 'security']
    )
    
    cached_time = time.time() - start_time
    print(f"   ✓ Cached verification: {len(result_a_cached)} requirements in {cached_time:.3f}s")
    print(f"   📈 Cache speedup: {first_time/cached_time:.2f}x")
    
    # Test concurrent verification
    print("\n4. ⚡ Testing Concurrent Verification...")
    start_time = time.time()
    
    concurrent_results = await scalable_prover.verify_compliance_concurrent(
        contracts=contracts,
        regulation=gdpr,
        focus_areas=['purpose_limitation', 'data_minimization', 'security'],
        max_concurrent=3
    )
    
    concurrent_time = time.time() - start_time
    print(f"   ✓ Concurrent verification: {len(concurrent_results)} contracts in {concurrent_time:.3f}s")
    
    for contract_id, results in concurrent_results.items():
        compliant_count = sum(1 for r in results.values() if r.compliant)
        print(f"      • {contract_id}: {compliant_count}/{len(results)} compliant")
    
    # Test batch processing
    print("\n5. 📦 Testing Batch Processing...")
    
    # Prepare batch requests
    batch_requests = []
    regulations = [GDPR(), AIAct()]
    
    for i, contract in enumerate(contracts[:2]):  # Test with first 2 contracts
        for reg in regulations:
            batch_requests.append({
                'contract': contract,
                'regulation': reg,
                'focus_areas': ['transparency', 'security'] if isinstance(reg, AIAct) else ['data_retention', 'security']
            })
    
    start_time = time.time()
    batch_results = await scalable_prover.batch_verify_compliance(
        batch_requests=batch_requests,
        batch_size=3
    )
    
    batch_time = time.time() - start_time
    print(f"   ✓ Batch processing: {len(batch_results)} requests in {batch_time:.3f}s")
    
    successful_batches = sum(1 for r in batch_results if r.get('status') == 'success')
    print(f"   📊 Success rate: {successful_batches}/{len(batch_results)} ({successful_batches/len(batch_results)*100:.1f}%)")
    
    # Test performance metrics
    print("\n6. 📊 Performance Metrics...")
    metrics = scalable_prover.get_performance_metrics()
    
    print(f"   Cache Statistics:")
    if metrics.get('adaptive_cache'):
        cache_stats = metrics['adaptive_cache']
        print(f"     • Hit rate: {cache_stats['hit_rate']:.2f}")
        print(f"     • Utilization: {cache_stats['utilization']:.2%}")
        print(f"     • Current size: {cache_stats['size']}/{cache_stats['current_limit']}")
        print(f"     • Adaptation active: {cache_stats['adaptation_active']}")
    
    print(f"   Resource Manager:")
    resource_stats = metrics['resource_manager']
    print(f"     • CPU usage: {resource_stats['cpu_percent']:.1f}%")
    print(f"     • Memory usage: {resource_stats['memory_percent']:.1f}%")
    print(f"     • Current workers: {resource_stats['current_workers']}")
    print(f"     • Resource status: {resource_stats['resource_status']}")
    
    if metrics.get('performance_stats'):
        perf_stats = metrics['performance_stats']
        print(f"   Performance Statistics:")
        print(f"     • Avg verification time: {perf_stats['avg_verification_time']:.3f}s")
        print(f"     • Avg contract size: {perf_stats['avg_contract_size']:.1f} clauses")
        print(f"     • Requests per second: {perf_stats['requests_per_second']:.2f}")
    
    # Test system optimization
    print("\n7. ⚙️  Testing System Optimization...")
    optimization_results = scalable_prover.optimize_system()
    
    print("   Optimization Results:")
    if 'cache_optimization' in optimization_results:
        cache_opt = optimization_results['cache_optimization']
        print(f"     • Cache size changed: {cache_opt['size_changed']}")
    
    if 'garbage_collection' in optimization_results:
        gc_result = optimization_results['garbage_collection']
        print(f"     • Objects collected: {gc_result['objects_collected']}")
    
    print(f"     • Resource status: {optimization_results['resource_check']['resource_status']}")
    
    # Test with different AI Act contract
    print("\n8. 🤖 Testing AI Act Compliance...")
    ai_act = AIAct()
    ai_results = scalable_prover.verify_compliance(
        contracts[1],  # AI System contract
        ai_act
    )
    
    ai_compliant = sum(1 for r in ai_results.values() if r.compliant)
    print(f"   ✓ AI Act compliance: {ai_compliant}/{len(ai_results)} requirements")
    
    # Sample AI Act results
    for req_id, result in list(ai_results.items())[:3]:
        status_emoji = "✅" if result.compliant else "❌"
        print(f"     {status_emoji} {req_id}: {result.confidence:.1%} confidence")
    
    # Performance summary
    print("\n9. 📈 Performance Summary...")
    print(f"   Single verification (first): {first_time:.3f}s")
    print(f"   Single verification (cached): {cached_time:.3f}s")
    print(f"   Concurrent verification: {concurrent_time:.3f}s ({len(contracts)} contracts)")
    print(f"   Batch processing: {batch_time:.3f}s ({len(batch_requests)} requests)")
    print(f"   Cache speedup factor: {first_time/cached_time:.2f}x")
    
    # Cleanup
    print("\n10. 🧹 Cleanup...")
    scalable_prover.cleanup()
    print("    ✓ Resources cleaned up")
    
    print(f"\n✨ Generation 3 testing completed successfully!")
    print(f"   Scalable features demonstrated:")
    print(f"   • ✅ Adaptive caching with {first_time/cached_time:.2f}x speedup")
    print(f"   • ✅ Concurrent processing of {len(contracts)} contracts")
    print(f"   • ✅ Batch processing of {len(batch_requests)} requests")
    print(f"   • ✅ Resource management and optimization")
    print(f"   • ✅ Performance monitoring and metrics")


async def main():
    """Main test function."""
    try:
        await test_scalable_features()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)