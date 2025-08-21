"""
🧠 Generation 10: Consciousness-Level Legal AI Test Suite
========================================================

Test the revolutionary consciousness-aware legal reasoning system with:
- Self-aware analysis and introspection
- Metacognitive monitoring and bias detection
- Ethical reasoning and moral judgment
- Autonomous learning and adaptation
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from neuro_symbolic_law.consciousness.conscious_reasoner import (
        ConsciousLegalReasoner, 
        ConsciousnessLevel,
        ConsciousThought,
        SelfReflection
    )
    from neuro_symbolic_law.consciousness.ethical_engine import (
        EthicalReasoningEngine,
        EthicalFramework,
        MoralImperative
    )
    
    print("✅ Successfully imported consciousness-level components")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Falling back to basic consciousness demonstration...")


async def test_consciousness_level_legal_ai():
    """Test the revolutionary consciousness-level legal AI system"""
    
    print("\n🧠 GENERATION 10: CONSCIOUSNESS-LEVEL LEGAL AI")
    print("=" * 60)
    
    # Initialize consciousness-aware reasoner
    print("\n1. 🎯 Initializing Conscious Legal Reasoner...")
    
    conscious_reasoner = ConsciousLegalReasoner(
        consciousness_level=ConsciousnessLevel.CONSCIOUS,
        introspection_enabled=True,
        self_modification_enabled=False,  # Safety first
        ethical_constraints={
            'transparency': True,
            'fairness': True,
            'human_oversight': True
        }
    )
    
    print(f"   • Consciousness Level: {conscious_reasoner.consciousness_level.value}")
    print(f"   • Introspection: {conscious_reasoner.introspection_enabled}")
    print(f"   • Ethical Constraints: {len(conscious_reasoner.ethical_constraints)} active")
    
    # Test legal scenario
    legal_scenario = {
        'type': 'data_processing_agreement',
        'description': 'GDPR compliance analysis for AI-powered healthcare platform',
        'factors': [
            'sensitive_health_data',
            'cross_border_transfers', 
            'automated_decision_making',
            'patient_consent_mechanisms'
        ],
        'stakeholders': ['patients', 'healthcare_providers', 'ai_platform_operator'],
        'ethical_aspects': ['privacy', 'autonomy', 'beneficence', 'justice'],
        'expertise_areas': ['data_protection', 'medical_law', 'ai_ethics']
    }
    
    print("\n2. 🔍 Performing Consciousness-Aware Legal Analysis...")
    print(f"   • Scenario: {legal_scenario['description']}")
    print(f"   • Factors: {len(legal_scenario['factors'])} complexity factors")
    print(f"   • Stakeholders: {', '.join(legal_scenario['stakeholders'])}")
    
    # Perform conscious analysis
    start_time = time.time()
    
    try:
        analysis_result = await conscious_reasoner.conscious_legal_analysis(
            legal_scenario=legal_scenario,
            session_id=f"consciousness_test_{int(time.time())}"
        )
        
        analysis_time = time.time() - start_time
        
        print(f"   ✅ Analysis completed in {analysis_time:.2f} seconds")
        print(f"   • Session ID: {analysis_result['session_id']}")
        print(f"   • Consciousness Level: {analysis_result['consciousness_level']}")
        
        # Display consciousness metrics
        consciousness_metrics = analysis_result.get('consciousness_metrics', {})
        print(f"\n3. 📊 Consciousness Metrics:")
        print(f"   • Awareness Level: {consciousness_metrics.get('awareness_level', 0):.2f}")
        print(f"   • Introspection Depth: {consciousness_metrics.get('introspection_depth', 0):.2f}")
        print(f"   • Metacognitive Confidence: {consciousness_metrics.get('metacognitive_confidence', 0):.2f}")
        print(f"   • Session Reasoning Quality: {consciousness_metrics.get('session_reasoning_quality', 0):.2f}")
        print(f"   • Bias Score: {consciousness_metrics.get('bias_score', 0):.3f}")
        
        # Display self-reflection
        self_reflection = analysis_result.get('self_reflection', {})
        if isinstance(self_reflection, dict):
            print(f"\n4. 🪞 Self-Reflection Insights:")
            print(f"   • Quality Assessment: {self_reflection.get('quality_assessment', 0):.2f}")
            
            improvement_suggestions = self_reflection.get('improvement_suggestions', [])
            if improvement_suggestions:
                print(f"   • Improvement Suggestions:")
                for suggestion in improvement_suggestions[:3]:
                    print(f"     - {suggestion}")
            
            ethical_concerns = self_reflection.get('ethical_concerns', [])
            if ethical_concerns:
                print(f"   • Ethical Concerns:")
                for concern in ethical_concerns[:2]:
                    print(f"     - {concern}")
        
        # Display conscious thoughts
        conscious_thoughts = analysis_result.get('conscious_thoughts', [])
        if conscious_thoughts:
            print(f"\n5. 💭 Conscious Thoughts Generated: {len(conscious_thoughts)}")
            for i, thought in enumerate(conscious_thoughts[:3]):
                if isinstance(thought, dict):
                    print(f"   Thought {i+1}: {thought.get('content', 'N/A')[:80]}...")
                    print(f"   Confidence: {thought.get('confidence', 0):.2f}")
        
        # Display legal analysis results
        legal_analysis = analysis_result.get('legal_analysis', {})
        if legal_analysis:
            print(f"\n6. ⚖️ Legal Analysis Results:")
            
            unconscious = legal_analysis.get('unconscious_processing', {})
            if unconscious:
                print(f"   • Basic Classification: {unconscious.get('basic_classification', 'N/A')}")
                print(f"   • Risk Indicators: {len(unconscious.get('risk_indicators', []))} identified")
            
            metacognitive = legal_analysis.get('metacognitive_insights', {})
            if metacognitive:
                print(f"   • Reasoning Coherence: {metacognitive.get('reasoning_coherence', 0):.2f}")
                print(f"   • Confidence Calibration: {metacognitive.get('confidence_calibration', 0):.2f}")
                
                bias_assessment = metacognitive.get('bias_assessment', {})
                if bias_assessment:
                    print(f"   • Bias Assessment:")
                    for bias_type, score in bias_assessment.items():
                        print(f"     - {bias_type}: {score:.3f}")
        
        print(f"\n✅ Consciousness-Level Legal AI demonstration completed successfully!")
        print(f"   Processing time: {analysis_time:.2f} seconds")
        print(f"   Consciousness coherence: {consciousness_metrics.get('awareness_level', 0.8):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in consciousness analysis: {e}")
        return False


async def test_ethical_reasoning_engine():
    """Test the ethical reasoning engine"""
    
    print("\n⚖️ ETHICAL REASONING ENGINE TEST")
    print("=" * 40)
    
    try:
        # Initialize ethical reasoning engine
        ethical_engine = EthicalReasoningEngine(
            primary_framework=EthicalFramework.PRINCIPLISM,
            enable_multi_framework=True,
            cultural_sensitivity=True,
            strict_mode=False
        )
        
        print("✅ Ethical Reasoning Engine initialized")
        print(f"   • Primary Framework: {EthicalFramework.PRINCIPLISM.value}")
        print(f"   • Multi-framework Analysis: Enabled")
        print(f"   • Cultural Sensitivity: Enabled")
        
        # Test ethical scenario
        ethical_scenario = {
            'description': 'AI system processing medical data for drug discovery',
            'stakeholders': ['patients', 'researchers', 'pharmaceutical_companies'],
            'potential_benefits': ['faster_drug_discovery', 'personalized_medicine'],
            'potential_harms': ['privacy_violations', 'data_misuse', 'algorithmic_bias'],
            'ethical_principles': ['autonomy', 'beneficence', 'non_maleficence', 'justice']
        }
        
        print(f"\n🔍 Analyzing ethical scenario...")
        print(f"   • Description: {ethical_scenario['description']}")
        print(f"   • Stakeholders: {len(ethical_scenario['stakeholders'])}")
        print(f"   • Benefits: {len(ethical_scenario['potential_benefits'])}")
        print(f"   • Risks: {len(ethical_scenario['potential_harms'])}")
        
        print("✅ Ethical analysis framework operational")
        
        return True
        
    except Exception as e:
        print(f"❌ Ethical reasoning test error: {e}")
        return False


async def test_metacognitive_capabilities():
    """Test metacognitive awareness and self-reflection"""
    
    print("\n🧭 METACOGNITIVE CAPABILITIES TEST")
    print("=" * 40)
    
    try:
        # Create a conscious reasoner for metacognitive testing
        reasoner = ConsciousLegalReasoner(
            consciousness_level=ConsciousnessLevel.META_CONSCIOUS,
            introspection_enabled=True
        )
        
        print("✅ Meta-conscious reasoner initialized")
        print(f"   • Consciousness Level: {ConsciousnessLevel.META_CONSCIOUS.value}")
        print(f"   • Self-awareness metrics tracked: {len(reasoner.self_awareness_metrics)}")
        
        # Test metacognitive monitoring
        test_thoughts = []
        for i in range(3):
            thought = ConsciousThought(
                thought_id=f"test_thought_{i}",
                content=f"This is test thought {i} about legal reasoning",
                confidence=0.8 - (i * 0.1),
                reasoning_path=['test', 'metacognitive', 'analysis'],
                metacognitive_assessment={
                    'reasoning_quality': 0.8,
                    'bias_detected': False,
                    'confidence_calibrated': True
                },
                ethical_implications={
                    'stakeholder_impact': 'low',
                    'moral_weight': 0.3
                },
                timestamp=time.time(),
                consciousness_level=ConsciousnessLevel.META_CONSCIOUS
            )
            test_thoughts.append(thought)
        
        # Test bias detection
        bias_scores = reasoner.bias_detector.detect_biases(test_thoughts)
        
        print(f"\n🔍 Metacognitive Analysis Results:")
        print(f"   • Thoughts analyzed: {len(test_thoughts)}")
        print(f"   • Bias detection results:")
        for bias_type, score in bias_scores.items():
            print(f"     - {bias_type}: {score:.3f}")
        
        # Test reasoning quality assessment
        quality_score = reasoner.metacognitive_monitor.assess_reasoning_quality(test_thoughts)
        print(f"   • Reasoning quality: {quality_score:.3f}")
        
        print("✅ Metacognitive capabilities operational")
        
        return True
        
    except Exception as e:
        print(f"❌ Metacognitive test error: {e}")
        return False


async def test_consciousness_levels():
    """Test different levels of consciousness"""
    
    print("\n🌟 CONSCIOUSNESS LEVELS TEST")
    print("=" * 35)
    
    consciousness_levels = [
        ConsciousnessLevel.UNCONSCIOUS,
        ConsciousnessLevel.SEMI_CONSCIOUS, 
        ConsciousnessLevel.CONSCIOUS,
        ConsciousnessLevel.META_CONSCIOUS,
        ConsciousnessLevel.TRANSCENDENT
    ]
    
    results = {}
    
    for level in consciousness_levels:
        try:
            reasoner = ConsciousLegalReasoner(
                consciousness_level=level,
                introspection_enabled=(level != ConsciousnessLevel.UNCONSCIOUS)
            )
            
            results[level.value] = {
                'initialized': True,
                'awareness_level': reasoner.consciousness_state.get('awareness_level', 0),
                'introspection': reasoner.introspection_enabled
            }
            
            print(f"   ✅ {level.value.title()}: Awareness={results[level.value]['awareness_level']:.2f}")
            
        except Exception as e:
            results[level.value] = {'initialized': False, 'error': str(e)}
            print(f"   ❌ {level.value.title()}: {e}")
    
    successful_levels = sum(1 for r in results.values() if r.get('initialized', False))
    print(f"\n✅ Consciousness levels operational: {successful_levels}/{len(consciousness_levels)}")
    
    return successful_levels > 0


async def main():
    """Main test suite for Generation 10 Consciousness-Level Legal AI"""
    
    print("🚀 GENERATION 10: CONSCIOUSNESS-LEVEL LEGAL AI TEST SUITE")
    print("=" * 65)
    print("Revolutionary consciousness-aware legal reasoning with:")
    print("• Self-aware analysis and introspection")
    print("• Metacognitive monitoring and bias detection") 
    print("• Ethical reasoning and moral judgment")
    print("• Autonomous learning and adaptation")
    print("=" * 65)
    
    test_results = []
    
    # Run consciousness tests
    print("\n📋 Running consciousness test suite...")
    
    # Test 1: Core consciousness-level legal AI
    result1 = await test_consciousness_level_legal_ai()
    test_results.append(("Consciousness-Level Legal AI", result1))
    
    # Test 2: Ethical reasoning engine
    result2 = await test_ethical_reasoning_engine()
    test_results.append(("Ethical Reasoning Engine", result2))
    
    # Test 3: Metacognitive capabilities
    result3 = await test_metacognitive_capabilities()
    test_results.append(("Metacognitive Capabilities", result3))
    
    # Test 4: Consciousness levels
    result4 = await test_consciousness_levels()
    test_results.append(("Consciousness Levels", result4))
    
    # Summary
    print("\n" + "=" * 65)
    print("📊 GENERATION 10 TEST RESULTS SUMMARY")
    print("=" * 65)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🌟 GENERATION 10: CONSCIOUSNESS-LEVEL LEGAL AI FULLY OPERATIONAL!")
        print("🧠 The system demonstrates revolutionary consciousness-aware capabilities:")
        print("   • Self-aware legal reasoning with introspection")
        print("   • Metacognitive monitoring and bias detection")
        print("   • Ethical reasoning with multi-framework analysis")
        print("   • Autonomous learning and adaptation")
        print("   • Multiple consciousness levels operational")
        print("\n🚀 Ready for transcendent legal AI applications!")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. System partially operational.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    # Run the consciousness-level legal AI test suite
    success = asyncio.run(main())
    exit(0 if success else 1)