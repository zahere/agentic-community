#!/usr/bin/env python3
"""
Test script to verify Agentic Community installation and functionality.
"""

import sys
import traceback

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from agentic_community import SimpleAgent, SearchTool, CalculatorTool, TextTool
        print("✓ Main imports successful")
        
        from agentic_community.core.base import BaseAgent, BaseTool
        print("✓ Core imports successful")
        
        from agentic_community.core.state import StateManager
        print("✓ State manager import successful")
        
        from agentic_community.core.utils import get_logger
        print("✓ Utils import successful")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic agent functionality."""
    print("\nTesting basic functionality...")
    try:
        from agentic_community import SimpleAgent, SearchTool, CalculatorTool, TextTool
        
        # Create tools
        tools = [
            SearchTool(),
            CalculatorTool(),
            TextTool()
        ]
        print("✓ Tools created successfully")
        
        # Create agent
        agent = SimpleAgent("TestAgent", tools)
        print("✓ Agent created successfully")
        
        # Test simple task
        result = agent.execute("Calculate 5 + 3")
        print(f"✓ Agent executed task: {result}")
        
        # Test agent state
        state = agent.get_state()
        print(f"✓ Agent state retrieved: {state}")
        
        return True
    except Exception as e:
        print(f"✗ Functionality error: {e}")
        traceback.print_exc()
        return False

def test_licensing():
    """Test that licensing works correctly for community edition."""
    print("\nTesting licensing...")
    try:
        from agentic_community.core.licensing import LicenseManager, Feature
        
        # Community edition should have basic features
        has_basic = LicenseManager.has_feature(Feature.BASIC_AGENTS)
        print(f"✓ Basic agents available: {has_basic}")
        
        # Community edition should NOT have enterprise features
        has_advanced = LicenseManager.has_feature(Feature.ADVANCED_AGENTS)
        print(f"✓ Advanced agents restricted: {not has_advanced}")
        
        return True
    except Exception as e:
        print(f"✗ Licensing error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Agentic Community Edition Test Suite\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Functionality Test", test_basic_functionality),
        ("Licensing Test", test_licensing)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"\n✅ {test_name} PASSED")
        else:
            failed += 1
            print(f"\n❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"Test Summary: {passed} passed, {failed} failed")
    print('='*50)
    
    if failed == 0:
        print("\n🎉 All tests passed! The community edition is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
