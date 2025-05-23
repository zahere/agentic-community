#!/usr/bin/env python3
"""
Debug script to check langchain installation and imports
"""

import sys
import subprocess

print("🔍 Debugging LangChain Installation\n")

# Check Python info
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
print()

# Check installed packages
print("📦 Checking installed packages:")
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
langchain_packages = [line for line in result.stdout.split('\n') if 'langchain' in line.lower()]
if langchain_packages:
    print("Found LangChain packages:")
    for pkg in langchain_packages:
        print(f"  {pkg}")
else:
    print("❌ No LangChain packages found!")
print()

# Try different imports
print("🧪 Testing imports:")

# Test 1: langchain-openai
try:
    from langchain_openai import OpenAI
    print("✅ from langchain_openai import OpenAI - SUCCESS")
except ImportError as e:
    print(f"❌ from langchain_openai import OpenAI - FAILED: {e}")

# Test 2: langchain-community
try:
    from langchain_community.llms import OpenAI
    print("✅ from langchain_community.llms import OpenAI - SUCCESS")
except ImportError as e:
    print(f"❌ from langchain_community.llms import OpenAI - FAILED: {e}")

# Test 3: langchain direct
try:
    from langchain.llms import OpenAI
    print("✅ from langchain.llms import OpenAI - SUCCESS")
except ImportError as e:
    print(f"❌ from langchain.llms import OpenAI - FAILED: {e}")

# Test 4: langchain_core
try:
    import langchain_core
    print(f"✅ import langchain_core - SUCCESS (version: {langchain_core.__version__})")
except ImportError as e:
    print(f"❌ import langchain_core - FAILED: {e}")

# Test 5: langgraph
try:
    import langgraph
    print(f"✅ import langgraph - SUCCESS (version: {langgraph.__version__})")
except ImportError as e:
    print(f"❌ import langgraph - FAILED: {e}")

print("\n💡 Recommendation:")
print("If imports are failing, try:")
print("1. pip install --force-reinstall langchain-community langchain-openai")
print("2. Create a fresh virtual environment")
print("3. Check if you're using the correct Python/pip")
