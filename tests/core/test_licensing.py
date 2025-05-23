"""
Tests for licensing functionality
"""

import pytest
import os
from datetime import datetime, timedelta
from pathlib import Path

from agentic_community.core.licensing import LicenseManager, License, get_license_manager


class TestLicense:
    """Test License model."""
    
    def test_license_creation(self):
        """Test creating license."""
        license = License(
            key="TEST-KEY-123",
            edition="community",
            seats=1
        )
        
        assert license.key == "TEST-KEY-123"
        assert license.edition == "community"
        assert license.seats == 1
        assert license.company is None
        assert license.expires_at is None
        assert license.features == {}
        assert license.metadata == {}
        
    def test_enterprise_license(self):
        """Test enterprise license."""
        expires = datetime.now() + timedelta(days=365)
        license = License(
            key="ENT-KEY-456",
            edition="enterprise",
            company="Test Corp",
            seats=10,
            expires_at=expires,
            features={"advanced_reasoning": True}
        )
        
        assert license.edition == "enterprise"
        assert license.company == "Test Corp"
        assert license.seats == 10
        assert license.expires_at == expires
        assert license.features["advanced_reasoning"] is True


class TestLicenseManager:
    """Test LicenseManager functionality."""
    
    def test_manager_initialization(self):
        """Test license manager initialization."""
        # Clear any existing license
        os.environ.pop("AGENTIC_LICENSE_KEY", None)
        
        manager = LicenseManager()
        
        assert manager.license is None
        assert manager.get_edition() == "community"
        
    def test_community_features(self):
        """Test community feature checking."""
        manager = LicenseManager()
        
        # Community features should always be available
        assert manager.check_feature("basic_reasoning") is True
        assert manager.check_feature("simple_tools") is True
        assert manager.check_feature("single_agent") is True
        
        # Enterprise features should not be available
        assert manager.check_feature("advanced_reasoning") is False
        assert manager.check_feature("multi_agent") is False
        
    def test_license_validation(self):
        """Test license validation."""
        manager = LicenseManager()
        
        # Valid enterprise license format
        valid = manager.validate_license("ENT-12345678901234567890123456789012")
        assert valid is True
        assert manager.get_edition() == "enterprise"
        
        # Invalid license
        manager.license = None
        invalid = manager.validate_license("INVALID-KEY")
        assert invalid is False
        assert manager.get_edition() == "community"
        
    def test_enterprise_features(self):
        """Test enterprise feature access."""
        manager = LicenseManager()
        
        # Set enterprise license
        manager.validate_license("ENT-12345678901234567890123456789012")
        
        # All features should be available
        assert manager.check_feature("advanced_reasoning") is True
        assert manager.check_feature("multi_agent") is True
        assert manager.check_feature("game_theory") is True
        
        # Community features still available
        assert manager.check_feature("basic_reasoning") is True
        
    def test_expired_license(self):
        """Test expired license handling."""
        manager = LicenseManager()
        
        # Create expired license
        manager.license = License(
            key="ENT-EXPIRED",
            edition="enterprise",
            expires_at=datetime.now() - timedelta(days=1),
            features={feature: True for feature in manager.ENTERPRISE_FEATURES}
        )
        
        # Enterprise features should not be available
        assert manager.check_feature("advanced_reasoning") is False
        assert manager.check_feature("multi_agent") is False
        
    def test_get_limits(self):
        """Test getting edition limits."""
        manager = LicenseManager()
        
        # Community limits
        limits = manager.get_limits()
        assert limits["max_agents"] == 1
        assert limits["max_tools"] == 3
        assert limits["max_iterations"] == 10
        assert limits["llm_providers"] == ["openai"]
        assert limits["support_level"] == "community"
        
        # Enterprise limits
        manager.validate_license("ENT-12345678901234567890123456789012")
        limits = manager.get_limits()
        assert limits["max_agents"] is None
        assert limits["max_tools"] is None
        assert limits["max_iterations"] == 50
        assert len(limits["llm_providers"]) > 1
        assert limits["support_level"] == "priority"
        
    def test_license_generation(self):
        """Test license key generation."""
        manager = LicenseManager()
        
        # Enterprise key
        key = manager.generate_license_key("enterprise", "Test Corp", 5)
        assert key.startswith("ENT-")
        assert len(key) == 36
        
        # Community key
        key = manager.generate_license_key("community", "Anyone")
        assert key == "COMMUNITY-FREE"
        
    def test_license_persistence(self, tmp_path):
        """Test saving and loading license."""
        manager = LicenseManager()
        
        # Create license
        manager.license = License(
            key="ENT-TEST",
            edition="enterprise",
            company="Test Corp"
        )
        
        # Save license
        license_file = tmp_path / "test_license.json"
        manager.save_license(license_file)
        
        assert license_file.exists()
        
        # Load in new manager
        new_manager = LicenseManager()
        new_manager.license = None
        
        # Manually load from file
        import json
        with open(license_file) as f:
            data = json.load(f)
            new_manager.license = License(**data)
            
        assert new_manager.license.key == "ENT-TEST"
        assert new_manager.license.company == "Test Corp"
        
    def test_get_license_manager(self):
        """Test global license manager singleton."""
        manager1 = get_license_manager()
        manager2 = get_license_manager()
        
        # Should be same instance
        assert manager1 is manager2
