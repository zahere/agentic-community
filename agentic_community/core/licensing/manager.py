"""
License Management System
Controls feature access based on edition
Copyright (c) 2025 Zaher Khateeb
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field

from agentic_community.core.utils import get_logger

logger = get_logger(__name__)


class License(BaseModel):
    """License model."""
    key: str = Field(description="License key")
    edition: str = Field(description="Edition: community or enterprise")
    company: Optional[str] = Field(default=None, description="Company name")
    seats: int = Field(default=1, description="Number of seats")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration date")
    features: Dict[str, bool] = Field(default_factory=dict, description="Feature flags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LicenseManager:
    """Manages licensing and feature access."""
    
    # Community features (always available)
    COMMUNITY_FEATURES = {
        "basic_reasoning",
        "simple_tools",
        "single_agent",
        "local_deployment",
        "basic_llm",  # OpenAI only
        "tool_limit_3",
        "basic_examples"
    }
    
    # Enterprise features (require license)
    ENTERPRISE_FEATURES = {
        "advanced_reasoning",
        "self_reflection",
        "multi_path_reasoning",
        "game_theory",
        "multi_agent",
        "unlimited_tools",
        "all_llm_providers",
        "enterprise_integrations",
        "cloud_deployment",
        "high_availability",
        "priority_support",
        "custom_development",
        "analytics",
        "monitoring",
        "sla"
    }
    
    def __init__(self):
        """Initialize license manager."""
        self.license: Optional[License] = None
        self._load_license()
        
    def _load_license(self) -> None:
        """Load license from environment or file."""
        # Check environment variable first
        license_key = os.getenv("AGENTIC_LICENSE_KEY")
        
        if license_key:
            self.validate_license(license_key)
        else:
            # Check for license file
            license_file = Path.home() / ".agentic" / "license.json"
            if license_file.exists():
                try:
                    with open(license_file, "r") as f:
                        data = json.load(f)
                        self.license = License(**data)
                        logger.info("License loaded from file")
                except Exception as e:
                    logger.error(f"Failed to load license file: {e}")
                    
    def validate_license(self, license_key: str) -> bool:
        """
        Validate a license key.
        
        Note: This is a simplified validation.
        Real implementation would verify against a license server.
        """
        try:
            # Simple validation for demo
            if license_key.startswith("ENT-") and len(license_key) == 36:
                # Extract metadata from key (simplified)
                self.license = License(
                    key=license_key,
                    edition="enterprise",
                    expires_at=datetime.now() + timedelta(days=365),
                    features={feature: True for feature in self.ENTERPRISE_FEATURES}
                )
                logger.info("Enterprise license validated")
                return True
            else:
                logger.warning("Invalid license key format")
                return False
                
        except Exception as e:
            logger.error(f"License validation failed: {e}")
            return False
            
    def get_edition(self) -> str:
        """Get current edition."""
        if self.license and self.license.edition == "enterprise":
            return "enterprise"
        return "community"
        
    def check_feature(self, feature: str) -> bool:
        """
        Check if a feature is available.
        
        Args:
            feature: Feature name to check
            
        Returns:
            True if feature is available
        """
        # Community features always available
        if feature in self.COMMUNITY_FEATURES:
            return True
            
        # Enterprise features require valid license
        if feature in self.ENTERPRISE_FEATURES:
            if self.license and self.license.edition == "enterprise":
                # Check expiration
                if self.license.expires_at and self.license.expires_at < datetime.now():
                    logger.warning(f"License expired, feature {feature} not available")
                    return False
                    
                # Check specific feature flag
                return self.license.features.get(feature, True)
                
        logger.debug(f"Feature {feature} not available in {self.get_edition()} edition")
        return False
    
    @staticmethod
    def check_feature_static(feature: str) -> bool:
        """Static method to check feature availability."""
        return get_license_manager().check_feature(feature)
        
    def get_limits(self) -> Dict[str, Any]:
        """Get current limits based on edition."""
        if self.get_edition() == "enterprise":
            return {
                "max_agents": None,  # Unlimited
                "max_tools": None,   # Unlimited
                "max_iterations": 50,
                "llm_providers": ["openai", "anthropic", "azure", "google", "local"],
                "support_level": "priority"
            }
        else:
            return {
                "max_agents": 1,
                "max_tools": 3,
                "max_iterations": 10,
                "llm_providers": ["openai"],
                "support_level": "community"
            }
            
    def generate_license_key(self, edition: str, company: str, seats: int = 1) -> str:
        """
        Generate a license key.
        
        Note: This is for demo purposes only.
        Real implementation would use cryptographic signing.
        """
        if edition == "enterprise":
            # Generate enterprise key
            data = f"{company}-{seats}-{datetime.now().isoformat()}"
            hash_val = hashlib.sha256(data.encode()).hexdigest()[:32]
            return f"ENT-{hash_val}"
        else:
            return "COMMUNITY-FREE"
            
    def save_license(self, license_path: Optional[Path] = None) -> None:
        """Save license to file."""
        if not self.license:
            logger.warning("No license to save")
            return
            
        if not license_path:
            license_path = Path.home() / ".agentic" / "license.json"
            
        license_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(license_path, "w") as f:
            json.dump(self.license.model_dump(mode="json"), f, indent=2)
            
        logger.info(f"License saved to {license_path}")


# Global instance
_license_manager = None


def get_license_manager() -> LicenseManager:
    """Get the global license manager instance."""
    global _license_manager
    if _license_manager is None:
        _license_manager = LicenseManager()
    return _license_manager


# For backward compatibility - add static method to class
LicenseManager.check_feature = LicenseManager.check_feature_static
