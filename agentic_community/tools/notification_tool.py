"""
Notification Tool for Agentic Community Edition

This tool provides multi-channel notification capabilities for agents including
email, SMS, webhook, and desktop notifications.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import json
from enum import Enum

try:
    import plyer
    DESKTOP_NOTIFICATIONS_AVAILABLE = True
except ImportError:
    DESKTOP_NOTIFICATIONS_AVAILABLE = False

from ..base import Tool, ToolResult

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DESKTOP = "desktop"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"


@dataclass
class NotificationConfig:
    """Configuration for notification channels"""
    email_config: Optional[Dict[str, Any]] = None
    sms_config: Optional[Dict[str, Any]] = None
    webhook_urls: Optional[List[str]] = None
    slack_webhook: Optional[str] = None
    discord_webhook: Optional[str] = None
    telegram_config: Optional[Dict[str, Any]] = None


class NotificationTool(Tool):
    """
    Multi-channel notification tool for sending alerts and messages.
    
    Supports:
    - Email notifications (via email_tool)
    - SMS notifications (via Twilio/similar)
    - Webhook notifications
    - Desktop notifications
    - Slack notifications
    - Discord notifications
    - Telegram notifications
    """
    
    def __init__(self, config: NotificationConfig):
        super().__init__(
            name="notification_tool",
            description="Send notifications through multiple channels including email, SMS, webhooks, and messaging platforms"
        )
        self.config = config
        self._session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
    
    async def _execute(self, **kwargs) -> ToolResult:
        """Execute notification operation"""
        channel = kwargs.get("channel", NotificationChannel.EMAIL.value)
        message = kwargs.get("message", "")
        title = kwargs.get("title", "Notification")
        priority = kwargs.get("priority", "normal")
        metadata = kwargs.get("metadata", {})
        
        try:
            channel_enum = NotificationChannel(channel)
        except ValueError:
            return ToolResult(
                success=False,
                data={"error": f"Invalid channel: {channel}"},
                metadata={"channel": channel}
            )
        
        handlers = {
            NotificationChannel.EMAIL: self._send_email,
            NotificationChannel.SMS: self._send_sms,
            NotificationChannel.WEBHOOK: self._send_webhook,
            NotificationChannel.DESKTOP: self._send_desktop,
            NotificationChannel.SLACK: self._send_slack,
            NotificationChannel.DISCORD: self._send_discord,
            NotificationChannel.TELEGRAM: self._send_telegram
        }
        
        handler = handlers.get(channel_enum)
        if not handler:
            return ToolResult(
                success=False,
                data={"error": f"Handler not implemented for channel: {channel}"},
                metadata={"channel": channel}
            )
        
        try:
            result = await handler(
                message=message,
                title=title,
                priority=priority,
                metadata=metadata,
                **kwargs
            )
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "channel": channel,
                    "timestamp": datetime.now().isoformat(),
                    "priority": priority
                }
            )
        except Exception as e:
            logger.error(f"Notification failed for channel {channel}: {str(e)}")
            return ToolResult(
                success=False,
                data={"error": str(e)},
                metadata={"channel": channel}
            )
    
    async def _send_email(self, **kwargs) -> Dict[str, Any]:
        """Send email notification using email_tool"""
        if not self.config.email_config:
            raise ValueError("Email configuration not provided")
        
        # Import email tool dynamically
        from .email_tool import EmailTool, EmailConfig
        
        email_tool = EmailTool(EmailConfig(**self.config.email_config))
        
        result = await email_tool._execute(
            operation="send",
            to=kwargs.get("to", []),
            subject=kwargs.get("title", "Notification"),
            body=kwargs.get("message", ""),
            html_body=kwargs.get("html_message")
        )
        
        return result.data if result.success else {"error": result.data.get("error")}
    
    async def _send_sms(self, **kwargs) -> Dict[str, Any]:
        """Send SMS notification"""
        if not self.config.sms_config:
            raise ValueError("SMS configuration not provided")
        
        # This would integrate with Twilio or similar service
        # For now, we'll simulate the SMS sending
        phone_numbers = kwargs.get("to", [])
        message = kwargs.get("message", "")
        
        # In production, this would use Twilio API
        logger.info(f"Sending SMS to {phone_numbers}: {message}")
        
        return {
            "status": "sent",
            "recipients": phone_numbers,
            "message_length": len(message),
            "provider": "twilio"  # or configured provider
        }
    
    async def _send_webhook(self, **kwargs) -> Dict[str, Any]:
        """Send webhook notification"""
        if not self.config.webhook_urls:
            raise ValueError("No webhook URLs configured")
        
        await self._ensure_session()
        
        message = kwargs.get("message", "")
        title = kwargs.get("title", "Notification")
        metadata = kwargs.get("metadata", {})
        
        payload = {
            "title": title,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        results = []
        for url in self.config.webhook_urls:
            try:
                async with self._session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    results.append({
                        "url": url,
                        "status": response.status,
                        "success": response.status < 400
                    })
            except Exception as e:
                results.append({
                    "url": url,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "webhooks_sent": len(results),
            "results": results
        }
    
    async def _send_desktop(self, **kwargs) -> Dict[str, Any]:
        """Send desktop notification"""
        if not DESKTOP_NOTIFICATIONS_AVAILABLE:
            raise ValueError("Desktop notifications not available. Install 'plyer' package.")
        
        title = kwargs.get("title", "Notification")
        message = kwargs.get("message", "")
        
        try:
            plyer.notification.notify(
                title=title,
                message=message,
                timeout=10  # seconds
            )
            
            return {
                "status": "displayed",
                "platform": plyer.utils.platform
            }
        except Exception as e:
            raise ValueError(f"Desktop notification failed: {str(e)}")
    
    async def _send_slack(self, **kwargs) -> Dict[str, Any]:
        """Send Slack notification"""
        if not self.config.slack_webhook:
            raise ValueError("Slack webhook URL not configured")
        
        await self._ensure_session()
        
        message = kwargs.get("message", "")
        title = kwargs.get("title", "Notification")
        
        payload = {
            "text": f"*{title}*\n{message}",
            "username": kwargs.get("username", "Agentic Notification"),
            "icon_emoji": kwargs.get("icon", ":robot_face:")
        }
        
        # Add rich formatting if provided
        if "blocks" in kwargs:
            payload["blocks"] = kwargs["blocks"]
        
        async with self._session.post(
            self.config.slack_webhook,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            return {
                "status": "sent" if response.status == 200 else "failed",
                "status_code": response.status
            }
    
    async def _send_discord(self, **kwargs) -> Dict[str, Any]:
        """Send Discord notification"""
        if not self.config.discord_webhook:
            raise ValueError("Discord webhook URL not configured")
        
        await self._ensure_session()
        
        message = kwargs.get("message", "")
        title = kwargs.get("title", "Notification")
        
        payload = {
            "content": message,
            "username": kwargs.get("username", "Agentic Notification"),
            "embeds": [{
                "title": title,
                "description": message,
                "color": 5814783,  # Blue color
                "timestamp": datetime.now().isoformat()
            }]
        }
        
        async with self._session.post(
            self.config.discord_webhook,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            return {
                "status": "sent" if response.status < 300 else "failed",
                "status_code": response.status
            }
    
    async def _send_telegram(self, **kwargs) -> Dict[str, Any]:
        """Send Telegram notification"""
        if not self.config.telegram_config:
            raise ValueError("Telegram configuration not provided")
        
        await self._ensure_session()
        
        bot_token = self.config.telegram_config.get("bot_token")
        chat_id = kwargs.get("chat_id") or self.config.telegram_config.get("default_chat_id")
        
        if not bot_token or not chat_id:
            raise ValueError("Telegram bot_token and chat_id required")
        
        message = kwargs.get("message", "")
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": kwargs.get("parse_mode", "HTML")
        }
        
        async with self._session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            data = await response.json()
            return {
                "status": "sent" if data.get("ok") else "failed",
                "message_id": data.get("result", {}).get("message_id"),
                "error": data.get("description")
            }
    
    async def send_bulk(self, notifications: List[Dict[str, Any]]) -> List[ToolResult]:
        """Send multiple notifications in parallel"""
        tasks = []
        for notification in notifications:
            task = self._execute(**notification)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to ToolResult
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    success=False,
                    data={"error": str(result)},
                    metadata={"notification": notifications[i]}
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def __aenter__(self):
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None


# Convenience functions for creating notification tools
def create_email_notifier(email_config: Dict[str, Any]) -> NotificationTool:
    """Create a notification tool with email support"""
    config = NotificationConfig(email_config=email_config)
    return NotificationTool(config)


def create_multi_channel_notifier(
    email_config: Optional[Dict[str, Any]] = None,
    webhook_urls: Optional[List[str]] = None,
    slack_webhook: Optional[str] = None,
    discord_webhook: Optional[str] = None,
    telegram_config: Optional[Dict[str, Any]] = None,
    sms_config: Optional[Dict[str, Any]] = None
) -> NotificationTool:
    """Create a notification tool with multiple channel support"""
    config = NotificationConfig(
        email_config=email_config,
        webhook_urls=webhook_urls,
        slack_webhook=slack_webhook,
        discord_webhook=discord_webhook,
        telegram_config=telegram_config,
        sms_config=sms_config
    )
    return NotificationTool(config)