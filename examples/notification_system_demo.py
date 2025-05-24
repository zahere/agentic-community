"""
Notification System Demo - Multi-channel notifications with the Agentic Framework

This example demonstrates:
1. Email notifications
2. Webhook notifications  
3. Slack/Discord notifications
4. Desktop notifications
5. Bulk notification sending
6. Priority-based notifications
"""

import asyncio
import os
from typing import List, Dict, Any

# Import from the agentic community package
from agentic_community.agents import SimpleAgent
from agentic_community.tools import NotificationTool
from agentic_community.tools.notification_tool import NotificationConfig, NotificationChannel
from agentic_community.llm import LLMProvider


async def demonstrate_notification_system():
    """Demonstrate the notification system capabilities"""
    
    print("🔔 Agentic Framework - Notification System Demo\n")
    
    # Configure notification channels
    # Note: In production, these would come from environment variables or config files
    config = NotificationConfig(
        # Email configuration (requires email server setup)
        email_config={
            "smtp_host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "imap_host": os.getenv("IMAP_HOST", "imap.gmail.com"),
            "imap_port": int(os.getenv("IMAP_PORT", "993")),
            "username": os.getenv("EMAIL_USERNAME", "demo@example.com"),
            "password": os.getenv("EMAIL_PASSWORD", "demo_password"),
            "use_tls": True
        },
        # Webhook URLs for testing
        webhook_urls=[
            "https://webhook.site/test",  # Replace with your webhook URL
            "https://httpbin.org/post"    # Public test endpoint
        ],
        # Slack webhook (requires Slack app setup)
        slack_webhook=os.getenv("SLACK_WEBHOOK"),
        # Discord webhook (requires Discord server setup)
        discord_webhook=os.getenv("DISCORD_WEBHOOK"),
        # Telegram configuration
        telegram_config={
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "default_chat_id": os.getenv("TELEGRAM_CHAT_ID")
        }
    )
    
    # Create notification tool
    notification_tool = NotificationTool(config)
    
    # Create an agent with notification capabilities
    agent = SimpleAgent(
        name="NotificationAgent",
        llm_provider=LLMProvider(
            provider="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY", "demo-key")
        ),
        tools=[notification_tool],
        system_prompt="""You are a notification management agent capable of sending 
        alerts and messages through multiple channels. You help users stay informed
        about important events and updates."""
    )
    
    # Example 1: Desktop notification
    print("1️⃣ Sending desktop notification...")
    try:
        desktop_result = await notification_tool._execute(
            channel="desktop",
            title="Agentic Framework",
            message="Welcome to the notification system demo!"
        )
        print(f"   ✅ Desktop notification: {desktop_result.data}\n")
    except Exception as e:
        print(f"   ⚠️  Desktop notifications not available: {e}\n")
    
    # Example 2: Webhook notification
    print("2️⃣ Sending webhook notifications...")
    webhook_result = await notification_tool._execute(
        channel="webhook",
        title="System Alert",
        message="New deployment completed successfully",
        metadata={
            "deployment_id": "dep_123",
            "environment": "production",
            "version": "2.0.1"
        }
    )
    print(f"   ✅ Webhook results: {webhook_result.data}\n")
    
    # Example 3: Slack notification (if configured)
    if config.slack_webhook:
        print("3️⃣ Sending Slack notification...")
        slack_result = await notification_tool._execute(
            channel="slack",
            title="🚀 Deployment Success",
            message="Version 2.0.1 deployed to production",
            icon=":rocket:",
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Deployment Details:*\n• Version: `2.0.1`\n• Environment: `production`\n• Status: ✅ Success"
                    }
                }
            ]
        )
        print(f"   ✅ Slack notification: {slack_result.data}\n")
    
    # Example 4: Priority-based notification routing
    print("4️⃣ Demonstrating priority-based routing...")
    
    async def send_priority_notification(priority: str, message: str):
        """Route notifications based on priority"""
        channels = []
        
        if priority == "critical":
            channels = ["desktop", "email", "slack", "sms"]
        elif priority == "high":
            channels = ["email", "slack"]
        elif priority == "normal":
            channels = ["email"]
        else:  # low
            channels = ["webhook"]
        
        print(f"   Priority: {priority} → Channels: {channels}")
        
        results = []
        for channel in channels:
            try:
                result = await notification_tool._execute(
                    channel=channel,
                    title=f"{priority.upper()} Alert",
                    message=message,
                    priority=priority
                )
                results.append(result)
            except Exception as e:
                print(f"   ⚠️  {channel} failed: {e}")
        
        return results
    
    # Send notifications with different priorities
    await send_priority_notification("critical", "Database connection lost!")
    await send_priority_notification("normal", "Daily backup completed")
    print()
    
    # Example 5: Bulk notifications
    print("5️⃣ Sending bulk notifications...")
    
    notifications = [
        {
            "channel": "webhook",
            "title": "User Registration",
            "message": "New user: user123",
            "metadata": {"user_id": "123"}
        },
        {
            "channel": "webhook",
            "title": "Order Placed",
            "message": "Order #456 received",
            "metadata": {"order_id": "456"}
        },
        {
            "channel": "desktop",
            "title": "Tasks Complete",
            "message": "All scheduled tasks finished"
        }
    ]
    
    bulk_results = await notification_tool.send_bulk(notifications)
    success_count = sum(1 for r in bulk_results if r.success)
    print(f"   ✅ Sent {success_count}/{len(notifications)} notifications\n")
    
    # Example 6: Agent-driven notifications
    print("6️⃣ Agent-driven notification scenario...")
    
    # Simulate a monitoring scenario
    monitoring_prompt = """
    The system has detected the following events:
    1. CPU usage is at 95% on server prod-web-01
    2. Response time increased to 2.5 seconds (threshold: 2s)
    3. 50 failed login attempts from IP 192.168.1.100
    
    Please analyze these events and send appropriate notifications.
    """
    
    response = await agent.process(monitoring_prompt)
    print(f"   Agent response: {response}\n")
    
    # Example 7: Template-based notifications
    print("7️⃣ Template-based notifications...")
    
    class NotificationTemplates:
        @staticmethod
        def deployment_success(version: str, environment: str) -> Dict[str, Any]:
            return {
                "title": "🎉 Deployment Successful",
                "message": f"Version {version} has been successfully deployed to {environment}",
                "metadata": {
                    "version": version,
                    "environment": environment,
                    "timestamp": asyncio.get_event_loop().time()
                }
            }
        
        @staticmethod
        def error_alert(service: str, error: str) -> Dict[str, Any]:
            return {
                "title": f"⚠️ Error in {service}",
                "message": f"An error occurred: {error}",
                "priority": "high",
                "metadata": {
                    "service": service,
                    "error_type": type(error).__name__ if hasattr(error, '__name__') else "Unknown"
                }
            }
    
    # Use templates
    deployment_notification = NotificationTemplates.deployment_success("2.1.0", "staging")
    await notification_tool._execute(channel="webhook", **deployment_notification)
    
    error_notification = NotificationTemplates.error_alert("AuthService", "Token validation failed")
    await notification_tool._execute(channel="webhook", **error_notification)
    
    print("   ✅ Template notifications sent\n")
    
    # Summary
    print("📊 Notification Demo Summary:")
    print("   • Multi-channel support: Email, SMS, Webhooks, Slack, Discord, Telegram")
    print("   • Priority-based routing for critical alerts")
    print("   • Bulk notification capabilities")
    print("   • Agent integration for intelligent notifications")
    print("   • Template system for consistent messaging")
    print("   • Async operation for high performance")
    
    print("\n✨ The notification system provides a unified interface for all your alerting needs!")


async def main():
    """Main entry point"""
    try:
        await demonstrate_notification_system()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("🔔 Agentic Framework - Notification System Demo")
    print("=" * 60)
    
    # Note about configuration
    print("\n⚙️  Configuration Note:")
    print("   Set the following environment variables for full functionality:")
    print("   - OPENAI_API_KEY: Your OpenAI API key")
    print("   - SMTP_HOST, SMTP_PORT, EMAIL_USERNAME, EMAIL_PASSWORD: Email config")
    print("   - SLACK_WEBHOOK: Your Slack webhook URL")
    print("   - DISCORD_WEBHOOK: Your Discord webhook URL")
    print("   - TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID: Telegram config")
    print("\n" + "=" * 60 + "\n")
    
    asyncio.run(main())