"""
Email Tool for Agentic Community Edition

This tool provides email capabilities including sending, receiving, and managing emails.
Supports multiple email providers and protocols.
"""

import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime
import re

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class EmailConfig(BaseModel):
    """Configuration for email operations."""
    smtp_server: str = Field(description="SMTP server address")
    smtp_port: int = Field(default=587, description="SMTP server port")
    imap_server: str = Field(description="IMAP server address")
    imap_port: int = Field(default=993, description="IMAP server port")
    username: str = Field(description="Email username")
    password: str = Field(description="Email password")
    use_tls: bool = Field(default=True, description="Use TLS encryption")


class EmailMessage(BaseModel):
    """Email message structure."""
    to: List[str] = Field(description="Recipient email addresses")
    cc: Optional[List[str]] = Field(default=None, description="CC recipients")
    bcc: Optional[List[str]] = Field(default=None, description="BCC recipients")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body")
    html_body: Optional[str] = Field(default=None, description="HTML email body")
    attachments: Optional[List[str]] = Field(default=None, description="File paths to attach")
    reply_to: Optional[str] = Field(default=None, description="Reply-to address")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Additional headers")


class EmailTool(BaseTool):
    """Tool for email operations."""
    
    name: str = "email_tool"
    description: str = """
    A tool for email operations including:
    - Sending emails with attachments
    - Reading emails from inbox
    - Searching emails
    - Managing email folders
    - Replying to emails
    - Forwarding emails
    """
    
    config: Optional[EmailConfig] = None
    
    def __init__(self, config: Optional[EmailConfig] = None):
        super().__init__()
        self.config = config or self._load_config_from_env()
        
    def _load_config_from_env(self) -> EmailConfig:
        """Load email configuration from environment variables."""
        return EmailConfig(
            smtp_server=os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=int(os.getenv("EMAIL_SMTP_PORT", "587")),
            imap_server=os.getenv("EMAIL_IMAP_SERVER", "imap.gmail.com"),
            imap_port=int(os.getenv("EMAIL_IMAP_PORT", "993")),
            username=os.getenv("EMAIL_USERNAME", ""),
            password=os.getenv("EMAIL_PASSWORD", ""),
            use_tls=os.getenv("EMAIL_USE_TLS", "true").lower() == "true"
        )
        
    def _run(self, query: str) -> str:
        """Run the email tool based on the query."""
        # Parse the query to determine the action
        query_lower = query.lower()
        
        if "send" in query_lower:
            return self._handle_send_query(query)
        elif "read" in query_lower or "check" in query_lower:
            return self._handle_read_query(query)
        elif "search" in query_lower:
            return self._handle_search_query(query)
        elif "reply" in query_lower:
            return self._handle_reply_query(query)
        elif "forward" in query_lower:
            return self._handle_forward_query(query)
        else:
            return "Please specify an email action: send, read, search, reply, or forward"
            
    async def _arun(self, query: str) -> str:
        """Async version of run."""
        return self._run(query)
        
    def send_email(self, message: EmailMessage) -> Dict[str, Any]:
        """Send an email."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config.username
            msg['To'] = ', '.join(message.to)
            msg['Subject'] = message.subject
            
            if message.cc:
                msg['Cc'] = ', '.join(message.cc)
            
            if message.reply_to:
                msg['Reply-To'] = message.reply_to
                
            if message.headers:
                for key, value in message.headers.items():
                    msg[key] = value
            
            # Add body
            if message.body:
                msg.attach(MIMEText(message.body, 'plain'))
                
            if message.html_body:
                msg.attach(MIMEText(message.html_body, 'html'))
            
            # Add attachments
            if message.attachments:
                for filepath in message.attachments:
                    self._attach_file(msg, filepath)
            
            # Connect and send
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()
                server.login(self.config.username, self.config.password)
                
                # Combine all recipients
                all_recipients = message.to
                if message.cc:
                    all_recipients.extend(message.cc)
                if message.bcc:
                    all_recipients.extend(message.bcc)
                    
                server.send_message(msg, to_addrs=all_recipients)
                
            return {
                "success": True,
                "message": f"Email sent successfully to {', '.join(message.to)}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def read_emails(
        self,
        folder: str = "INBOX",
        limit: int = 10,
        unread_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Read emails from a folder."""
        emails = []
        
        try:
            # Connect to IMAP server
            with imaplib.IMAP4_SSL(self.config.imap_server, self.config.imap_port) as mail:
                mail.login(self.config.username, self.config.password)
                mail.select(folder)
                
                # Search for emails
                search_criteria = 'UNSEEN' if unread_only else 'ALL'
                _, message_numbers = mail.search(None, search_criteria)
                
                # Get latest emails
                message_ids = message_numbers[0].split()
                latest_ids = message_ids[-limit:] if len(message_ids) > limit else message_ids
                
                for msg_id in reversed(latest_ids):
                    _, msg_data = mail.fetch(msg_id, '(RFC822)')
                    
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            
                            # Extract email details
                            email_dict = {
                                "id": msg_id.decode(),
                                "from": msg.get("From"),
                                "to": msg.get("To"),
                                "subject": msg.get("Subject"),
                                "date": msg.get("Date"),
                                "body": self._get_email_body(msg),
                                "attachments": self._get_attachments(msg)
                            }
                            emails.append(email_dict)
                            
        except Exception as e:
            return [{"error": str(e)}]
            
        return emails
        
    def search_emails(
        self,
        search_query: str,
        folder: str = "INBOX",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search emails based on criteria."""
        emails = []
        
        try:
            with imaplib.IMAP4_SSL(self.config.imap_server, self.config.imap_port) as mail:
                mail.login(self.config.username, self.config.password)
                mail.select(folder)
                
                # Build search criteria
                criteria = self._build_search_criteria(search_query)
                _, message_numbers = mail.search(None, criteria)
                
                message_ids = message_numbers[0].split()
                latest_ids = message_ids[-limit:] if len(message_ids) > limit else message_ids
                
                for msg_id in reversed(latest_ids):
                    _, msg_data = mail.fetch(msg_id, '(RFC822)')
                    
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            
                            email_dict = {
                                "id": msg_id.decode(),
                                "from": msg.get("From"),
                                "to": msg.get("To"),
                                "subject": msg.get("Subject"),
                                "date": msg.get("Date"),
                                "snippet": self._get_email_body(msg)[:200] + "..."
                            }
                            emails.append(email_dict)
                            
        except Exception as e:
            return [{"error": str(e)}]
            
        return emails
        
    def reply_to_email(
        self,
        original_email_id: str,
        reply_body: str,
        reply_all: bool = False
    ) -> Dict[str, Any]:
        """Reply to an email."""
        try:
            # Fetch original email
            with imaplib.IMAP4_SSL(self.config.imap_server, self.config.imap_port) as mail:
                mail.login(self.config.username, self.config.password)
                mail.select("INBOX")
                
                _, msg_data = mail.fetch(original_email_id.encode(), '(RFC822)')
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        original_msg = email.message_from_bytes(response_part[1])
                        
                        # Create reply
                        reply = MIMEMultipart()
                        reply['From'] = self.config.username
                        reply['To'] = original_msg.get("From")
                        
                        if reply_all and original_msg.get("Cc"):
                            reply['Cc'] = original_msg.get("Cc")
                            
                        # Add Re: to subject if not present
                        original_subject = original_msg.get("Subject", "")
                        if not original_subject.startswith("Re:"):
                            reply['Subject'] = f"Re: {original_subject}"
                        else:
                            reply['Subject'] = original_subject
                            
                        # Add In-Reply-To header
                        if original_msg.get("Message-ID"):
                            reply['In-Reply-To'] = original_msg.get("Message-ID")
                            reply['References'] = original_msg.get("Message-ID")
                        
                        # Create reply body with quoted original
                        quoted_body = self._quote_email_body(
                            self._get_email_body(original_msg),
                            original_msg.get("From"),
                            original_msg.get("Date")
                        )
                        
                        full_body = f"{reply_body}\n\n{quoted_body}"
                        reply.attach(MIMEText(full_body, 'plain'))
                        
                        # Send reply
                        with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                            if self.config.use_tls:
                                server.starttls()
                            server.login(self.config.username, self.config.password)
                            server.send_message(reply)
                            
                        return {
                            "success": True,
                            "message": "Reply sent successfully"
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def forward_email(
        self,
        original_email_id: str,
        forward_to: List[str],
        forward_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Forward an email."""
        try:
            # Fetch original email
            with imaplib.IMAP4_SSL(self.config.imap_server, self.config.imap_port) as mail:
                mail.login(self.config.username, self.config.password)
                mail.select("INBOX")
                
                _, msg_data = mail.fetch(original_email_id.encode(), '(RFC822)')
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        original_msg = email.message_from_bytes(response_part[1])
                        
                        # Create forward message
                        fwd = MIMEMultipart()
                        fwd['From'] = self.config.username
                        fwd['To'] = ', '.join(forward_to)
                        fwd['Subject'] = f"Fwd: {original_msg.get('Subject', '')}"
                        
                        # Create forward body
                        forward_body = ""
                        if forward_message:
                            forward_body = f"{forward_message}\n\n"
                            
                        forward_body += f"---------- Forwarded message ---------\n"
                        forward_body += f"From: {original_msg.get('From')}\n"
                        forward_body += f"Date: {original_msg.get('Date')}\n"
                        forward_body += f"Subject: {original_msg.get('Subject')}\n"
                        forward_body += f"To: {original_msg.get('To')}\n\n"
                        forward_body += self._get_email_body(original_msg)
                        
                        fwd.attach(MIMEText(forward_body, 'plain'))
                        
                        # Forward attachments if any
                        for part in original_msg.walk():
                            if part.get_content_disposition() == 'attachment':
                                fwd.attach(part)
                        
                        # Send forward
                        with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                            if self.config.use_tls:
                                server.starttls()
                            server.login(self.config.username, self.config.password)
                            server.send_message(fwd)
                            
                        return {
                            "success": True,
                            "message": f"Email forwarded to {', '.join(forward_to)}"
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def _attach_file(self, msg: MIMEMultipart, filepath: str):
        """Attach a file to an email message."""
        with open(filepath, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={os.path.basename(filepath)}'
            )
            msg.attach(part)
            
    def _get_email_body(self, msg: email.message.Message) -> str:
        """Extract email body from message."""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
        return body
        
    def _get_attachments(self, msg: email.message.Message) -> List[str]:
        """Get list of attachment filenames."""
        attachments = []
        
        for part in msg.walk():
            if part.get_content_disposition() == 'attachment':
                filename = part.get_filename()
                if filename:
                    attachments.append(filename)
                    
        return attachments
        
    def _quote_email_body(self, body: str, sender: str, date: str) -> str:
        """Quote email body for reply."""
        lines = body.split('\n')
        quoted_lines = [f"> {line}" for line in lines]
        
        header = f"On {date}, {sender} wrote:"
        return f"{header}\n" + '\n'.join(quoted_lines)
        
    def _build_search_criteria(self, query: str) -> str:
        """Build IMAP search criteria from natural language query."""
        criteria_parts = []
        
        # Extract sender
        sender_match = re.search(r'from[:\s]+(\S+)', query, re.IGNORECASE)
        if sender_match:
            criteria_parts.append(f'FROM "{sender_match.group(1)}"')
            
        # Extract subject
        subject_match = re.search(r'subject[:\s]+([^,]+)', query, re.IGNORECASE)
        if subject_match:
            criteria_parts.append(f'SUBJECT "{subject_match.group(1).strip()}"')
            
        # Extract date
        if "today" in query.lower():
            today = datetime.now().strftime("%d-%b-%Y")
            criteria_parts.append(f'ON {today}')
        elif "yesterday" in query.lower():
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%d-%b-%Y")
            criteria_parts.append(f'ON {yesterday}')
            
        # Default to ALL if no criteria
        if not criteria_parts:
            return 'ALL'
            
        return ' '.join(criteria_parts)
        
    def _handle_send_query(self, query: str) -> str:
        """Handle send email query."""
        # Extract email details from query
        # This is a simplified parser - in production, use NLP
        try:
            # Extract recipient
            to_match = re.search(r'to[:\s]+(\S+@\S+)', query, re.IGNORECASE)
            if not to_match:
                return "Please specify recipient email address"
                
            # Extract subject
            subject_match = re.search(r'subject[:\s]+([^,]+)', query, re.IGNORECASE)
            subject = subject_match.group(1).strip() if subject_match else "No Subject"
            
            # Extract body
            body_match = re.search(r'body[:\s]+(.+)', query, re.IGNORECASE)
            body = body_match.group(1).strip() if body_match else query
            
            message = EmailMessage(
                to=[to_match.group(1)],
                subject=subject,
                body=body
            )
            
            result = self.send_email(message)
            return result["message"] if result["success"] else f"Error: {result['error']}"
            
        except Exception as e:
            return f"Error parsing email query: {str(e)}"
            
    def _handle_read_query(self, query: str) -> str:
        """Handle read email query."""
        # Determine parameters
        limit = 5
        unread_only = "unread" in query.lower()
        
        # Extract limit if specified
        limit_match = re.search(r'(\d+)\s+(email|emails)', query, re.IGNORECASE)
        if limit_match:
            limit = int(limit_match.group(1))
            
        emails = self.read_emails(limit=limit, unread_only=unread_only)
        
        if not emails:
            return "No emails found"
            
        # Format response
        response = f"Found {len(emails)} emails:\n\n"
        for email in emails:
            if "error" in email:
                return f"Error: {email['error']}"
                
            response += f"From: {email['from']}\n"
            response += f"Subject: {email['subject']}\n"
            response += f"Date: {email['date']}\n"
            response += f"Preview: {email['body'][:100]}...\n"
            response += "-" * 50 + "\n"
            
        return response
        
    def _handle_search_query(self, query: str) -> str:
        """Handle search email query."""
        emails = self.search_emails(query)
        
        if not emails:
            return "No emails found matching your search"
            
        response = f"Found {len(emails)} matching emails:\n\n"
        for email in emails:
            if "error" in email:
                return f"Error: {email['error']}"
                
            response += f"From: {email['from']}\n"
            response += f"Subject: {email['subject']}\n"
            response += f"Date: {email['date']}\n"
            response += f"Preview: {email['snippet']}\n"
            response += "-" * 50 + "\n"
            
        return response
        
    def _handle_reply_query(self, query: str) -> str:
        """Handle reply to email query."""
        # This would need email ID from context
        return "To reply to an email, please provide the email ID and your reply message"
        
    def _handle_forward_query(self, query: str) -> str:
        """Handle forward email query."""
        # This would need email ID from context
        return "To forward an email, please provide the email ID and recipient addresses"
