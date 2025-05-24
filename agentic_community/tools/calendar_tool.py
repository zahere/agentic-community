"""
Calendar Tool for Agentic Community Edition

This tool provides calendar management capabilities including creating events,
checking availability, scheduling meetings, and managing calendar entries.
Supports multiple calendar providers.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
import os
import re
from dataclasses import dataclass
from enum import Enum

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# For production, you would import actual calendar APIs
# from google.oauth2.credentials import Credentials
# from googleapiclient.discovery import build
# import caldav


class EventStatus(Enum):
    """Status of a calendar event."""
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class RecurrenceFrequency(Enum):
    """Frequency of recurring events."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class CalendarEvent:
    """Calendar event structure."""
    title: str
    start_time: datetime
    end_time: datetime
    description: Optional[str] = None
    location: Optional[str] = None
    attendees: Optional[List[str]] = None
    status: EventStatus = EventStatus.CONFIRMED
    reminder_minutes: Optional[int] = 15
    recurrence: Optional[Dict[str, Any]] = None
    color: Optional[str] = None
    event_id: Optional[str] = None


class CalendarConfig(BaseModel):
    """Configuration for calendar operations."""
    provider: str = Field(default="google", description="Calendar provider (google, outlook, caldav)")
    calendar_id: str = Field(default="primary", description="Calendar ID to use")
    timezone: str = Field(default="UTC", description="Default timezone")
    # Add provider-specific config fields as needed


class CalendarTool(BaseTool):
    """Tool for calendar operations."""
    
    name: str = "calendar_tool"
    description: str = """
    A tool for calendar operations including:
    - Creating calendar events
    - Checking availability
    - Scheduling meetings
    - Finding free time slots
    - Managing recurring events
    - Sending meeting invitations
    - Checking conflicts
    """
    
    config: Optional[CalendarConfig] = None
    
    def __init__(self, config: Optional[CalendarConfig] = None):
        super().__init__()
        self.config = config or self._load_config_from_env()
        # In production, initialize calendar API client here
        self._events_cache: List[CalendarEvent] = []  # Mock storage
        
    def _load_config_from_env(self) -> CalendarConfig:
        """Load calendar configuration from environment variables."""
        return CalendarConfig(
            provider=os.getenv("CALENDAR_PROVIDER", "google"),
            calendar_id=os.getenv("CALENDAR_ID", "primary"),
            timezone=os.getenv("CALENDAR_TIMEZONE", "UTC")
        )
        
    def _run(self, query: str) -> str:
        """Run the calendar tool based on the query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["create", "schedule", "book", "add"]):
            return self._handle_create_event_query(query)
        elif any(word in query_lower for word in ["check", "availability", "free", "busy"]):
            return self._handle_check_availability_query(query)
        elif any(word in query_lower for word in ["list", "show", "what", "events"]):
            return self._handle_list_events_query(query)
        elif any(word in query_lower for word in ["cancel", "delete", "remove"]):
            return self._handle_cancel_event_query(query)
        elif any(word in query_lower for word in ["reschedule", "move", "change"]):
            return self._handle_reschedule_event_query(query)
        elif "find" in query_lower and "time" in query_lower:
            return self._handle_find_time_query(query)
        else:
            return "Please specify a calendar action: create event, check availability, list events, cancel event, reschedule event, or find time"
            
    async def _arun(self, query: str) -> str:
        """Async version of run."""
        return self._run(query)
        
    def create_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Create a calendar event."""
        try:
            # In production, this would use the actual calendar API
            # For now, we'll simulate the creation
            import uuid
            event.event_id = str(uuid.uuid4())
            
            # Check for conflicts
            conflicts = self.check_conflicts(event.start_time, event.end_time)
            if conflicts:
                return {
                    "success": False,
                    "error": f"Time conflict with: {conflicts[0]['title']}",
                    "conflicts": conflicts
                }
            
            # Add to cache (in production, add to actual calendar)
            self._events_cache.append(event)
            
            # Send invitations if attendees specified
            invitation_status = "No attendees specified"
            if event.attendees:
                invitation_status = f"Invitations sent to {len(event.attendees)} attendees"
            
            return {
                "success": True,
                "event_id": event.event_id,
                "message": f"Event '{event.title}' created successfully",
                "invitation_status": invitation_status,
                "event_details": {
                    "title": event.title,
                    "start": event.start_time.isoformat(),
                    "end": event.end_time.isoformat(),
                    "location": event.location,
                    "attendees": event.attendees
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def check_availability(
        self,
        start_time: datetime,
        end_time: datetime,
        calendars: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Check availability for a time period."""
        busy_times = []
        
        try:
            # In production, query actual calendar API
            # For now, check our cache
            for event in self._events_cache:
                if event.status == EventStatus.CANCELLED:
                    continue
                    
                # Check if event overlaps with requested time
                if (event.start_time < end_time and event.end_time > start_time):
                    busy_times.append({
                        "title": event.title,
                        "start": event.start_time.isoformat(),
                        "end": event.end_time.isoformat(),
                        "attendees": event.attendees
                    })
                    
            return busy_times
            
        except Exception as e:
            return [{"error": str(e)}]
            
    def find_free_slots(
        self,
        duration_minutes: int,
        search_start: datetime,
        search_end: datetime,
        min_slot_size: int = 30,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Find available time slots of specified duration."""
        free_slots = []
        
        try:
            # Get all busy times in the search period
            busy_times = self.check_availability(search_start, search_end)
            
            # Sort busy times by start time
            busy_times.sort(key=lambda x: x['start'])
            
            # Find gaps between busy times
            current_time = search_start
            
            for busy in busy_times:
                busy_start = datetime.fromisoformat(busy['start'])
                busy_end = datetime.fromisoformat(busy['end'])
                
                # Check gap before this busy time
                if busy_start > current_time:
                    gap_duration = (busy_start - current_time).total_seconds() / 60
                    
                    if gap_duration >= duration_minutes:
                        free_slots.append({
                            "start": current_time.isoformat(),
                            "end": (current_time + timedelta(minutes=duration_minutes)).isoformat(),
                            "duration_minutes": duration_minutes
                        })
                        
                        if len(free_slots) >= max_results:
                            break
                            
                current_time = max(current_time, busy_end)
            
            # Check final gap after last busy time
            if current_time < search_end and len(free_slots) < max_results:
                remaining_time = (search_end - current_time).total_seconds() / 60
                if remaining_time >= duration_minutes:
                    free_slots.append({
                        "start": current_time.isoformat(),
                        "end": (current_time + timedelta(minutes=duration_minutes)).isoformat(),
                        "duration_minutes": duration_minutes
                    })
                    
            return free_slots
            
        except Exception as e:
            return [{"error": str(e)}]
            
    def list_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """List calendar events."""
        events = []
        
        try:
            # Default to next 7 days if no dates specified
            if not start_date:
                start_date = datetime.now(timezone.utc)
            if not end_date:
                end_date = start_date + timedelta(days=7)
                
            # Filter and sort events
            filtered_events = [
                event for event in self._events_cache
                if event.status != EventStatus.CANCELLED
                and event.start_time >= start_date
                and event.start_time <= end_date
            ]
            
            # Sort by start time
            filtered_events.sort(key=lambda x: x.start_time)
            
            # Convert to dict format
            for event in filtered_events[:limit]:
                events.append({
                    "event_id": event.event_id,
                    "title": event.title,
                    "start": event.start_time.isoformat(),
                    "end": event.end_time.isoformat(),
                    "location": event.location,
                    "description": event.description,
                    "attendees": event.attendees,
                    "status": event.status.value
                })
                
            return events
            
        except Exception as e:
            return [{"error": str(e)}]
            
    def cancel_event(self, event_id: str, notify_attendees: bool = True) -> Dict[str, Any]:
        """Cancel a calendar event."""
        try:
            # Find event
            event = None
            for e in self._events_cache:
                if e.event_id == event_id:
                    event = e
                    break
                    
            if not event:
                return {
                    "success": False,
                    "error": "Event not found"
                }
                
            # Cancel event
            event.status = EventStatus.CANCELLED
            
            # Notify attendees
            notification_status = "No attendees to notify"
            if notify_attendees and event.attendees:
                notification_status = f"Cancellation sent to {len(event.attendees)} attendees"
                
            return {
                "success": True,
                "message": f"Event '{event.title}' cancelled",
                "notification_status": notification_status
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def reschedule_event(
        self,
        event_id: str,
        new_start: datetime,
        new_end: datetime,
        notify_attendees: bool = True
    ) -> Dict[str, Any]:
        """Reschedule a calendar event."""
        try:
            # Find event
            event = None
            for e in self._events_cache:
                if e.event_id == event_id:
                    event = e
                    break
                    
            if not event:
                return {
                    "success": False,
                    "error": "Event not found"
                }
                
            # Check conflicts at new time
            conflicts = self.check_conflicts(new_start, new_end, exclude_event_id=event_id)
            if conflicts:
                return {
                    "success": False,
                    "error": f"Time conflict at new time with: {conflicts[0]['title']}",
                    "conflicts": conflicts
                }
                
            # Update event times
            old_start = event.start_time
            event.start_time = new_start
            event.end_time = new_end
            
            # Notify attendees
            notification_status = "No attendees to notify"
            if notify_attendees and event.attendees:
                notification_status = f"Reschedule notification sent to {len(event.attendees)} attendees"
                
            return {
                "success": True,
                "message": f"Event '{event.title}' rescheduled",
                "old_time": old_start.isoformat(),
                "new_time": new_start.isoformat(),
                "notification_status": notification_status
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def check_conflicts(
        self,
        start_time: datetime,
        end_time: datetime,
        exclude_event_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Check for scheduling conflicts."""
        conflicts = []
        
        for event in self._events_cache:
            if event.status == EventStatus.CANCELLED:
                continue
            if exclude_event_id and event.event_id == exclude_event_id:
                continue
                
            # Check if times overlap
            if (event.start_time < end_time and event.end_time > start_time):
                conflicts.append({
                    "event_id": event.event_id,
                    "title": event.title,
                    "start": event.start_time.isoformat(),
                    "end": event.end_time.isoformat()
                })
                
        return conflicts
        
    def _parse_datetime(self, time_str: str) -> Optional[datetime]:
        """Parse datetime from natural language."""
        # Simple parser - in production use dateutil or similar
        now = datetime.now(timezone.utc)
        
        # Handle relative times
        if "tomorrow" in time_str.lower():
            base_date = now + timedelta(days=1)
        elif "next week" in time_str.lower():
            base_date = now + timedelta(weeks=1)
        else:
            base_date = now
            
        # Extract time
        time_match = re.search(r'(\d{1,2}):?(\d{0,2})\s*(am|pm)?', time_str.lower())
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            
            if time_match.group(3) == 'pm' and hour < 12:
                hour += 12
            elif time_match.group(3) == 'am' and hour == 12:
                hour = 0
                
            return base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
        return None
        
    def _handle_create_event_query(self, query: str) -> str:
        """Handle create event query."""
        try:
            # Parse event details from query
            # This is simplified - in production use NLP
            
            # Extract title
            title_match = re.search(r'(?:create|schedule|book|add)\s+(?:a\s+)?(?:meeting|event|appointment)?\s*(?:called|titled|named|about|for)?\s*["\']?([^"\']+)["\']?', query, re.IGNORECASE)
            if not title_match:
                # Try to extract from context
                title = "New Event"
            else:
                title = title_match.group(1).strip()
                
            # Extract time
            # Look for patterns like "tomorrow at 3pm" or "next week at 10:30am"
            start_time = self._parse_datetime(query)
            if not start_time:
                return "Please specify when to schedule the event"
                
            # Default duration 1 hour
            duration_match = re.search(r'for\s+(\d+)\s*(hour|minute|min)', query, re.IGNORECASE)
            if duration_match:
                duration = int(duration_match.group(1))
                unit = duration_match.group(2).lower()
                if 'hour' in unit:
                    duration_minutes = duration * 60
                else:
                    duration_minutes = duration
            else:
                duration_minutes = 60
                
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            # Extract attendees
            attendees = []
            attendee_match = re.search(r'with\s+([^,]+(?:,\s*[^,]+)*)', query, re.IGNORECASE)
            if attendee_match:
                attendees = [a.strip() for a in attendee_match.group(1).split(',')]
                
            # Extract location
            location = None
            location_match = re.search(r'(?:at|in)\s+([^,]+)', query, re.IGNORECASE)
            if location_match:
                location = location_match.group(1).strip()
                
            # Create event
            event = CalendarEvent(
                title=title,
                start_time=start_time,
                end_time=end_time,
                location=location,
                attendees=attendees if attendees else None
            )
            
            result = self.create_event(event)
            
            if result["success"]:
                response = f"✅ {result['message']}\n"
                response += f"📅 Time: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}\n"
                if location:
                    response += f"📍 Location: {location}\n"
                if attendees:
                    response += f"👥 Attendees: {', '.join(attendees)}\n"
                response += f"🔔 Reminder set for {event.reminder_minutes} minutes before"
            else:
                response = f"❌ Failed to create event: {result['error']}"
                if "conflicts" in result:
                    response += "\nConflicts with:"
                    for conflict in result["conflicts"]:
                        response += f"\n  - {conflict['title']} at {conflict['start']}"
                        
            return response
            
        except Exception as e:
            return f"Error creating event: {str(e)}"
            
    def _handle_check_availability_query(self, query: str) -> str:
        """Handle check availability query."""
        try:
            # Parse time range
            start_time = self._parse_datetime(query)
            if not start_time:
                # Default to today
                start_time = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
                
            # Default to business hours
            end_time = start_time.replace(hour=17, minute=0)
            
            # Check availability
            busy_times = self.check_availability(start_time, end_time)
            
            if not busy_times:
                return f"✅ You are free from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} on {start_time.strftime('%Y-%m-%d')}"
                
            response = f"📅 Schedule for {start_time.strftime('%Y-%m-%d')}:\n\n"
            response += "Busy times:\n"
            
            for busy in busy_times:
                busy_start = datetime.fromisoformat(busy['start'])
                busy_end = datetime.fromisoformat(busy['end'])
                response += f"  🔴 {busy_start.strftime('%H:%M')} - {busy_end.strftime('%H:%M')}: {busy['title']}\n"
                
            # Find free slots
            free_slots = self.find_free_slots(30, start_time, end_time)
            if free_slots:
                response += "\nAvailable slots:\n"
                for slot in free_slots[:3]:
                    slot_start = datetime.fromisoformat(slot['start'])
                    slot_end = datetime.fromisoformat(slot['end'])
                    response += f"  🟢 {slot_start.strftime('%H:%M')} - {slot_end.strftime('%H:%M')}\n"
                    
            return response
            
        except Exception as e:
            return f"Error checking availability: {str(e)}"
            
    def _handle_list_events_query(self, query: str) -> str:
        """Handle list events query."""
        try:
            # Determine time range
            if "today" in query.lower():
                start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date + timedelta(days=1)
            elif "tomorrow" in query.lower():
                start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                end_date = start_date + timedelta(days=1)
            elif "week" in query.lower():
                start_date = datetime.now(timezone.utc)
                end_date = start_date + timedelta(days=7)
            else:
                # Default to next 3 days
                start_date = datetime.now(timezone.utc)
                end_date = start_date + timedelta(days=3)
                
            events = self.list_events(start_date, end_date)
            
            if not events:
                return f"No events scheduled from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                
            response = f"📅 Upcoming events:\n\n"
            
            current_date = None
            for event in events:
                event_start = datetime.fromisoformat(event['start'])
                event_end = datetime.fromisoformat(event['end'])
                
                # Group by date
                event_date = event_start.date()
                if event_date != current_date:
                    current_date = event_date
                    response += f"\n{event_date.strftime('%A, %B %d, %Y')}:\n"
                    
                response += f"  • {event_start.strftime('%H:%M')} - {event_end.strftime('%H:%M')}: {event['title']}"
                
                if event.get('location'):
                    response += f" 📍 {event['location']}"
                if event.get('attendees'):
                    response += f" 👥 {len(event['attendees'])} attendees"
                    
                response += "\n"
                
            return response
            
        except Exception as e:
            return f"Error listing events: {str(e)}"
            
    def _handle_cancel_event_query(self, query: str) -> str:
        """Handle cancel event query."""
        # In a real implementation, we'd need to identify which event to cancel
        # This could be done by title, time, or event ID
        return "To cancel an event, please specify which event (by title or time)"
        
    def _handle_reschedule_event_query(self, query: str) -> str:
        """Handle reschedule event query."""
        # Similar to cancel, we'd need to identify the event and new time
        return "To reschedule an event, please specify which event and the new time"
        
    def _handle_find_time_query(self, query: str) -> str:
        """Handle find available time query."""
        try:
            # Extract duration
            duration_match = re.search(r'(\d+)\s*(hour|minute|min)', query, re.IGNORECASE)
            if duration_match:
                duration = int(duration_match.group(1))
                unit = duration_match.group(2).lower()
                if 'hour' in unit:
                    duration_minutes = duration * 60
                else:
                    duration_minutes = duration
            else:
                duration_minutes = 60  # Default 1 hour
                
            # Search next 5 days
            search_start = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
            search_end = search_start + timedelta(days=5)
            
            free_slots = self.find_free_slots(
                duration_minutes,
                search_start,
                search_end,
                max_results=5
            )
            
            if not free_slots:
                return f"No available {duration_minutes}-minute slots found in the next 5 days"
                
            response = f"🔍 Available {duration_minutes}-minute slots:\n\n"
            
            for slot in free_slots:
                slot_start = datetime.fromisoformat(slot['start'])
                slot_end = datetime.fromisoformat(slot['end'])
                
                response += f"  ✅ {slot_start.strftime('%a %b %d, %H:%M')} - {slot_end.strftime('%H:%M')}\n"
                
            return response
            
        except Exception as e:
            return f"Error finding available time: {str(e)}"
