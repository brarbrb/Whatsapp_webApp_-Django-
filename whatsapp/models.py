import uuid
from django.db import models
from django.db.models import Q
from datetime import datetime


# Session is used to track WhatsApp Web sessions 
class Session(models.Model):
    class State(models.TextChoices):
        PENDING = "pending", "Pending"
        AUTHENTICATED = "authenticated", "Authenticated"
        READY = "ready", "Ready"
        DISCONNECTED = "disconnected", "Disconnected"
        FAILED = "failed", "Failed"
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # Node/whatsapp-web.js' internal session identifier
    node_session_id = models.CharField(max_length=255, blank=True, default="", db_index=True)
    state = models.CharField(max_length=32, choices=State.choices, default=State.PENDING, db_index=True)
    last_state_change = models.DateTimeField(default=datetime.now)
    last_error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(auto_now_add=True)
    last_seen_at = models.DateTimeField(auto_now=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True, db_index=True)

    def __str__(self):
        return f"Session {self.id} ({self.state})"

# Models to track unread/opened chats per session
class UnreadSession(models.Model):
    """Each row = one unread message captured during scraping for a session."""
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="unreads")
    wa_msg_id = models.CharField(max_length=512, db_index=True) # long + indexed for idempotency
    chat_id = models.CharField(max_length=255, db_index=True)
    sender = models.CharField(max_length=255, null=True, blank=True)
    body = models.TextField(blank=True, default="")
    msg_ts = models.DateTimeField(null=True, blank=True, db_index=True) # convert it from string to datetime when saving!
    scraped_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        unique_together = (("session", "wa_msg_id"),)
        indexes = [
            models.Index(fields=["session", "wa_msg_id"]),
            models.Index(fields=["session", "chat_id", "scraped_at"]),
        ]

    def __str__(self):
        return f"Unread[{self.chat_id}] {self.wa_msg_id} (sess={self.session_id})"
    
# Store all messages per chat 
class ChatMessage(models.Model):
    """All messages for a given chat in a given session."""
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="chat_messages")
    chat_id = models.CharField(max_length=255, db_index=True)
    wa_msg_id = models.CharField(max_length=512, db_index=True)
    sender = models.CharField(max_length=255, null=True, blank=True)
    body = models.TextField(blank=True, default="")
    from_me = models.BooleanField(default=False, db_index=True)
    msg_ts = models.DateTimeField(null=True, blank=True, db_index=True)
    scraped_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        unique_together = (("session", "wa_msg_id"),)
        indexes = [
            models.Index(fields=["session", "chat_id", "msg_ts"]),
        ]

    def __str__(self):
        return f"ChatMessage[{self.chat_id}] {self.wa_msg_id}"
