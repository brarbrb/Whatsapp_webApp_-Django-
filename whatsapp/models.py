import uuid
from django.db import models
from django.utils import timezone
from django.db.models import Q


# ──────────────────────────────────────────────────────────────────────────────
# Small-project friendly models for a single active user/session at a time.
# Keeps the structure minimal but robust (state tracking, idempotency, indexes).
# ──────────────────────────────────────────────────────────────────────────────

class Session(models.Model):
    class State(models.TextChoices):
        PENDING = "pending", "Pending"
        AUTHENTICATED = "authenticated", "Authenticated"
        READY = "ready", "Ready"
        DISCONNECTED = "disconnected", "Disconnected"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Node/whatsapp-web.js' internal session identifier (if you create one)
    node_session_id = models.CharField(max_length=255, blank=True, default="", db_index=True)

    state = models.CharField(max_length=32, choices=State.choices, default=State.PENDING, db_index=True)
    last_state_change = models.DateTimeField(default=timezone.now)
    last_error = models.TextField(blank=True, default="")

    started_at = models.DateTimeField(auto_now_add=True)
    last_seen_at = models.DateTimeField(auto_now=True)
    ended_at = models.DateTimeField(null=True, blank=True)

    is_active = models.BooleanField(default=True, db_index=True)

    def __str__(self):
        return f"Session {self.id} ({self.state})"


class ScrapeJob(models.Model):
    """Lightweight job row so the UI can poll a single place for progress."""
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        ERROR = "error", "Error"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="scrapes")

    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED, db_index=True)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    # Optional tiny metrics (handy for a progress bar)
    num_chats = models.IntegerField(default=0)
    num_msgs = models.IntegerField(default=0)

    error = models.TextField(blank=True, default="")

    class Meta:
        indexes = [
            models.Index(fields=["session", "status"]),
            models.Index(fields=["status", "started_at"]),
        ]

    def __str__(self):
        return f"ScrapeJob {self.id} [{self.status}]"


class UnreadSession(models.Model):
    """Each row = one unread message captured during scraping for a session."""
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="unreads")

    wa_msg_id = models.CharField(max_length=512, db_index=True)  # long + indexed for idempotency
    chat_id = models.CharField(max_length=255, db_index=True)

    sender = models.CharField(max_length=255, null=True, blank=True)
    body = models.TextField(blank=True, default="")

    # Store as timezone-aware DateTime; convert from epoch during ingestion if needed
    msg_ts = models.DateTimeField(null=True, blank=True, db_index=True)

    scraped_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        unique_together = (("session", "wa_msg_id"),)
        indexes = [
            models.Index(fields=["session", "wa_msg_id"]),
            models.Index(fields=["session", "chat_id", "scraped_at"]),
        ]

    def __str__(self):
        return f"Unread[{self.chat_id}] {self.wa_msg_id} (sess={self.session_id})"


class OpenedSession(models.Model):
    """Tracks when the user opens/replies to an unread message in a session."""
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="opened")

    wa_msg_id = models.CharField(max_length=512, db_index=True)
    chat_id = models.CharField(max_length=255, db_index=True)

    opened_at = models.DateTimeField(auto_now_add=True, db_index=True)

    # If user replied, link to the reply (optional)
    replied_msg_id = models.CharField(max_length=512, null=True, blank=True, db_index=True)
    replied_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        unique_together = (("session", "wa_msg_id"),)
        indexes = [
            models.Index(fields=["session", "wa_msg_id"]),
            models.Index(fields=["session", "chat_id", "opened_at"]),
        ]
        constraints = [
            # Keep replied_* in sync: either both null or both set
            models.CheckConstraint(
                name="opened_reply_fields_null_together",
                check=(
                    (Q(replied_msg_id__isnull=True) & Q(replied_at__isnull=True)) |
                    (Q(replied_msg_id__isnull=False) & Q(replied_at__isnull=False))
                ),
            ),
        ]

    def __str__(self):
        return f"Opened[{self.chat_id}] {self.wa_msg_id} (sess={self.session_id})"