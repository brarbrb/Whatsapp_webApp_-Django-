from django.apps import AppConfig
from django.conf import settings
from django.db.utils import OperationalError, ProgrammingError


class WhatsappConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "whatsapp"

    def ready(self):
        """
        Runs when Django starts this app.

        In DEBUG mode, we reset all Session rows to is_active=False so we
        don't carry over old sessions between dev runs.
        """
        if not settings.DEBUG:
            return  # do nothing in production

        # Import inside ready() to avoid circular imports
        from .models import Session

        try:
            Session.objects.filter(is_active=True).update(is_active=False) # rises errors, nothing critical
        except (OperationalError, ProgrammingError):
            # Table doesn't exist yet (first migrate) – ignore.
            pass