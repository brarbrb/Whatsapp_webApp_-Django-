# from django.apps import AppConfig


# class WhatsappConfig(AppConfig):
#     default_auto_field = 'django.db.models.BigAutoField'
#     name = 'whatsapp'

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
            # If migrations haven't run yet, this can raise errors.
            Session.objects.filter(is_active=True).update(is_active=False)
        except (OperationalError, ProgrammingError):
            # Table doesn't exist yet (e.g. first migrate) – ignore.
            pass
