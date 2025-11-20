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
        from .rag_pipe import WhatsAppRAG
        print("Initializing global RAG pipeline...")

        kb_path = "C:/Users/areg6/Documents/github/Whatsapp_webApp_-Django-/RAG/RAG_data/KB_data.csv"
        self.rag = WhatsAppRAG(
            kb_path=kb_path,
            receiver_user_id="u_barbara",
            context_conv_id="chat:u_barbara_u_maayan",
        )

        print("RAG pipeline initialized.")
        
        if not settings.DEBUG or settings.DEBUG:
            return  # do nothing in production

        # Import inside ready() to avoid circular imports when Django loads apps in the first time
        from .models import Session
        try:
            Session.objects.filter(is_active=True).update(is_active=False) # rises errors, nothing critical
        except (OperationalError, ProgrammingError):
            # Table doesn't exist yet (first migrate) – ignore.
            pass