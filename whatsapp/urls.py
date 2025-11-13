from django.urls import path
from . import views

urlpatterns = [
    path("", views.login_page, name="home"),  # root URL redirects to login_page
    path("login/", views.login_page, name="login_page"),
    path("login/start", views.login_start, name="login_start"),
    path("login/wait/<uuid:session_id>/", views.login_wait, name="login_wait"),
    path("ingest/session/<uuid:session_id>/state", views.ingest_session_state, name="ingest_session_state"),
]
