# from django.urls import path
# from . import views

# urlpatterns = [
#     path("", views.login_page, name="home"),  # root URL redirects to login_page
#     path("login/", views.login_page, name="login_page"),
#     path("login/start", views.login_start, name="login_start"),
#     path("login/wait/<uuid:session_id>/", views.login_wait, name="login_wait"),
#     path("ingest/session/<uuid:session_id>/state", views.ingest_session_state, name="ingest_session_state"),
# ]
from django.urls import path
from . import views

urlpatterns = [
    # Landing / login
    path("", views.login_page, name="home"),  # root URL redirects to login_page
    path("login/", views.login_page, name="login_page"),
    path("login/start", views.login_start, name="login_start"),
    path("login/wait/<uuid:session_id>/", views.login_wait, name="login_wait"),

    # Node → Django: session state + scraping
    path("ingest/session/<uuid:session_id>/state", views.ingest_session_state, name="ingest_session_state"),
    path("ingest/unread/<uuid:session_id>/batch", views.ingest_unread_batch, name="ingest_unread_batch"),
    path("ingest/scrape/<uuid:session_id>/state", views.ingest_scrape_state, name="ingest_scrape_state"),

    # Main unread page
    path("unread/", views.unread_page, name="unread_page"),
]
