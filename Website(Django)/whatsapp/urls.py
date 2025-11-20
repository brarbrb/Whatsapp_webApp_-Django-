from django.urls import path
from . import views

urlpatterns = [
    path("", views.login_page, name="home"),  # root URL redirects to login_page
    path("login/", views.login_page, name="login_page"),
    path("login/start", views.login_start, name="login_start"),
    path("login/wait/<uuid:session_id>/", views.login_wait, name="login_wait"),

    # Node to Django communication: session state (ready, authenticated) + scraping jobs that node sends back
    path("ingest/session/<uuid:session_id>/state", views.ingest_session_state, name="ingest_session_state"),
    path("ingest/unread/<uuid:session_id>/batch", views.ingest_unread_batch, name="ingest_unread_batch"),
    path("unread/", views.unread_page, name="unread_page"), # main page of unread messages
    # chats
    path("chats/<str:chat_id>/", views.chat_detail, name="chat_detail"),
    path("chats/<str:chat_id>/send/", views.send_suggestion, name="send_suggestion"),
]
