from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.contrib import messages
from django.conf import settings
from django.apps import apps

from datetime import datetime

import json
import requests

from .models import Session, UnreadSession
from django.db.models import Count, Max 
from .rag_pipe import WhatsAppRAG  # the pipline we showed in ipynb converted to a class

NODE_BASE_URL = getattr(settings, "NODE_BASE_URL", "http://localhost:3001") #acces what is in settings.py or set default 
REQUEST_TIMEOUT = (3.0, 6.0)  # used for requests to Node server (connect timeout, read timeout) in seconds


# _______Some login view, very basic!_______
#these decorators ensure that the views only respond to specific HTTP methods (GET or POST).
@require_GET
def login_page(request):
    # Just show login button
    return render(request, "login.html")


@require_POST
def login_start(request):
    """ Starts a WhatsApp session by requesting Node to create one.
        redirects to login_wait page on success.
        Think like initializing a session on the Node side.
    """
    # Read an active session from db if exists, otherwise create a new one
    sess = Session.objects.filter(is_active=True).order_by("-started_at").first()
    if not sess:
        sess = Session.objects.create(is_active=True)
    try:
        r = requests.post(
            f"{NODE_BASE_URL}/node/session",
            json={"session_hint": str(sess.id)},
            timeout=REQUEST_TIMEOUT,
        )
        data = r.json()
    except Exception as e:
        # mostly will be connection errors / timeouts
        messages.warning(request, f"Could not start WhatsApp session. Try again. ({e})")
        return redirect("login_page")

    # Update the db
    sess.node_session_id = data.get("session_id", "")
    sess.state = Session.State.PENDING # waiting for QR scan that node generate ans send to us
    sess.last_state_change = datetime.now()
    sess.last_error = ""
    sess.save()
    return redirect("login_wait", session_id=sess.id)


@require_GET
def login_wait(request, session_id):
    """ Shows the 'waiting for QR scan' page.
        Fetches QR code from Node server to display.
        keeps refresghing until qr code is scanned and user passes authentication: state becomes READY.
        Note: that the Node server will notify us of state changes via /ingest/session/<session_id>/state endpoint.
    """
    # here we fetch the Session fromdb with URL param and check its state 
    sess = get_object_or_404(Session, id=session_id)
    # if that session is READY, user has authenticated succeqssfully
    if sess.state == Session.State.READY:
        return redirect("unread_page")

    qr = None
    if sess.node_session_id: # just extra caution
        try:
            r = requests.get(
                f"{NODE_BASE_URL}/node/session/{sess.node_session_id}/qr",
                timeout=REQUEST_TIMEOUT,
            )
            if r.ok:
                # parsing JSON response to get QR code to display it to client
                qr = r.json().get("qr")
        except Exception:
            pass

    # Show warning if auth failed or disconnected
    if sess.state in (Session.State.FAILED, Session.State.DISCONNECTED):
        messages.warning(
            request, "Authentication failed or disconnected. Please rescan QR."
        )
    # we keep rendering the page til authentication passed
    return render(request, "login_wait.html", {"session": sess, "qr": qr})




# ENDPOINTS FOR NODE TO INGEST SCRAPED DATA
def _check_node_secret(request):
    """
    Small helper to validate X-Node-Secret header.
    """
    expected = getattr(settings, "NODE_SHARED_SECRET", "")
    got = request.headers.get("X-Node-Secret") or request.META.get(
        "HTTP_X_NODE_SECRET"
    )
    if not expected or got != expected:
        return False
    return True


# the csrf_exempt decorator is used to disable CSRF protection for this view
# allowing Node server to post data without needing a CSRF token, which is required by Django.
@csrf_exempt
@require_POST
def ingest_session_state(request, session_id):
    """
    Node tells us when session state changes.
    through this endpoint: /ingest/session/<session_id>/state
    """
    # Very simple shared-secret check to avoid randoms hitting this endpoint
    # shared = getattr(settings, "NODE_SHARED_SECRET", "")
    # node_secret = request.headers.get("X-Node-Secret", "")
    if not _check_node_secret(request):
        return HttpResponseForbidden("incorrect secret")
    
    try: # parsing POST body to get state 
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return HttpResponseBadRequest("invalid json")

    state = payload.get("state")
    error = payload.get("error", "")
    sess = Session.objects.filter(id=session_id).first()
    if not sess:
        return JsonResponse({"error": "session_not_found"}, status=404) # convention (from SE in ML course)

    # valid_states = {c[0] for c in Session.State.choices}
    # if state not in valid_states:
    #     return HttpResponseBadRequest("invalid state")

    sess.state = state
    sess.last_state_change = datetime.now() 
    sess.last_error = error
    if state in (Session.State.DISCONNECTED, Session.State.FAILED):
        sess.is_active = False
        sess.ended_at = datetime.now()
    sess.save()

    return JsonResponse({"ok": True})


@csrf_exempt 
@require_POST
def ingest_unread_batch(request, session_id):
    """
    Node sends batches of unread messages here with POST.
    URL:   /ingest/unread/<session_id>/batch
    Body:  { "job_id": "<uuid>", "messages": [ {...}, ... ] }
    """
    if not _check_node_secret(request):
        return HttpResponseForbidden("incorrect secret")
    sess = Session.objects.filter(id=session_id).first()
    if not sess:
        return JsonResponse({"error": "session_not_found"}, status=404)
    try:
        payload = json.loads(request.body or b"{}")
    except ValueError:
        return HttpResponseBadRequest("invalid json of unread batch")

    messages = payload.get("messages") or []
    new_msgs = 0
    # Store unread messages (deduplicated by wa_msg_id)
    for msg in messages:
        wa_msg_id = msg.get("wa_msg_id")
        chat_id = msg.get("chat_id")
        sender = msg.get("sender")
        # print("sender:", sender) # Debugging line
        body = msg.get("body", "") or ""
        msg_ts = msg.get("msg_ts")
        dt = datetime.fromtimestamp(float(msg_ts))
        obj, created = UnreadSession.objects.get_or_create(
            session=sess,
            wa_msg_id=wa_msg_id,
            defaults={
                "chat_id": chat_id or "",
                "sender": sender or "",
                "body": body,
                "msg_ts": dt,
            },
        )
        if created:
            new_msgs += 1
    return JsonResponse({"ok": True, "new_msgs": new_msgs})


@require_GET
def unread_page(request):
    """
    Main page the user lands on after login.
    Shows the list of unread chats with number of unread messages.
    Note that it assumes single active user/session (for production needs to be updated!).
    """
    # take latest active, READY session (since we assume single-user for now)
    sess = (
        Session.objects.filter(is_active=True, state=Session.State.READY)
        .order_by("-started_at")
        .first()
    )
    if not sess:
        return JsonResponse({"error": "session_not_found"}, status=404)

    # Group unread messages by chat_id, count messages, get latest timestamp
    chat_summaries = (
        UnreadSession.objects.filter(session=sess)
        .values("chat_id").annotate(
            num_unread=Count("id"),
            last_msg_ts=Max("msg_ts"),).order_by("-last_msg_ts")  # newest chats first
    )
    unreads = (
        UnreadSession.objects.filter(session=sess)
        .order_by("chat_id", "msg_ts", "scraped_at")
    )
    context = {
        "session": sess,
        "chat_summaries": chat_summaries,
        "unreads": unreads,
    }
    return render(request, "chats.html", context)



# _____scraping the messages for specific chat___
def fetch_chat_messages_from_node(session, chat_id):
    """
    Asks Node for messages of a specific chat.
    Returns a list of dicts with keys:
      wa_msg_id, sender, body, from_me, msg_ts
    """
    if not session or not session.node_session_id:
        return []
    try:
        r = requests.get(
            f"{NODE_BASE_URL}/node/session/{session.node_session_id}/chat/{chat_id}/messages",
            timeout=REQUEST_TIMEOUT,
        )
        if not r.ok:
            return []
        data = r.json()
        return data.get("messages", [])
    except Exception as e:
        print("Error fetching chat messages from Node.",e)
        return []


def chat_detail(request, chat_id):
    """
    Show all messages for a given chat in the active READY session.
    Also contains:
      - LLM suggestion (if generated)
      - Prompt form to refine the suggestion
    """

    # Find current active session
    sess = (
        Session.objects.filter(is_active=True, state=Session.State.READY)
        .order_by("-started_at")
        .first()
    )
    if not sess:
        return JsonResponse({"error": "session_not_found"}, status=404)

    # fetching messages for this chat from Node
    # messages_list = fetch_chat_messages_from_node(sess, chat_id)
    messages_list = sorted(
    fetch_chat_messages_from_node(sess, chat_id),
    key=lambda m: m.get("msg_ts", 0),
    reverse=True  # NEW → sort newest first
    )
    # print("Last message in chat:", messages_list[-1] if messages_list else "no messages")
    suggestion_text = None

    rag = apps.get_app_config("whatsapp").rag
    if rag:
        print("RAG initialized successfully.")

    # query = "i need help with my students, did you taught them already the embeddings ppt?"
    query = messages_list[0]["body"] if messages_list else ""
    print(messages_list)
    prompt_text = ""
    print("Generating reply for prompt:")
    print(query)
    if request.method == "POST":
        # User clicked "Send to model" – generate suggestion from LLM later
        prompt_text = request.POST.get("prompt", "").strip()
        if prompt_text:
            reply = rag.generate_reply(query, instructions=prompt_text)
            suggestion_text = reply
            print("Generated reply:")
            print(reply)
        else:
            messages.info(request, "Please enter a prompt to guide the suggestion.")
            print("Generated reply:")
            print(reply)
    else:
        reply = rag.generate_reply(query, instructions="")
        suggestion_text = reply

    context = {
        "session": sess,
        "chat_id": chat_id,
        "messages_list": messages_list,
        "suggestion_text": suggestion_text,
        "prompt_text": prompt_text,
    }
    return render(request, "chat_detail.html", context)

@require_POST
def send_suggestion(request, chat_id):
    """
    User is satisfied with the LLM suggestion and wants to send it via WhatsApp.
    Django takes the suggestion text and asks Node to send the message.
    """
    sess = (
        Session.objects.filter(is_active=True, state=Session.State.READY)
        .order_by("-started_at")
        .first()
    )
    if not sess:
        return JsonResponse({"error": "session_not_found"}, status=404)
    
    suggestion_text = request.POST.get("suggestion", "").strip()
    if not suggestion_text:
        messages.warning(request, "Suggestion text is empty.")
        return redirect("chat_detail", chat_id=chat_id)

    try:
        r = requests.post(
            f"{NODE_BASE_URL}/node/session/{sess.node_session_id}/chat/{chat_id}/send",
            json={"body": suggestion_text},
            timeout=REQUEST_TIMEOUT,
        )
        if not r.ok:
            messages.error(request, f"Failed to send message (Node HTTP {r.status_code}).")
        else:
            messages.success(request, "Suggestion sent to WhatsApp 🎉")
    except Exception as e:
        messages.error(request, f"Failed to send message: {e}")

    return redirect("chat_detail", chat_id=chat_id)