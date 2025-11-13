# whatsapp/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET, require_POST
from django.contrib import messages
from django.utils import timezone
from django.conf import settings
import requests

from .models import Session


NODE_BASE_URL = getattr(settings, "NODE_BASE_URL", "http://localhost:3001")
REQUEST_TIMEOUT = (3.0, 6.0)


@require_GET
def login_page(request):
    return render(request, "login.html")


@require_POST
def login_start(request):
    # reuse or create new session
    sess = Session.objects.filter(is_active=True).order_by("-started_at").first()
    if not sess:
        sess = Session.objects.create(is_active=True)

    # call node.js to create/ensure whatsapp-web.js session
    try:
        r = requests.post(f"{NODE_BASE_URL}/node/session",
                          json={"session_hint": str(sess.id)},
                          timeout=REQUEST_TIMEOUT)
        data = r.json()
        sess.node_session_id = data.get("session_id", "")
        sess.state = Session.State.PENDING
        sess.last_state_change = timezone.now()
        sess.last_error = ""
        sess.save()
    except Exception:
        messages.warning(request, "Could not start WhatsApp session. Try again.")
        return redirect("login_page")

    return redirect("login_wait", session_id=sess.id)


@require_GET
def login_wait(request, session_id):
    sess = get_object_or_404(Session, id=session_id)

    # if user is already authenticated, go to next page
    if sess.state == Session.State.READY:
        return redirect("unread_page")  # implement later

    # try to get QR from Node
    qr = None
    if sess.node_session_id:
        try:
            r = requests.get(f"{NODE_BASE_URL}/node/session/{sess.node_session_id}/qr",
                             timeout=REQUEST_TIMEOUT)
            if r.ok:
                qr = r.json().get("qr")
        except Exception:
            pass

    # show yellow message if auth failed or disconnected
    if sess.state in (Session.State.FAILED, Session.State.DISCONNECTED):
        messages.warning(request, "Authentication failed or disconnected. Please rescan QR.")

    return render(request, "login_wait.html", {"session": sess, "qr": qr})


@require_POST
def ingest_session_state(request, session_id):
    # Node calls this to update auth state
    import json
    from django.http import JsonResponse, HttpResponseBadRequest

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return HttpResponseBadRequest("invalid json")

    state = payload.get("state")
    error = payload.get("error", "")

    sess = Session.objects.filter(id=session_id).first()
    if not sess:
        return JsonResponse({"error": "session_not_found"}, status=404)

    valid_states = {c[0] for c in Session.State.choices}
    if state not in valid_states:
        return HttpResponseBadRequest("invalid state")

    sess.state = state
    sess.last_state_change = timezone.now()
    sess.last_error = error
    if state in (Session.State.DISCONNECTED, Session.State.FAILED):
        sess.is_active = False
        sess.ended_at = timezone.now()
    sess.save()

    return JsonResponse({"ok": True})