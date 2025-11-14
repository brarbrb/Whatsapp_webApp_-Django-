from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.contrib import messages
# from django.utils import timezone
from django.conf import settings
from datetime import datetime

import json
import requests

from .models import Session, ScrapeJob, UnreadSession


NODE_BASE_URL = getattr(settings, "NODE_BASE_URL", "http://localhost:3001")
REQUEST_TIMEOUT = (3.0, 6.0)  # (connect, read) seconds


# ───────────────────────────── Login & Session flow (simple) ─────────────────────────────

@require_GET
def login_page(request):
    # Just show login button, no auto-redirect to /unread/
    return render(request, "login.html")


@require_POST
def login_start(request):
    # Reuse an active session if exists, otherwise create a new one
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
        messages.warning(request, f"Could not start WhatsApp session. Try again. ({e})")
        return redirect("login_page")

    sess.node_session_id = data.get("session_id", "")
    sess.state = Session.State.PENDING
    # sess.last_state_change = timezone.now()
    sess.last_state_change = datetime.now()
    sess.last_error = ""
    sess.save()

    return redirect("login_wait", session_id=sess.id)


@require_GET
def login_wait(request, session_id):
    sess = get_object_or_404(Session, id=session_id)

    # If session is already ready → unread page
    if sess.state == Session.State.READY:
        return redirect("unread_page")

    # Try to get QR from Node
    qr = None
    if sess.node_session_id:
        try:
            r = requests.get(
                f"{NODE_BASE_URL}/node/session/{sess.node_session_id}/qr",
                timeout=REQUEST_TIMEOUT,
            )
            if r.ok:
                qr = r.json().get("qr")
        except Exception:
            pass

    # Show warning if auth failed or disconnected
    if sess.state in (Session.State.FAILED, Session.State.DISCONNECTED):
        messages.warning(
            request, "Authentication failed or disconnected. Please rescan QR."
        )

    # IMPORTANT: context key is 'qr' (like your old working version)
    return render(request, "login_wait.html", {"session": sess, "qr": qr})


@csrf_exempt
@require_POST
def ingest_session_state(request, session_id):
    # Very simple shared-secret check (same as your old version)
    shared = getattr(settings, "NODE_SHARED_SECRET", "")
    node_secret = request.headers.get("X-Node-Secret", "")

    if shared and node_secret != shared:
        return HttpResponseForbidden("bad node secret")

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
    sess.last_state_change = datetime.now() 
    sess.last_error = error
    if state in (Session.State.DISCONNECTED, Session.State.FAILED):
        sess.is_active = False
        sess.ended_at = datetime.now()
    sess.save()

    return JsonResponse({"ok": True})


# ─────────────────────── Unread scraping ingest endpoints ───────────────────────

def _check_node_secret(request):
    """
    Shared helper to validate X-Node-Secret header.
    """
    expected = getattr(settings, "NODE_SHARED_SECRET", "")
    got = request.headers.get("X-Node-Secret") or request.META.get(
        "HTTP_X_NODE_SECRET"
    )
    if not expected or got != expected:
        return False
    return True


@csrf_exempt
@require_POST
def ingest_unread_batch(request, session_id):
    """
    Node sends batches of unread messages here.

    URL:   /ingest/unread/<session_id>/batch
    Body:  { "job_id": "<uuid>", "messages": [ {...}, ... ] }
    Header: X-Node-Secret: <secret>
    """
    if not _check_node_secret(request):
        return HttpResponseForbidden("invalid secret")

    try:
        sess = Session.objects.get(id=session_id)
    except Session.DoesNotExist:
        return JsonResponse({"error": "session_not_found"}, status=404)

    try:
        payload = json.loads(request.body or b"{}")
    except ValueError:
        return HttpResponseBadRequest("invalid json")

    job_id = payload.get("job_id")
    messages = payload.get("messages") or []

    if not job_id:
        return HttpResponseBadRequest("missing job_id")

    # Use ScrapeJob.id as the external job_id for simplicity
    try:
        job = ScrapeJob.objects.get(id=job_id, session=sess)
        # If job exists and was queued, mark as running now
        if job.status == ScrapeJob.Status.QUEUED:
            job.status = ScrapeJob.Status.RUNNING
    except ScrapeJob.DoesNotExist:
        job = ScrapeJob.objects.create(
            id=job_id,
            session=sess,
            status=ScrapeJob.Status.RUNNING,
        )

    new_msgs = 0

    for msg in messages:
        wa_msg_id = msg.get("wa_msg_id")
        chat_id = msg.get("chat_id")
        sender = msg.get("sender")
        body = msg.get("body", "") or ""
        msg_ts = msg.get("msg_ts")

        if not wa_msg_id:
            continue

        # Convert epoch seconds to aware DateTime, if provided
        dt = None
        if msg_ts:
            try:
                dt = datetime.fromtimestamp(float(msg_ts))
            except (TypeError, ValueError, OSError):
                dt = None
        # if msg_ts:
        #     try:
        #         dt = timezone.datetime.fromtimestamp(
        #             float(msg_ts), tz=timezone.utc
        #         )
        #     except (TypeError, ValueError, OSError):
        #         dt = None

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

    if new_msgs:
        job.num_msgs = job.num_msgs + new_msgs
        # recompute distinct chat count for this session (cheap at small scale)
        job.num_chats = (
            UnreadSession.objects.filter(session=sess)
            .values("chat_id")
            .distinct()
            .count()
        )

    job.save()

    return JsonResponse({"ok": True, "job_id": str(job.id), "new_msgs": new_msgs})


@csrf_exempt
@require_POST
def ingest_scrape_state(request, session_id):
    """
    Node tells us when a scrape job is done or errored.

    URL:   /ingest/scrape/<session_id>/state
    Body:  {
              "job_id": "<uuid>",
              "status": "done" | "running" | "error",
              "num_chats": 5,
              "num_msgs": 42,
              "error": "..."
           }
    Header: X-Node-Secret: <secret>
    """
    if not _check_node_secret(request):
        return HttpResponseForbidden("invalid secret")

    try:
        sess = Session.objects.get(id=session_id)
    except Session.DoesNotExist:
        return JsonResponse({"error": "session_not_found"}, status=404)

    try:
        payload = json.loads(request.body or b"{}")
    except ValueError:
        return HttpResponseBadRequest("invalid json")

    job_id = payload.get("job_id")
    status = payload.get("status")
    num_chats = payload.get("num_chats")
    num_msgs = payload.get("num_msgs")
    error = payload.get("error", "") or ""

    if not job_id:
        return HttpResponseBadRequest("missing job_id")

    valid_statuses = {c[0] for c in ScrapeJob.Status.choices}
    if status not in valid_statuses:
        return HttpResponseBadRequest("invalid status")

    try:
        job = ScrapeJob.objects.get(id=job_id, session=sess)
    except ScrapeJob.DoesNotExist:
        job = ScrapeJob(id=job_id, session=sess)

    job.status = status

    if status in (ScrapeJob.Status.DONE, ScrapeJob.Status.ERROR):
        job.finished_at = datetime.now()

    if isinstance(num_chats, int):
        job.num_chats = num_chats
    if isinstance(num_msgs, int):
        job.num_msgs = num_msgs

    job.error = error
    job.save()

    return JsonResponse({"ok": True, "job_id": str(job.id)})


# ───────────────────────────── /unread/ page ─────────────────────────────


@require_GET
def unread_page(request):
    """
    Main page the user lands on after login.
    Shows either:
      - 'Scanning your unread chats…' while scrape is running, or
      - the list of unread chats/messages once done.
    """
    # Single-user assumption: take latest active session
    sess = (
        Session.objects.filter(is_active=True)
        .order_by("-started_at")
        .first()
    )

    if not sess or sess.state != Session.State.READY:
        messages.error(
            request,
            "No active WhatsApp session. Please log in again.",
        )
        return redirect("login_page")

    job = (
        ScrapeJob.objects.filter(session=sess)
        .order_by("-started_at")
        .first()
    )

    # If no job yet or still running → show 'scanning' view.
    if job is None or job.status in (
        ScrapeJob.Status.QUEUED,
        ScrapeJob.Status.RUNNING,
    ):
        context = {
            "session": sess,
            "job": job,
            "is_scanning": True,
            "unreads": [],
        }
        # For now we reuse chats.html; you can later create a dedicated 'unread_wait.html'
        response = render(request, "chats.html", context)
        # Optionally add auto-refresh header. You can also add <meta refresh> in the template using is_scanning.
        response["Refresh"] = "3"  # refresh every 3 seconds
        return response

    # If error → show message and send back to login (or a nicer error page)
    if job.status == ScrapeJob.Status.ERROR:
        messages.error(
            request,
            "Could not fetch unread chats. Please try again later.",
        )
        return redirect("login_page")

    # DONE: load unread messages for this session
    unreads = (
        UnreadSession.objects.filter(session=sess)
        .order_by("chat_id", "msg_ts", "scraped_at")
    )

    context = {
        "session": sess,
        "job": job,
        "is_scanning": False,
        "unreads": unreads,
    }
    # For now we reuse chats.html; later you can customise it to show `unreads`.
    return render(request, "chats.html", context)
