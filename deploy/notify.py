"""
Failure notification — Section 9.2's three-layer design.

UNTESTED for actual delivery (no real webhook/healthchecks URL in this
sandbox) — the request-building logic is tested with synthetic calls
against httpbin-style echo behavior is NOT done here either, since that
needs network access this sandbox doesn't have to arbitrary domains.
Verify against your real webhook URL before relying on this.
"""

import os
import requests


def send_webhook_alert(message: str, webhook_url: str = None) -> bool:
    """
    Layer 3 of Section 9.2 — the shared Slack/Discord webhook.
    Works with either: Slack and Discord both accept a JSON body with
    a "text"/"content" key respectively via simple incoming webhooks.
    """
    webhook_url = webhook_url or os.environ.get("ALERT_WEBHOOK_URL")
    if not webhook_url:
        print(f"[notify] no webhook configured, alert not sent: {message}")
        return False

    is_discord = "discord.com" in webhook_url
    payload = {"content": message} if is_discord else {"text": message}

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        # Deliberately don't raise here — a failed alert shouldn't crash
        # the job on top of whatever already went wrong. Print so it at
        # least shows up in Render's own logs.
        print(f"[notify] webhook send failed: {e}")
        return False


def ping_heartbeat(heartbeat_url: str = None, failed: bool = False) -> bool:
    """
    Layer 2 of Section 9.2 — heartbeat/dead-man's-switch monitor (e.g.
    healthchecks.io). Ping on success; healthchecks.io's convention is
    appending /fail to the same URL to explicitly signal a failure
    rather than just letting the ping go missing.
    """
    heartbeat_url = heartbeat_url or os.environ.get("HEARTBEAT_URL")
    if not heartbeat_url:
        print("[notify] no heartbeat URL configured, skipping ping")
        return False

    url = f"{heartbeat_url}/fail" if failed else heartbeat_url
    try:
        requests.get(url, timeout=10)
        return True
    except Exception as e:
        print(f"[notify] heartbeat ping failed: {e}")
        return False


def report_success(job_name: str, summary: str = ""):
    ping_heartbeat(failed=False)
    print(f"[{job_name}] success: {summary}")


def report_failure(job_name: str, error: str):
    message = f":rotating_light: **{job_name} failed**\n{error}"
    send_webhook_alert(message)
    ping_heartbeat(failed=True)
    print(f"[{job_name}] failure reported: {error}")


if __name__ == "__main__":
    print("Testing notification logic with no real webhook configured (expect graceful no-ops):")
    report_success("test_job", summary="42 games processed")
    report_failure("test_job", error="CFBD API returned 500")
