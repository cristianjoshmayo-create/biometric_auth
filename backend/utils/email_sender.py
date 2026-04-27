# backend/utils/email_sender.py
# Gmail SMTP sender for enrollment verification and anomaly alerts.

import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER     = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Biometric Authentication").strip()


def _send(to_email: str, subject: str, html_body: str, text_body: str) -> bool:
    if not SMTP_USER or not SMTP_PASSWORD:
        print("[email] SMTP_USER or SMTP_PASSWORD missing; skipping send")
        return False

    msg = MIMEMultipart("alternative")
    msg["From"]    = formataddr((SMTP_FROM_NAME, SMTP_USER))
    msg["To"]      = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx, timeout=15) as server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [to_email], msg.as_string())
        print(f"[email] sent to {to_email}: {subject}")
        return True
    except Exception as e:
        print(f"[email] send failed to {to_email}: {type(e).__name__}: {e}")
        return False


def send_anomaly_alert(to_email: str, event_type: str, details: dict) -> bool:
    """
    Sends a security alert email to the account owner when an authentication
    anomaly is detected.

    details dict keys (all optional):
      summary     – one-line plain-English description
      timestamp   – ISO 8601 string (UTC)
      local_time  – human-readable Asia/Manila time
      ip          – remote IP of the attempt
      user_agent  – browser / OS string
      scores      – dict of relevant scores (keystroke confidence, voice sim, etc.)
      reason      – technical reason (e.g. "dwell_mean z=6.2")
    """
    summary    = details.get("summary", event_type)
    ts_utc     = details.get("timestamp", "-")
    ts_local   = details.get("local_time", "-")
    ip         = details.get("ip", "-")
    ua         = details.get("user_agent", "-")
    reason     = details.get("reason", "")
    scores     = details.get("scores", {}) or {}

    scores_html = ""
    scores_text = ""
    if scores:
        rows = []
        text_rows = []
        for k, v in scores.items():
            rows.append(
                f'<tr><td style="color:#888;padding:4px 12px 4px 0;">{k}</td>'
                f'<td style="color:#eee;font-family:monospace;">{v}</td></tr>'
            )
            text_rows.append(f"  {k}: {v}")
        scores_html = ('<table style="margin-top:8px;">' + "".join(rows) + "</table>")
        scores_text = "\n".join(text_rows)

    reason_block_html = (
        f'<p style="color:#f59e0b;font-size:13px;">Reason: {reason}</p>' if reason else ""
    )
    reason_block_text = f"Reason: {reason}\n" if reason else ""

    subject = f"[Security Alert] {event_type} — Biometric Authentication"

    text = (
        f"Security alert on your Biometric Authentication account ({to_email}).\n\n"
        f"{summary}\n\n"
        f"When (UTC):   {ts_utc}\n"
        f"When (local): {ts_local}\n"
        f"IP:           {ip}\n"
        f"Browser/OS:   {ua}\n"
        f"{reason_block_text}"
        f"{'Scores:' + chr(10) + scores_text + chr(10) if scores_text else ''}\n"
        f"If this was you, you can ignore this email.\n"
        f"If this wasn't you, change your password immediately."
    )

    html = f"""\
<html><body style="font-family:Arial,sans-serif;background:#0b0b0f;padding:24px;color:#eee;">
  <div style="max-width:520px;margin:auto;background:#17171f;border-radius:12px;padding:28px;">
    <h2 style="color:#ef4444;margin-top:0;">⚠ Security Alert</h2>
    <p style="color:#ccc;line-height:1.5;">{summary}</p>
    {reason_block_html}
    <table style="color:#ccc;font-size:13px;margin-top:16px;border-collapse:collapse;">
      <tr><td style="color:#888;padding:4px 12px 4px 0;">When (UTC)</td>
          <td style="color:#eee;font-family:monospace;">{ts_utc}</td></tr>
      <tr><td style="color:#888;padding:4px 12px 4px 0;">When (local)</td>
          <td style="color:#eee;font-family:monospace;">{ts_local}</td></tr>
      <tr><td style="color:#888;padding:4px 12px 4px 0;">IP</td>
          <td style="color:#eee;font-family:monospace;">{ip}</td></tr>
      <tr><td style="color:#888;padding:4px 12px 4px 0;">Browser/OS</td>
          <td style="color:#eee;font-family:monospace;word-break:break-all;">{ua}</td></tr>
    </table>
    {scores_html}
    <hr style="border:none;border-top:1px solid #333;margin:20px 0;">
    <p style="color:#ccc;font-size:13px;line-height:1.5;">
      <strong style="color:#fff;">Was this you?</strong><br>
      If yes, you can safely ignore this email.<br>
      If no, change your password immediately — someone may be attempting to access your account.
    </p>
  </div>
</body></html>
"""
    return _send(to_email, subject, html, text)


def send_unlock_email(to_email: str, unlock_link: str) -> bool:
    subject = "Account Locked — Recovery Link Inside"
    text = (
        f"Your Biometric Authentication account was locked after 3 failed "
        f"security-question attempts.\n\n"
        f"If this was you, click the link below to unlock your account:\n\n"
        f"{unlock_link}\n\n"
        f"This link expires in 30 minutes and can only be used once.\n\n"
        f"If this was NOT you, someone may be trying to access your account. "
        f"Do not click the link — change your password as soon as possible."
    )
    html = f"""\
<html><body style="font-family:Arial,sans-serif;background:#0b0b0f;padding:24px;color:#eee;">
  <div style="max-width:520px;margin:auto;background:#17171f;border-radius:12px;padding:28px;">
    <h2 style="color:#ef4444;margin-top:0;">🔒 Account Locked</h2>
    <p style="color:#ccc;line-height:1.5;">
      Your Biometric Authentication account was locked after
      <strong>3 failed security-question attempts</strong>.
    </p>
    <p style="text-align:center;margin:28px 0;">
      <a href="{unlock_link}"
         style="background:#7c3aed;color:#fff;text-decoration:none;
                padding:12px 24px;border-radius:8px;font-weight:600;display:inline-block;">
        Unlock My Account
      </a>
    </p>
    <p style="color:#888;font-size:12px;">
      Or paste this link into your browser:<br>
      <span style="color:#a78bfa;word-break:break-all;">{unlock_link}</span>
    </p>
    <hr style="border:none;border-top:1px solid #333;margin:20px 0;">
    <p style="color:#ccc;font-size:13px;line-height:1.5;">
      <strong style="color:#fff;">Was this you?</strong><br>
      If yes, click the button above to restore access. Link expires in 30 minutes and is single-use.<br>
      If no, someone may be trying to access your account — do NOT click the link and change your password immediately.
    </p>
  </div>
</body></html>
"""
    return _send(to_email, subject, html, text)


def send_password_reset_email(to_email: str, reset_link: str) -> bool:
    subject = "Password Reset Requested — Biometric Authentication"
    text = (
        f"A password reset was requested for your Biometric Authentication account.\n\n"
        f"If this was you, click the link below to continue. You will be asked to "
        f"verify your identity via keystroke + voice biometrics before setting a new password:\n\n"
        f"{reset_link}\n\n"
        f"This link expires in 15 minutes and can only be used once.\n\n"
        f"If you did NOT request this reset, someone may be attempting to access your account. "
        f"Do not click the link. Your account is still protected — the attacker would also need "
        f"to pass your biometric challenge, which they cannot do."
    )
    html = f"""\
<html><body style="font-family:Arial,sans-serif;background:#0b0b0f;padding:24px;color:#eee;">
  <div style="max-width:520px;margin:auto;background:#17171f;border-radius:12px;padding:28px;">
    <h2 style="color:#f59e0b;margin-top:0;">🔑 Password Reset Requested</h2>
    <p style="color:#ccc;line-height:1.5;">
      Someone asked to reset the password on your Biometric Authentication account.
      To continue, you'll need to pass a biometric challenge (type + speak your passphrase).
    </p>
    <p style="text-align:center;margin:28px 0;">
      <a href="{reset_link}"
         style="background:#7c3aed;color:#fff;text-decoration:none;
                padding:12px 24px;border-radius:8px;font-weight:600;display:inline-block;">
        Continue Reset
      </a>
    </p>
    <p style="color:#888;font-size:12px;">
      Or paste this link into your browser:<br>
      <span style="color:#a78bfa;word-break:break-all;">{reset_link}</span>
    </p>
    <hr style="border:none;border-top:1px solid #333;margin:20px 0;">
    <p style="color:#ccc;font-size:13px;line-height:1.5;">
      <strong style="color:#fff;">Didn't request this?</strong><br>
      Do not click the link. Your account is still safe — the attacker would also need to
      pass your keystroke + voice biometric challenge, which they cannot do.
      Link expires in 15 minutes and is single-use.
    </p>
  </div>
</body></html>
"""
    return _send(to_email, subject, html, text)


def send_password_changed_email(to_email: str) -> bool:
    subject = "Your password was changed — Biometric Authentication"
    text = (
        f"The password on your Biometric Authentication account was just changed via the reset flow.\n\n"
        f"If this was you, you can ignore this email.\n\n"
        f"If this was NOT you, contact support immediately — someone may have compromised "
        f"both your email inbox and completed a biometric challenge on your account."
    )
    html = f"""\
<html><body style="font-family:Arial,sans-serif;background:#0b0b0f;padding:24px;color:#eee;">
  <div style="max-width:520px;margin:auto;background:#17171f;border-radius:12px;padding:28px;">
    <h2 style="color:#22c55e;margin-top:0;">✅ Password Changed</h2>
    <p style="color:#ccc;line-height:1.5;">
      The password on your Biometric Authentication account was just changed via the
      password-reset flow. The biometric challenge was passed successfully.
    </p>
    <hr style="border:none;border-top:1px solid #333;margin:20px 0;">
    <p style="color:#ccc;font-size:13px;line-height:1.5;">
      <strong style="color:#fff;">Was this you?</strong><br>
      If yes, you can safely ignore this email.<br>
      If no, contact support immediately — someone may have gained access to both your
      email inbox and completed a biometric challenge on your account.
    </p>
  </div>
</body></html>
"""
    return _send(to_email, subject, html, text)


def send_verification_email(to_email: str, verify_link: str) -> bool:
    subject = "Verify your email — Biometric Authentication"
    text = (
        f"Welcome to Biometric Authentication.\n\n"
        f"Click the link below to verify your email and continue enrollment:\n\n"
        f"{verify_link}\n\n"
        f"This link expires in 15 minutes.\n"
        f"If you did not request this, ignore this email."
    )
    html = f"""\
<html><body style="font-family:Arial,sans-serif;background:#0b0b0f;padding:24px;color:#eee;">
  <div style="max-width:480px;margin:auto;background:#17171f;border-radius:12px;padding:28px;">
    <h2 style="color:#a78bfa;margin-top:0;">Verify your email</h2>
    <p style="color:#ccc;line-height:1.5;">
      You're one click away from finishing your Biometric Authentication enrollment.
      Click the button below to verify this email address.
    </p>
    <p style="text-align:center;margin:28px 0;">
      <a href="{verify_link}"
         style="background:#7c3aed;color:#fff;text-decoration:none;
                padding:12px 24px;border-radius:8px;font-weight:600;display:inline-block;">
        Verify Email
      </a>
    </p>
    <p style="color:#888;font-size:12px;">
      Or paste this link into your browser:<br>
      <span style="color:#a78bfa;word-break:break-all;">{verify_link}</span>
    </p>
    <p style="color:#666;font-size:12px;margin-top:24px;">
      This link expires in 15 minutes. If you did not request this, you can safely ignore this email.
    </p>
  </div>
</body></html>
"""
    return _send(to_email, subject, html, text)
