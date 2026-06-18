"""
WhatsApp alert delivery via Meta Cloud API.
Supports: single farmer, cooperative batch, template messages.
"""

import os, requests, json
from typing import List


def send_whatsapp_alert(phone_numbers: List[str], message: str) -> List[dict]:
    """
    Send a WhatsApp text message to one or more farmers.
    
    Requirements:
      - WHATSAPP_TOKEN    : Meta permanent token (from Meta Business Manager)
      - WHATSAPP_PHONE_ID : Your WhatsApp Business phone number ID
    
    For production, use a pre-approved message template for outbound marketing.
    For 24-hr session window, free-form text is permitted.
    
    Returns list of {phone, success, status_code, message_id} dicts.
    """
    token    = os.getenv("WHATSAPP_TOKEN", "")
    phone_id = os.getenv("WHATSAPP_PHONE_ID", "")

    if not token or not phone_id:
        return [{"phone": p, "success": False,
                 "error": "WHATSAPP_TOKEN or WHATSAPP_PHONE_ID not set"} for p in phone_numbers]

    url     = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    results = []

    for phone in phone_numbers:
        # Normalise: strip spaces, ensure + prefix
        clean = phone.strip().replace(" ", "")
        if not clean.startswith("+"):
            clean = "+" + clean

        payload = {
            "messaging_product": "whatsapp",
            "to":                clean,
            "type":              "text",
            "text":              {"preview_url": False, "body": message},
        }

        try:
            resp = requests.post(url, headers=headers,
                                  data=json.dumps(payload), timeout=15)
            data = resp.json()
            if resp.status_code == 200:
                results.append({
                    "phone":      clean,
                    "success":    True,
                    "message_id": data.get("messages", [{}])[0].get("id", ""),
                })
            else:
                results.append({
                    "phone":   clean,
                    "success": False,
                    "error":   data.get("error", {}).get("message", str(resp.status_code)),
                })
        except requests.exceptions.RequestException as e:
            results.append({"phone": clean, "success": False, "error": str(e)})

    return results


def send_template_alert(phone: str, crop: str, price: float,
                        signal: str, mandi: str) -> dict:
    """
    Send a pre-approved WhatsApp template message.
    Template name: 'agri_price_alert'
    Parameters: {{1}} = crop, {{2}} = price, {{3}} = signal, {{4}} = mandi
    
    Register this template at: https://business.facebook.com/wa/manage/message-templates/
    """
    token    = os.getenv("WHATSAPP_TOKEN", "")
    phone_id = os.getenv("WHATSAPP_PHONE_ID", "")

    payload = {
        "messaging_product": "whatsapp",
        "to":   phone,
        "type": "template",
        "template": {
            "name":     "agri_price_alert",
            "language": {"code": "en"},
            "components": [{
                "type": "body",
                "parameters": [
                    {"type": "text", "text": crop},
                    {"type": "text", "text": f"₹{price:,.0f}/Quintal"},
                    {"type": "text", "text": signal},
                    {"type": "text", "text": mandi},
                ],
            }],
        },
    }

    try:
        resp = requests.post(
            f"https://graph.facebook.com/v18.0/{phone_id}/messages",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type":  "application/json"},
            data=json.dumps(payload),
            timeout=15,
        )
        return {"success": resp.status_code == 200, "data": resp.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}
