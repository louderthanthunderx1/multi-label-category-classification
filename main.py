from openai import OpenAI
import os, json, hashlib, math, re, time
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# === OpenAI Setup ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Config ===
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 10
EMBED_CACHE_PATH = "definition_embeddings.json"
MIN_CONFIDENCE = 0.6

# === Inquiry Type ===
inquiry_types = [
    # 1‒10  ── Billing & Account
    {"type": "Billing Issue",               "definition": "Customer questions billing errors, overcharges, or unclear charges."},
    {"type": "Refund Request",              "definition": "Customer wants a refund for unused service, plan, days, or billing errors."},
    {"type": "Late Payment Arrangement",    "definition": "Customer needs extra time or a payment plan to settle an overdue bill."},
    {"type": "Auto-Pay Setup",              "definition": "Customer wants to enroll a card or bank account for automatic payments."},
    {"type": "Auto-Pay Cancellation",       "definition": "Customer wants to remove their automatic-payment setting."},
    {"type": "Billing Cycle Change",        "definition": "Customer wants the statement or due date moved to a different day."},
    {"type": "Payment Method Update",       "definition": "Customer needs to add, delete, or edit a card/bank account on file."},
    {"type": "Balance Inquiry (Prepaid)",   "definition": "Customer asks how much credit or data remains on their prepaid account."},
    {"type": "Expired Credit Recovery",     "definition": "Customer tries to restore expired prepaid credit or rollover balance."},
    {"type": "Suspicious Charge Alert",     "definition": "Customer reports a charge they don’t recognize or suspect is fraudulent."},

    # 11‒20 ── Plans & Promotions
    {"type": "Plan Upgrade",                "definition": "Customer wants to move to a more expensive or higher-speed plan."},
    {"type": "Plan Downgrade",              "definition": "Customer wants to switch to a cheaper or lower-tier plan."},
    {"type": "Family Plan Add-Line",        "definition": "Customer wants to add a new line, SIM, or user to an existing family plan."},
    {"type": "Family Plan Remove-Line",     "definition": "Customer wants to remove a line or user from their family plan."},
    {"type": "Contract Renewal",            "definition": "Customer is ready to renew or extend an expiring service agreement."},
    {"type": "Upgrade Eligibility Inquiry", "definition": "Customer wants to know if they qualify for a discounted device or plan change."},
    {"type": "Loyalty Discount Inquiry",    "definition": "Customer asks about tenure-based discounts or perks."},
    {"type": "Promotional Offer Inquiry",   "definition": "Customer wants details or help redeeming a current promo or coupon."},
    {"type": "Student Discount Inquiry",    "definition": "Customer asks about plans or pricing tailored for students."},
    {"type": "Senior Plan Inquiry",         "definition": "Customer asks about age-based senior plans or rates."},

    # 21‒30 ── Technical Service
    {"type": "Technical Support",           "definition": "Customer reports a technical problem with a device, app, or network."},
    {"type": "Slow Internet Speed",         "definition": "Customer complains that internet/data throughput is lower than expected."},
    {"type": "Call Drop Issue",             "definition": "Customer experiences frequent dropped or failed voice calls."},
    {"type": "Coverage Complaint",          "definition": "Customer states that service is weak or unavailable in a specific area."},
    {"type": "Network Outage Report",       "definition": "Customer reports a widespread outage affecting multiple users or services."},
    {"type": "Hotspot Tethering Issue",     "definition": "Customer cannot use, enable, or limit mobile-hotspot data."},
    {"type": "VoLTE Issue",                 "definition": "Customer has trouble making HD Voice / VoLTE calls."},
    {"type": "Wi-Fi Calling Issue",         "definition": "Customer cannot place or receive Wi-Fi–calling calls."},
    {"type": "Spam Call Report",            "definition": "Customer reports unsolicited or fraudulent phone calls."},
    {"type": "Spam Text Block",             "definition": "Customer wants to stop or report unwanted SMS/MMS messages."},

    # 31‒40 ── Device & Hardware
    {"type": "Device Repair Request",       "definition": "Customer requests diagnostic or repair service for a malfunctioning device."},
    {"type": "Device Replacement",          "definition": "Customer seeks a replacement for a defective or damaged device."},
    {"type": "Lost or Stolen Device",       "definition": "Customer needs to suspend, locate, or replace a lost/stolen phone."},
    {"type": "Warranty Claim",              "definition": "Customer files a claim under the manufacturer or extended warranty."},
    {"type": "Insurance Claim",             "definition": "Customer files a claim through a device-protection or insurance plan."},
    {"type": "Device Unlock Request",       "definition": "Customer wants the carrier lock removed from their device."},
    {"type": "Device Trade-In Quote",       "definition": "Customer asks how much credit they will get for trading in a device."},
    {"type": "Device Trade-In Status",      "definition": "Customer wants an update on a previously submitted trade-in."},
    {"type": "Device Compatibility Check",  "definition": "Customer asks if a specific device will work on the network."},
    {"type": "Accessory Purchase Support",  "definition": "Customer needs help buying or troubleshooting accessories (chargers, earbuds, etc.)."},

    # 41‒50 ── SIM & Number
    {"type": "SIM Activation",              "definition": "Customer activates a new SIM or eSIM for service."},
    {"type": "SIM Replacement",             "definition": "Customer needs a new SIM due to loss, damage, or size change."},
    {"type": "eSIM Activation",             "definition": "Customer wants to download or transfer an eSIM profile."},
    {"type": "Number Port-In",              "definition": "Customer requests to bring a phone number from another carrier."},
    {"type": "Number Port-Out",             "definition": "Customer requests to move their number to another carrier."},
    {"type": "Static IP Request",           "definition": "Customer wants a fixed public IP address on their internet service."},
    {"type": "Public IP Assignment Issue",  "definition": "Customer reports problems obtaining or maintaining a public IP."},
    {"type": "International Roaming Activation", "definition": "Customer wants roaming enabled before traveling abroad."},
    {"type": "International Roaming Issue", "definition": "Customer reports service problems while roaming internationally."},
    {"type": "International Call Block",    "definition": "Customer wants to enable or disable outbound international calls."},

    # 51‒60 ── Messaging & Content
    {"type": "SMS Delivery Issue",          "definition": "Customer’s text messages fail to send or receive."},
    {"type": "MMS Issue",                   "definition": "Customer reports problems sending or receiving picture/video messages."},
    {"type": "Voicemail Issue",             "definition": "Customer cannot set up, access, or get notifications for voicemail."},
    {"type": "Visual Voicemail Issue",      "definition": "Customer’s visual-voicemail app is malfunctioning."},
    {"type": "Emergency Alert Issue",       "definition": "Customer is missing or over-receiving government/emergency alerts."},
    {"type": "Content Subscription Cancellation", "definition": "Customer wants to stop a third-party content or premium SMS subscription."},
    {"type": "Ringtone Purchase Refund",    "definition": "Customer wants a refund for a ringtone or content download."},
    {"type": "Streaming Quality Issue",     "definition": "Customer reports buffering or low resolution on carrier-bundled streaming."},
    {"type": "Parental Controls Setup",     "definition": "Customer wants to enable or adjust content filters or time limits."},
    {"type": "Usage Alert Setup",           "definition": "Customer wants email/SMS alerts for data, minutes, or text thresholds."},

    # 61‒70 ── Orders & Logistics
    {"type": "Shipping Status Inquiry",     "definition": "Customer asks where their device or accessory shipment is."},
    {"type": "Order Cancellation",          "definition": "Customer wants to cancel an online or phone order before it ships."},
    {"type": "Order Return",                "definition": "Customer needs to return a shipped item within the return window."},
    {"type": "Store Appointment Scheduling","definition": "Customer wants to book an in-store visit for sales or service."},
    {"type": "Technician Visit Scheduling", "definition": "Customer needs an on-site tech appointment for home internet/TV."},
    {"type": "Shipping Damage Report",      "definition": "Customer received an item damaged in transit and needs resolution."},
    {"type": "Back-Order Update",           "definition": "Customer asks when an out-of-stock item will ship."},
    {"type": "Trade-In Shipping Kit Request","definition": "Customer needs a prepaid kit to mail in a trade-in device."},
    {"type": "Gift Card Redemption",        "definition": "Customer needs help applying a promo or gift card to an order."},
    {"type": "Accessory Return",            "definition": "Customer wants to return or exchange a purchased accessory."},

    # 71‒80 ── Account Management
    {"type": "Password Reset",              "definition": "Customer cannot access the account due to forgotten credentials."},
    {"type": "Account Login Issue",         "definition": "Customer experiences errors or MFA problems signing in."},
    {"type": "Security Concern (Account Hacked)", "definition": "Customer suspects unauthorized access and wants account secured."},
    {"type": "Address Change",              "definition": "Customer must update their service or billing address."},
    {"type": "Contact Info Update",         "definition": "Customer wants to change phone, email, or notification preferences."},
    {"type": "Name Change",                 "definition": "Customer needs their legal or display name updated on the account."},
    {"type": "Data Privacy Request (GDPR)", "definition": "Customer requests data export, deletion, or restriction under privacy law."},
    {"type": "Subpoena / Legal Request",    "definition": "Law enforcement or legal team requests records or account holds."},
    {"type": "Fraud Investigation",         "definition": "Customer or internal team opens a case for suspected fraud."},
    {"type": "Regulatory Compliance Inquiry","definition": "Customer asks about carrier compliance with telecom regulations."},

    # 81‒90 ── Specialized / B2B
    {"type": "Corporate Account Support",   "definition": "Business customer needs help with multi-line enterprise services."},
    {"type": "M2M / IoT SIM Support",       "definition": "Customer needs assistance with machine-to-machine or IoT connectivity."},
    {"type": "Static IP Renewal",           "definition": "Business customer needs to renew or extend a static IP assignment."},
    {"type": "VPN Compatibility Inquiry",   "definition": "Customer asks if the network supports specific VPN traffic or protocols."},
    {"type": "Port Blocking Issue",         "definition": "Customer reports blocked ports affecting servers or applications."},
    {"type": "Public Safety Priority Access","definition": "First responder requests priority network access or re-provisioning."},
    {"type": "Corporate Discount Enrollment","definition": "Employee wants to add an employer-negotiated service discount."},
    {"type": "Static IP Removal",           "definition": "Customer wants to switch back to dynamic IP assignment."},
    {"type": "Dedicated APN Setup",         "definition": "Business customer requests a private or custom APN configuration."},
    {"type": "SLA Credit Request",          "definition": "Business customer claims SLA breach and requests service credit."},

    # 91‒100 ── Feedback & Misc
    {"type": "Complaint Escalation",        "definition": "Customer is dissatisfied and wants to speak with a supervisor or manager."},
    {"type": "Compliment / Praise",         "definition": "Customer provides positive feedback about service or staff."},
    {"type": "Survey Feedback",             "definition": "Customer replies to a satisfaction survey with detailed comments."},
    {"type": "Feature Request",             "definition": "Customer suggests a new product or service feature."},
    {"type": "Accessibility Support Request","definition": "Customer with disability needs accessibility accommodations or services."},
    {"type": "Deaf or Hard-of-Hearing Support","definition": "Customer needs TTY, RTT, or other hearing-assistive services."},
    {"type": "Military Discount Inquiry",   "definition": "Customer asks about active-duty, veteran, or military family pricing."},
    {"type": "Community Outreach Program Inquiry","definition": "Customer or organization wants info on carrier social-impact programs."},
    {"type": "Store Locator Inquiry",       "definition": "Customer asks where the nearest retail or service store is located."},
    {"type": "Service Cancellation",        "definition": "Customer wants to terminate service and possibly port out their number."}
]

# === Helper Functions ===
def _extract_json(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return match.group(0)

@lru_cache(maxsize=None)
def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return dot / (mag_a * mag_b + 1e-8)

# === Save/load embeddings ===
def save_definition_embeddings():
    data = []
    for item in inquiry_types:
        emb = get_embedding(item["definition"])
        data.append({"type": item["type"], "embedding": emb})
    with open(EMBED_CACHE_PATH, "w") as f:
        json.dump(data, f)

def load_definition_embeddings():
    if not os.path.exists(EMBED_CACHE_PATH):
        print("Saving definition embeddings...")
        save_definition_embeddings()
    with open(EMBED_CACHE_PATH) as f:
        data = json.load(f)
    return [item["embedding"] for item in data]

# === Load definition embeddings once
definition_embeddings = load_definition_embeddings()

def retrieve_definitions(message: str, k: int = TOP_K):
    q_vec = get_embedding(message)
    scored = [
        {"type": item["type"], "definition": item["definition"], "score": cosine(q_vec, d_emb)}
        for item, d_emb in zip(inquiry_types, definition_embeddings)
    ]
    max_score = max(s["score"] for s in scored)
    min_score = min(s["score"] for s in scored)
    for s in scored:
        s["normalized"] = (s["score"] - min_score) / (max_score - min_score + 1e-8)
    top = sorted(scored, key=lambda x: x["score"], reverse=True)[:k]

    print("Top-K retrieved labels:")
    for i, item in enumerate(top, start=1):
        print(f"{i:2d}. {item['type']:30}  score: {item['score']:.3f}")

    return top

# === Prompt Template ===
# FEW_SHOT = """
# Message: "I want to upgrade my data plan and increase my internet speed."
# Labels: {"Billing Issue": false, "Plan Upgrade": true, "Technical Support": false}

# Message: "I was charged twice this month and don't know why."
# Labels: {"Billing Issue": true, "Plan Upgrade": false, "Technical Support": false}
# """.strip()
FEW_SHOT = """
Example 1:
Message: "I want to upgrade my data plan and increase my internet speed."
Labels: {
  "Plan Upgrade": {
    "reason": "mentions 'upgrade my data plan and increase my internet speed'",
    "confidence": 0.98,
    "retrieved_rank": 1
  },
  "Slow Internet Speed": {
    "reason": "implied by 'increase my internet speed'",
    "confidence": 0.89,
    "retrieved_rank": 2
  },
  "Billing Issue": false,
  "Technical Support": false
}

Example 2:
Message: "I was charged twice this month and don't know why."
Labels: {
  "Billing Issue": {
    "reason": "mentions 'charged twice this month'",
    "confidence": 0.95,
    "retrieved_rank": 1
  },
  "Refund Request": {
    "reason": "likely wants a refund due to double charge",
    "confidence": 0.83,
    "retrieved_rank": 2
  },
  "Plan Upgrade": false,
  "Technical Support": false
}
""".strip()


def build_prompt(msg: str, cands: list[dict]) -> str:
    # lab_lines = ",\n  ".join(f'"{c["type"]}": true/false' for c in cands)
    lab_lines = ",\n  ".join(f'"{c["type"]}": true/false or explanation object' for c in cands)
    # return f"""{FEW_SHOT}

    # Message: "{msg}"
    # Labels (true/false): {{
    # {lab_lines}
    # }}

    # Return **only** a valid JSON object.
    # """

    return f"""
    You are a precise multi-label classifier. Do not invent labels or explanations.

    {FEW_SHOT}

    Now classify the following messages:
    Message: "{msg}"
    
    output format:
    Return a valid JSON object. Each key must be one of the given labels.

    Each key must map to either:
    - false
    - OR, if true, an object:
    {{
        "reason": "...short explanation or matched text from the message...",
        "confidence": <0 to 1>,
        "retrieved_rank": <1 to {TOP_K}>
    }}

    Only output the JSON object. Do not include any other text or comments.
    {{
        {lab_lines}
    }}
    """

# === Classify ===
def classify_message(message: str):

    print("Not in cache, calling OpenAI API...")
    candidates = retrieve_definitions(message)
    prompt = build_prompt(message, candidates)
    print(f"--- prompt ---\n {prompt}")
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,
        response_format={"type": "json_object"}
    )

    raw = resp.choices[0].message.content
    print(f"--- raw ---\n {raw}")
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError:
        labels = json.loads(_extract_json(raw))

    # Output with confidence + rank
    filtered_out = {}
    output = {}
    # for idx, item in enumerate(candidates):
    #     label = item["type"]
    #     conf = item['normalized']
    #     # if labels.get(label) is True:
    #     #     output[label] = {
    #     #         "confidence": round(item["normalized"], 3),
    #     #         "retrieved_rank": idx + 1
    #     #     }
    #     # if labels.get(label) is True and item["normalized"] >= MIN_CONFIDENCE:
    #     if labels.get(label) is True:
    #         if conf >= MIN_CONFIDENCE:
    #             output[label] = {
    #                 "confidence": round(conf, 3),
    #                 "retrieved_rank": idx + 1
    #             }
    #         else:
    #             filtered_out[label] = round(conf, 3)
    
    for idx, item in enumerate(candidates):
        label = item['type']
        raw = labels.get(label)

        if isinstance(raw, dict):
            conf = raw.get("confidence", item['normalized'])
            if conf >= MIN_CONFIDENCE:
                output[label] = {
                    "reason": raw.get("reason", ""),
                    "confidence": round(conf, 3),
                    "retrieved_rank": raw.get("retrieved_rank", idx + 1)
                }
            else:
                filtered_out[label] = round(conf, 3)

    if filtered_out:
        print("low-confidence labels:", filtered_out)

    return output

def pretty(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))

# === Run ===
if __name__ == "__main__":
    test_input = """
    Hi, I want to cancel my current family plan. Also, I’d like to port out two of the numbers to a prepaid plan, 
    and get a refund for the unused days if possible.
    Sure, I can help with that. Just to confirm — you want to cancel your family plan, g
    move two numbers to prepaid, and request a refund for the remaining days on your billing cycle?
    Exactly.
    """
    start = time.time()
    result = classify_message(test_input)
    pretty(result)
    print(f"Took {time.time() - start:.2f} seconds")