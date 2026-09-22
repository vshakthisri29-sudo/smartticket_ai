import os
import json
import random
from datetime import datetime, timedelta
import pandas as pd
from config import DATA_DIR, TICKETS_DATA_PATH, DATASET_METADATA_PATH, CATEGORIES, PRIORITIES, EMOTIONS

def generate_dataset(num_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    random.seed(random_seed)
    
    # Core complaint templates with variety and natural language variations
    templates = {
        "Payment": [
            "My card was charged but my order was cancelled immediately.",
            "I was charged twice for the exact same purchase on transaction #{tx}.",
            "The payment failed on checkout but the amount of ${amt} was deducted from my bank balance.",
            "The transaction shows successful in my banking app but my order confirmation is missing.",
            "I noticed an unauthorized charge on my debit statement from your service.",
            "Double billing on my monthly invoice. Please reverse the duplicate charge.",
            "Payment went through gateway #{tx} but website threw a 500 error and didn't generate an invoice.",
            "Funds debited twice while purchasing online. Bank says money reached merchant.",
            "Card was charged three times during checkout timeout.",
            "Why did you charge my card when the subscription was cancelled last month?",
            "Payment verification error during checkout, yet money was withdrawn.",
            "I got an alert from my bank for a charge I never authorized on your site."
        ],
        "Delivery": [
            "My package has not arrived even though the delivery date passed three days ago.",
            "The tracking status has not changed for five days and still shows pending pickup.",
            "My package was marked delivered yesterday but I checked everywhere and did not receive it.",
            "Courier left the parcel in the rain and the contents are ruined.",
            "The shipment tracking number #{tx} says invalid on the carrier website.",
            "Order was delivered to the wrong address in a completely different neighborhood.",
            "Package was delayed with no updated delivery estimate provided.",
            "The delivery driver did not ring the bell and claimed delivery attempt failed.",
            "Shipment is stuck at customs sorting facility for over a week.",
            "Expected courier delivery yesterday morning, still no package or dispatch notification.",
            "My delivery address was changed without my permission.",
            "Tracking shows returned to sender because driver could not find entrance."
        ],
        "Account": [
            "I cannot reset my password because the verification email never arrives in my inbox.",
            "My account is locked and I cannot log in to access my active services.",
            "I received a login alert from an IP address and device I do not recognize.",
            "Two-factor authentication code is not sending to my registered mobile phone.",
            "Someone changed my account profile email and I have been locked out.",
            "Need to update my business email on file as the old company domain expired.",
            "Account shows suspended due to suspicious activity, please verify my identity.",
            "Cannot bypass the captcha verification screen on the login page.",
            "Session expires immediately after signing in, forcing a loop back to login.",
            "Please delete all my personal data and close this account under privacy regulations.",
            "Unable to modify my primary contact number in user security settings.",
            "Locked out of administrator dashboard after attempting password update."
        ],
        "Technical": [
            "The application crashes with an unhandled exception when I try to upload a document.",
            "The checkout page is completely blank and not loading any payment scripts.",
            "The website becomes completely unresponsive and freezes after user login.",
            "Getting a 502 Bad Gateway error whenever I try to access the customer reporting page.",
            "The mobile app on Android freezes and closes abruptly when searching products.",
            "API endpoint returns 403 forbidden error on authenticated requests.",
            "Search filter dropdown does not function and breaks the UI layout.",
            "Browser console reports syntax errors in main bundle javascript file.",
            "File export to CSV results in a corrupt 0-byte file download.",
            "Dark mode toggle breaks the contrast making text completely illegible.",
            "WebSocket connection constantly disconnects every 30 seconds.",
            "Unable to submit ticket form due to client-side validation bug."
        ],
        "Refund": [
            "I requested a refund ten days ago but have not received it in my bank account.",
            "The refund amount processed is incorrect and short by ${amt}.",
            "My refund has been pending approval for two weeks with no status update.",
            "I returned the item via pre-paid label last week, where is my refund credit?",
            "Customer agent promised a full refund in 48 hours but nothing has been credited.",
            "Please cancel my order #{tx} and return my money immediately as promised.",
            "Refund was supposedly sent to an expired card instead of my active bank account.",
            "Returned product was received by warehouse yesterday, awaiting reimbursement notification.",
            "I was charged a restocking fee that was not disclosed, I want a full refund.",
            "Invoice shows credit memo issued, but funds never settled in PayPal.",
            "Dispute regarding refund timeline, need money back as services were not rendered.",
            "Still waiting for reimbursement after returned goods were inspected and approved."
        ],
        "Product": [
            "The product I received is severely damaged with a cracked outer casing.",
            "The product does not match the description and specs listed on the storefront.",
            "An important power adapter accessory is missing from the sealed box.",
            "The device will not power on even after charging overnight with the supplied cable.",
            "Received the completely wrong color and model variation than what I ordered.",
            "Item broke on first day of normal usage due to defective plastic assembly.",
            "The user manual and software license key are missing from the package.",
            "Appliance emits a burning smell when plugged in, dangerous hazard.",
            "Product dimensions are much smaller than advertised in specifications table.",
            "Defective screen has dead pixels and flickering backlight.",
            "Received counterfeit or substandard replacement instead of OEM original.",
            "Garment sizing is completely inconsistent with the sizing chart."
        ]
    }
    
    # Emotion prefixes and phrasing
    emotion_phrases = {
        "Angry": [
            "This is totally unacceptable!",
            "I am furious with this terrible service.",
            "This is outrageous and completely unprofessional!",
            "I have been treated disrespectfully and want an answer now!",
            "Fix this immediately or I am filing a formal complaint!",
            "I demand to speak to a supervisor right away!"
        ],
        "Frustrated": [
            "I have contacted support multiple times with zero progress.",
            "This is extremely frustrating and wasting my work hours.",
            "I am tired of getting automated responses that do not help.",
            "Why is it so difficult to get a straight answer?",
            "Still waiting after being bounced between three different agents.",
            "This ongoing headache has persisted for over a week."
        ],
        "Worried": [
            "I am very concerned about the security of my money and account.",
            "I am worried that my important order won't make it in time for the event.",
            "Please help urgently, I am stressed about this unexpected charge.",
            "I am anxious about whether my identity has been compromised.",
            "Could someone please clarify, I am really uneasy about this.",
            "Need reassurance that this will be handled before tomorrow morning."
        ],
        "Neutral": [
            "Hello, could you please look into this issue for me?",
            "Inquiring regarding the status of the following item.",
            "Please assist with the request detailed below.",
            "Kindly advise on the standard procedure for this matter.",
            "Opening a ticket to report an unexpected issue.",
            "Following up on the regular update for this ticket."
        ],
        "Satisfied": [
            "I appreciate your team's quick assistance so far.",
            "Thanks for the prompt help on previous tickets, just have a minor follow-up.",
            "Generally very satisfied with your platform, just noticed this small issue.",
            "Thank you for looking into this minor detail.",
            "Great support team, hope you can help with this quick query.",
            "Appreciate the support and timely guidance."
        ]
    }
    
    # Urgency suffixes that impact Priority
    urgency_suffixes = {
        "Critical": [
            " This is a critical business blocker and halting our entire operations.",
            " We are losing revenue every minute this is down, urgent intervention required!",
            " Immediate escalation required, legal notice will follow if unresolved today.",
            " This is an emergency security breach affecting thousands of customers!"
        ],
        "High": [
            " Need this resolved ASAP as deadline is today.",
            " Please expedite this urgently as I cannot proceed without it.",
            " Please prioritize this ticket as it is time-sensitive.",
            " Urgent attention needed to avoid further escalations."
        ],
        "Medium": [
            " Please resolve this as soon as convenient.",
            " Looking forward to hearing back within normal business hours.",
            " A resolution within the next 24 to 48 hours would be appreciated.",
            " Please update me when someone has reviewed this."
        ],
        "Low": [
            " No rush on this, whenever your team has bandwidth.",
            " Low priority question, just wanted to check for future reference.",
            " Whenever possible, thanks.",
            " Just submitting feedback for your roadmap, take your time."
        ]
    }
    
    # Realistic noise/typo generator
    common_typos = {
        "account": "acount",
        "cancelled": "cancled",
        "charged": "chagred",
        "received": "recieved",
        "urgent": "urgnt",
        "delivery": "delivry",
        "refund": "refnd",
        "website": "websit"
    }

    rows = []
    base_time = datetime(2026, 1, 15, 9, 0)
    
    for i in range(num_samples):
        ticket_id = f"TICK-{i+1:05d}"
        customer_id = f"CUST-{random.randint(1001, 9999)}"
        category = random.choice(CATEGORIES)
        base_text = random.choice(templates[category])
        
        # Format variables in template
        base_text = base_text.replace("{tx}", str(random.randint(100000, 999999)))
        base_text = base_text.replace("{amt}", str(random.randint(25, 950)))
        
        # Decide Emotion (stratified weights)
        emotion = random.choices(
            ["Neutral", "Frustrated", "Angry", "Worried", "Satisfied"],
            weights=[30, 30, 20, 15, 5]
        )[0]
        
        # Decide Priority based on Category and natural urgency logic
        if category in ["Payment", "Technical"]:
            priority = random.choices(["Low", "Medium", "High", "Critical"], weights=[10, 25, 45, 20])[0]
        elif category in ["Refund", "Account"]:
            priority = random.choices(["Low", "Medium", "High", "Critical"], weights=[15, 35, 35, 15])[0]
        else:
            priority = random.choices(["Low", "Medium", "High", "Critical"], weights=[30, 45, 20, 5])[0]
            
        # Combine emotion phrase, base complaint, and urgency suffix
        emotion_prefix = random.choice(emotion_phrases[emotion])
        urgency_suffix = random.choice(urgency_suffixes[priority]) if random.random() < 0.75 else ""
        
        # Construct full ticket text
        ticket_text = f"{emotion_prefix} {base_text}{urgency_suffix}"
        
        # Add realistic noise/typo in 15% of tickets
        if random.random() < 0.15:
            for word, typo in common_typos.items():
                if word in ticket_text and random.random() < 0.5:
                    ticket_text = ticket_text.replace(word, typo, 1)
                    break
        
        # Created timestamp
        created_at = (base_time + timedelta(minutes=random.randint(10, 15000))).strftime("%Y-%m-%d %H:%M:%S")
        
        # Resolution status
        resolution_status = random.choices(
            ["Open", "Under Review", "In Progress", "Resolved", "Escalated"],
            weights=[35, 20, 20, 20, 5]
        )[0]
        
        # Previous ticket id for repeat simulation in ~15%
        previous_ticket_id = f"TICK-{random.randint(1, max(1, i)):05d}" if (i > 10 and random.random() < 0.15) else ""
        
        rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "ticket_text": ticket_text,
            "category": category,
            "priority": priority,
            "emotion": emotion,
            "created_at": created_at,
            "resolution_status": resolution_status,
            "previous_ticket_id": previous_ticket_id
        })
    
    df = pd.DataFrame(rows)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(TICKETS_DATA_PATH, index=False)
    
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_tickets": len(df),
        "random_seed": random_seed,
        "categories": df["category"].value_counts().to_dict(),
        "priorities": df["priority"].value_counts().to_dict(),
        "emotions": df["emotion"].value_counts().to_dict(),
        "resolution_statuses": df["resolution_status"].value_counts().to_dict(),
        "features": list(df.columns)
    }
    
    with open(DATASET_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Generated {len(df)} realistic tickets saved to {TICKETS_DATA_PATH}")
    return df

if __name__ == "__main__":
    generate_dataset()
