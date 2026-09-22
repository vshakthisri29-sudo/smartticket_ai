import re

FINANCIAL_KEYWORDS = ["charged", "charge", "card", "bank", "deducted", "refund", "billing", "unauthorized", "fee", "money", "dollars", "transaction"]
URGENCY_KEYWORDS = ["urgent", "urgently", "asap", "emergency", "immediately", "critical", "blocker", "lawsuit", "police", "legal"]
SECURITY_KEYWORDS = ["hacked", "security", "locked", "stolen", "unauthorized", "suspicious", "breach", "password", "compromised"]

def calculate_risk(prediction_output: dict, raw_text: str = "") -> dict:
    """
    Transparent rule-based risk assessment engine powered by ML model outputs.
    
    Formula:
    Base Risk is derived from ML Priority & Confidence, supplemented by Category
    exposure, Emotion severity, Repeat Complaint probability, and text-level impact markers.
    """
    category = prediction_output.get("category", "General")
    priority = prediction_output.get("priority", "Medium")
    priority_conf = prediction_output.get("priority_confidence", 0.5)
    emotion = prediction_output.get("emotion", "Neutral")
    similarity_score = prediction_output.get("similarity_score", 0.0)
    is_repeat = prediction_output.get("is_repeat", False)
    
    risk_score = 0
    factors = []
    
    # 1. ML Priority Weighting (Up to 40 pts)
    if priority == "Critical":
        pts = int(35 + (priority_conf * 5))
        risk_score += pts
        factors.append(f"+{pts} pts: ML predicted Critical priority (confidence: {priority_conf:.0%})")
    elif priority == "High":
        pts = int(25 + (priority_conf * 5))
        risk_score += pts
        factors.append(f"+{pts} pts: ML predicted High priority (confidence: {priority_conf:.0%})")
    elif priority == "Medium":
        pts = 15
        risk_score += pts
        factors.append(f"+{pts} pts: ML predicted Medium priority")
    else: # Low
        pts = 5
        risk_score += pts
        factors.append(f"+{pts} pts: ML predicted Low priority")
        
    # 2. Category Sensitivity (Up to 20 pts)
    if category in ["Payment", "Refund"]:
        risk_score += 15
        factors.append("+15 pts: Financial transaction category exposure")
    elif category in ["Account"]:
        risk_score += 12
        factors.append("+12 pts: Account authentication & security domain")
    elif category in ["Technical"]:
        risk_score += 10
        factors.append("+10 pts: System / platform technical availability issue")
        
    # 3. Customer Emotion Impact (Up to 15 pts)
    if emotion == "Angry":
        risk_score += 15
        factors.append("+15 pts: Severe negative customer sentiment (Angry)")
    elif emotion == "Frustrated":
        risk_score += 10
        factors.append("+10 pts: Customer friction detected (Frustrated)")
    elif emotion == "Worried":
        risk_score += 8
        factors.append("+8 pts: Customer vulnerability / concern (Worried)")
        
    # 4. Repeat Complaint Detection (Up to 15 pts)
    if is_repeat or similarity_score >= 0.70:
        pts = int(10 + (similarity_score * 5))
        risk_score += pts
        factors.append(f"+{pts} pts: Repeat complaint detected (similarity: {similarity_score:.0%})")
        
    # 5. Direct Financial / Security / Urgency Impact Text Signals (Up to 15 pts)
    text_lower = raw_text.lower()
    has_financial = any(w in text_lower for w in FINANCIAL_KEYWORDS)
    has_urgency = any(w in text_lower for w in URGENCY_KEYWORDS)
    has_security = any(w in text_lower for w in SECURITY_KEYWORDS)
    
    if has_financial and category not in ["Payment", "Refund"]:
        risk_score += 6
        factors.append("+6 pts: Financial dispute terms detected in complaint body")
    if has_security and category != "Account":
        risk_score += 7
        factors.append("+7 pts: Identity / security tokens found in ticket")
    if has_urgency:
        risk_score += 6
        factors.append("+6 pts: Explicit high-urgency keywords detected")
        
    # Clamp risk score to [0, 100]
    risk_score = max(0, min(100, risk_score))
    
    # Determine Risk Level
    if risk_score >= 80:
        risk_level = "Critical"
        escalation_required = True
        business_impact = "Critical"
    elif risk_score >= 60:
        risk_level = "High"
        escalation_required = True
        business_impact = "High"
    elif risk_score >= 35:
        risk_level = "Medium"
        escalation_required = False
        business_impact = "Medium"
    else:
        risk_level = "Low"
        escalation_required = False
        business_impact = "Low"
        
    return {
        "risk_score": int(risk_score),
        "risk_level": risk_level,
        "contributing_factors": factors,
        "escalation_required": escalation_required,
        "business_impact": business_impact,
        "method": "Rule-based risk assessment using ML outputs"
    }
