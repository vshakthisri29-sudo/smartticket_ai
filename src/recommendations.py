def recommend_action(prediction_output: dict, risk_output: dict) -> dict:
    """
    Synthesize ML classifications and risk evaluation into an actionable
    support routing recommendation for human support specialists.
    """
    category = prediction_output.get("category", "General")
    priority = prediction_output.get("priority", "Medium")
    emotion = prediction_output.get("emotion", "Neutral")
    risk_level = risk_output.get("risk_level", "Medium")
    is_repeat = prediction_output.get("is_repeat", False)
    
    # Domain specific core workflow
    workflows = {
        "Payment": "Verify banking transaction status in payment gateway, confirm merchant settlement, and initiate appropriate ledger reconciliation or immediate charge reversal.",
        "Delivery": "Inspect carrier tracking timeline with logistics partner, verify delivery coordinates, and authorize express dispatch or replacement package if parcel lost.",
        "Account": "Perform secondary identity verification, lock compromised authentication sessions, and send a time-limited secure recovery token via authenticated channel.",
        "Technical": "Inspect server error logs matching customer session timestamp, attempt issue reproduction in staging environment, and link ticket to ongoing bug tracking ticket.",
        "Refund": "Review refund eligibility under terms of service, verify bank transaction reference, and authorize disbursement approval through financial operations.",
        "Product": "Request photographic documentation of product condition, verify warranty coverage, and trigger automated pre-paid return label with replacement dispatch."
    }
    
    primary_action = workflows.get(category, "Review customer inquiry and assign to senior triage specialist.")
    
    # Modifiers based on priority, risk, and repeat
    notes = []
    if risk_level in ["Critical", "High"]:
        notes.append("URGENT: Prioritize within 30-minute SLA window.")
    if is_repeat:
        notes.append("REPEAT COMPLAINT: Escalate to Tier-2 supervisor for warm transfer and past-ticket history review.")
    if emotion in ["Angry", "Frustrated"]:
        notes.append("CUSTOMER EMPATHY: Deploy de-escalation tone and consider courtesy credit or apology voucher.")
        
    recommended_text = primary_action
    if notes:
        recommended_text += " " + " ".join(notes)
        
    return {
        "action_text": recommended_text,
        "human_review_required": True,
        "disclaimer": "Generated from ML predictions + business rules. Human Review Recommended before taking customer-impacting action.",
        "routing_department": f"{category} Support Operations"
    }
