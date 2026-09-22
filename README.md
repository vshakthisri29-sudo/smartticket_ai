# 🤖 SmartTicket AI

> **From Customer Complaint to Intelligent Action — Powered by Machine Learning.**

An explainable machine-learning system for customer support ticket classification, prioritization, emotion analysis, repeat-complaint detection, transparent risk scoring, and intelligent decision support.

---

## 📌 Problem Statement
Customer support departments receive massive volumes of unstructured tickets every hour. Traditional systems rely on either manual triage—which causes severe delays for urgent financial and security blockers—or opaque black-box rules. Furthermore, repeat complaints are frequently misrouted to junior queues as new cases.

**SmartTicket AI** transforms unstructured complaint text into an auditable, multi-output prediction hierarchy:
1. **Category ML Model**: Routes complaints to the proper domain team.
2. **Priority ML Model**: Automatically identifies SLA urgency directly from text.
3. **Emotion Classifier**: Detects customer friction and distress.
4. **TF-IDF Vector Similarity**: Detects repeat complaints across historical ticket archives.
5. **Explainable AI (XAI)**: Reveals exact linear token contributions behind predictions.
6. **Transparent Risk Engine**: Computes a 0–100 composite risk score using ML outputs.
7. **Decision Support**: Delivers specific workflow recommendations for human agents.

---

## 💡 Design Thinking Framework

- **Empathize**: Customers dread repeated explanations, sluggish response times, and unhelpful canned replies. Support specialists struggle under ticket backlogs without clarity on which tickets represent severe business or security risks.
- **Define**: Manual triage creates operational bottlenecks, misses mission-critical escalations, and fails to identify customer churn signals early.
- **Ideate**: Build a transparent, explainable ML triage system that classifies categories, flags priorities, quantifies risk, and highlights which exact words drove the predictions.
- **Prototype**: An interactive Streamlit application featuring live inference, token highlighting, dynamic risk analysis, and historical audit logs.
- **Test**: Rigorous empirical evaluation across held-out test splits, measuring precision, recall, confusion matrices, and validating against real-world noisy ticket variations.

---

## 🏗️ System Architecture

```
                       Customer Complaint Text
                                 │
                                 ▼
                     src/preprocessing.py
         [Whitespace normalization, token preservation]
                                 │
                                 ▼
                 TfidfVectorizer(ngram_range=(1,2))
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
Category Pipeline        Priority Pipeline        Emotion Pipeline
(Logistic Regression)    (Logistic Regression)    (Logistic Regression)
  6 Classes                4 Classes                5 Classes
        │                        │                        │
        └────────────────────────┬────────────────────────┘
                                 │
             ┌───────────────────┴───────────────────┐
             ▼                                       ▼
    src/explain.py                          src/similarity.py
  Intrinsic Linear Attribution            TF-IDF Cosine Similarity
  (x_i * w_ij contribution)               against Historical Tickets
             │                                       │
             └───────────────────┬───────────────────┘
                                 ▼
                         src/risk_engine.py
                     [Transparent 0–100 Score]
                                 │
                                 ▼
                       src/recommendations.py
                   [Workflow Recommendation]
                                 │
                                 ▼
                      Human Support Specialist
                                 │
                                 ▼
                      Customer Ticket Resolved
```

---

## ⚖️ ML Models vs. Business Logic Layer

| Dimension | Probabilistic ML Layer | Deterministic Business Logic |
| :--- | :--- | :--- |
| **Components** | Category, Priority, Emotion, Cosine Similarity | Composite Risk Score (0-100), Escalation Rules, SLAs |
| **Technique** | TF-IDF + Multinomial Logistic Regression | Auditable weighted formulas & rule boundaries |
| **Output** | Posterior probabilities (`predict_proba`) | Discrete action workflows, human review flags |
| **Transparency** | Mathematical feature weight attribution ($x_i \cdot w_i$) | Clear point-by-point factor breakdown |

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.11+
- Virtual environment recommended

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Train Models
```bash
python -m src.generate_dataset
python -m src.train
python -m src.evaluate
```

### 4. Run Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests
```bash
pytest
```
Tests validate text preprocessing, 0–100 risk score bounds, probability ranges, and repeat similarity detection.

---

## 📊 Dataset & Class Taxonomy
- **Total Generated Tickets**: 1,500 synthetic tickets with realistic noise, typos, and urgency tokens.
- **Categories (6)**: `Payment`, `Delivery`, `Account`, `Technical`, `Refund`, `Product`.
- **Priorities (4)**: `Low`, `Medium`, `High`, `Critical`.
- **Emotions (5)**: `Neutral`, `Frustrated`, `Angry`, `Worried`, `Satisfied`.

---

## ⚠️ Limitations & Future Improvements
- **Calibration**: Probabilities are raw model estimates. Future work can implement `CalibratedClassifierCV` (isotonic or Platt scaling).
- **Embeddings**: Future enhancements could include domain-adapted dense sentence embeddings (e.g., Sentence-Transformers) for multilingual ticket similarity.
- **Model Drift**: Implement automated population stability index (PSI) monitoring over ticket history logs.
