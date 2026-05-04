"""
RefugeeMatch Caseworker Dashboard
Streamlit app for predicting whether a newly-arrived refugee family will be
able to afford monthly living expenses, with explanations and recommendations.

Author: Abbot Tubeine, Sattler College

v2 changes:
  - Streamlined to 5 intake features (age, years_in_camp, education,
    work_status_home, marital_status) — see notebook intro for rationale.
  - Production model is Calibrated Logistic Regression (monotonic, explainable).
  - Risk bands anchored to the training-set base rate so a "16%" prediction
    reads as "barely above average" rather than "MODERATE RISK."
  - Attributions are coefficient-based (structurally monotonic).
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="RefugeeMatch — Caseworker Dashboard",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    with open("refugee_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle = load_model()
    MODEL = bundle["model"]
    THRESHOLD = bundle["threshold"]
    FEATURE_NAMES = bundle["feature_names"]
    MAPS = bundle["encoding_maps"]
    META = bundle["metadata"]
    BASE_RATE = bundle.get("base_rate", 0.127)  # fallback to known value
except FileNotFoundError:
    st.error(
        "❌ Model file `refugee_model.pkl` not found. "
        "Run the training notebook first (section 14) to export the model."
    )
    st.stop()

# Risk-band thresholds anchored to base rate
LOW_THRESHOLD = BASE_RATE * 1.2       # ~15% with base rate of 12.7%
HIGH_THRESHOLD = BASE_RATE * 2.0      # ~25%

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
.risk-card {
    padding: 24px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 16px;
}
.risk-low { background-color: #d4edda; border-left: 6px solid #28a745; }
.risk-moderate { background-color: #fff3cd; border-left: 6px solid #ffc107; }
.risk-high { background-color: #f8d7da; border-left: 6px solid #dc3545; }
.risk-percent { font-size: 56px; font-weight: 700; color: #222; }
.risk-label { font-size: 22px; font-weight: 600; margin-top: 4px; color: #222; }
.factor-positive { color: #c0392b; font-weight: 600; }
.factor-negative { color: #1e8449; font-weight: 600; }
.metric-small { font-size: 13px; color: #666; }
.baseline-note { font-size: 14px; color: #555; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — context for the user
# ============================================================
with st.sidebar:
    st.title("🤝 RefugeeMatch")
    st.caption("Decision support for refugee resettlement caseworkers")

    st.markdown("### About this tool")
    st.write(
        "This tool flags newly-arrived refugee families who may benefit from "
        "**additional case-management support** in their first year, based on "
        "an intake profile compared against ~1,400 historical resettlement outcomes."
    )
    st.write(
        "It's a **structured second opinion**, not a verdict. Caseworker judgment "
        "always takes precedence."
    )

    st.markdown("### Model details")
    st.metric("Test F1 Score", f"{META['test_f1']:.3f}")
    st.metric("Test AUC", f"{META['test_auc']:.3f}")
    st.caption(f"Model: {META.get('model_type', 'Calibrated Logistic Regression')}")
    st.caption(f"Trained on {META['training_rows']:,} refugee records")
    st.caption(f"Source: 2022 Annual Survey of Refugees (ASR)")
    st.caption(f"Population base rate: {BASE_RATE*100:.1f}% report difficulty")

    st.markdown("### Important")
    st.warning(
        "This is a **decision-support tool**. The model has modest predictive "
        "performance (AUC ≈ 0.61) and known limitations: small training sample, "
        "single-year coverage, self-reported outcomes. All flags should be "
        "reviewed by a qualified caseworker."
    )

    st.markdown("---")
    st.caption("Built by Abbot Tubeine · Sattler College · BUS302 Capstone")

# ============================================================
# MAIN HEADER
# ============================================================
st.title("Refugee Family Financial Risk Assessment")
st.markdown(
    "Enter the family's intake information below. Five questions — each one "
    "selected because it has measurable predictive signal in the historical data "
    "and can be defensibly asked at intake."
)

# ============================================================
# INTAKE FORM
# ============================================================
st.markdown("## Intake Form")

with st.form("intake_form"):
    col_a, col_b = st.columns(2)

    # ---------- LEFT COLUMN ----------
    with col_a:
        st.markdown("### 📋 Demographics & Background")
        st.caption("Typically from UNHCR case file")

        family_name = st.text_input(
            "Family / Case ID",
            value="",
            placeholder="e.g. Abebe Family / Case #2026-0142",
            help="For your records — not used by the model.",
        )

        age = st.number_input(
            "Age of primary respondent",
            min_value=16, max_value=99, value=32, step=1,
            help="Working-age adult most likely to be the household earner.",
        )

        marital_status = st.selectbox(
            "Marital status",
            options=[
                "Now married, spouse living in household",
                "Now married, spouse not living in HH",
                "Never married",
                "Divorced or separated",
                "Widowed",
                "Other",
            ],
            help="Captures household composition — single-earner vs. dual-earner households face different financial pressures.",
        )

        st.markdown("### 📋 Pre-Arrival Background")

        lived_in_camp = st.radio(
            "Lived in a refugee camp before coming to the U.S.?",
            options=["No", "Yes"],
            horizontal=True,
        )

        if lived_in_camp == "Yes":
            years_in_camp_label = st.selectbox(
                "How long in the camp?",
                options=["Less than a year", "A year or more", "Your whole life"],
                index=1,
                help="Captures displacement duration — a proxy for skill atrophy and re-entry challenges.",
            )
        else:
            years_in_camp_label = "Did not live in camp"

    # ---------- RIGHT COLUMN ----------
    with col_b:
        st.markdown("### 🗣️ Asked at Intake Interview")
        st.caption("Caseworker asks/assesses these directly")

        education = st.selectbox(
            "Highest level of school completed (before coming to the U.S.)",
            options=[
                "No schooling",
                "Religious school",
                "Primary or elementary school",
                "Lower secondary school or middle school",
                "Upper secondary school (or high school)",
                "Some technical or vocational training",
                "Some university but did not get degree",
                "University (bachelor's degree)",
                "Advanced degree",
                "Other",
            ],
            index=4,
            help="The single strongest predictor in the historical data (p = 0.002).",
        )

        work_status = st.selectbox(
            "What was their work status in their home country?",
            options=[
                "Working for paid jobs",
                "Self-employed",
                "Not working for pay",
                "Other",
            ],
            help='If they were a student, retired, caregiver, or any other reason for not working, choose "Not working for pay".'
        )

        st.markdown("### ℹ️ Note on excluded fields")
        st.caption(
            "Earlier versions of this tool collected sex, household size, English ability, "
            "native-language literacy, and resettlement region. Those fields were removed "
            "because (a) sex is a protected demographic, and (b) the others showed no "
            "measurable predictive signal in the historical data (p > 0.15). Asking them "
            "would add noise to the explanation without improving the assessment."
        )

    submitted = st.form_submit_button(
        "🔍 Assess Risk",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
def build_feature_row(intake_input: dict) -> pd.DataFrame:
    """Convert intake form values into the model's 5-feature schema."""
    row = pd.Series(0, index=FEATURE_NAMES, dtype=float)

    # Numeric / ordinal features
    if "age" in row.index:
        row["age"] = float(intake_input["age"])
    if "years_in_camp" in row.index:
        row["years_in_camp"] = MAPS["camp_years_map"].get(intake_input["years_in_camp_label"], 0)
    if "education_pre_arrival" in row.index:
        row["education_pre_arrival"] = MAPS["edu_map"][intake_input["education"]]

    # One-hot dummies — marital status
    marital_collapsed = MAPS["marital_status_map"].get(intake_input["marital_status"], "Other")
    marital_col = f"marital_status_{marital_collapsed}"
    if marital_col in row.index:
        row[marital_col] = 1

    # One-hot dummies — work status
    work_collapsed = MAPS["work_status_map"].get(intake_input["work_status"], "Other")
    work_col = f"work_status_home_{work_collapsed}"
    if work_col in row.index:
        row[work_col] = 1

    return row.to_frame().T


def explain_prediction(feature_row: pd.DataFrame, intake_input: dict) -> list:
    """
    Compute per-feature contributions by substituting in a low-risk baseline
    profile, one feature group at a time, and observing the change in predicted
    probability. With Logistic Regression this is structurally monotonic:
    setting a feature to its low-risk baseline can only ever decrease (or leave
    unchanged) the predicted risk.
    """
    base_prob = MODEL.predict_proba(feature_row)[0, 1]

    # Low-risk baseline values — chosen to match the protective end of each feature's range
    baseline_overrides = {
        "education_pre_arrival": 7,           # university bachelor's degree
        "years_in_camp": 0,                   # did not live in camp
        "age": 32,                            # prime working age
        # marital baseline: married with spouse in household
        "marital_status_Married": 1,
        "marital_status_Married_apart": 0,
        "marital_status_Never_married": 0,
        "marital_status_Divorced_Widowed": 0,
        "marital_status_Other": 0,
        # work baseline: employed in home country
        "work_status_home_Employed": 1,
        "work_status_home_Self_employed": 0,
        "work_status_home_Not_working": 0,
        "work_status_home_Other": 0,
    }

    feature_groups = {
        "Education": ["education_pre_arrival"],
        "Refugee camp history": ["years_in_camp"],
        "Age": ["age"],
        "Marital status": [c for c in FEATURE_NAMES if c.startswith("marital_status_")],
        "Work history (home country)": [c for c in FEATURE_NAMES if c.startswith("work_status_home_")],
    }

    contributions = []
    for label, cols in feature_groups.items():
        cols_present = [c for c in cols if c in FEATURE_NAMES]
        if not cols_present:
            continue
        modified = feature_row.copy()
        for c in cols_present:
            modified[c] = baseline_overrides.get(c, 0)
        new_prob = MODEL.predict_proba(modified)[0, 1]
        contribution = base_prob - new_prob  # positive = this feature pushed risk UP vs. baseline
        contributions.append({
            "factor": label,
            "contribution": contribution,
            "value": _human_readable_value(label, intake_input),
        })

    return sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)


def _human_readable_value(label: str, intake: dict) -> str:
    """Format the user-facing value of each input factor."""
    if label == "Education":
        return intake["education"]
    if label == "Refugee camp history":
        if intake["lived_in_camp"] == "No":
            return "Did not live in a refugee camp"
        return f"Lived in camp: {intake['years_in_camp_label']}"
    if label == "Age":
        return f"{intake['age']} years"
    if label == "Marital status":
        return intake["marital_status"]
    if label == "Work history (home country)":
        return intake["work_status"]
    return ""


def risk_band(prob: float) -> tuple:
    """Map probability to a base-rate-anchored risk band.

    Returns (css_class, label, summary_message_kind).
    """
    if prob < LOW_THRESHOLD:
        return "risk-low", "LOW RISK", "low"
    elif prob < HIGH_THRESHOLD:
        return "risk-moderate", "ELEVATED RISK", "moderate"
    else:
        return "risk-high", "HIGH RISK", "high"


def get_recommendations(top_factors: list, prob: float) -> list:
    """Generate caseworker action recommendations based on top risk drivers."""
    recs = []
    # Only act on factors that contribute meaningfully (>1pp) AND push risk up
    risk_factors = [f for f in top_factors[:5] if f["contribution"] > 0.01]

    factor_to_recs = {
        "Education": [
            "Consider connecting the family with adult education or GED programs.",
            "Match the primary earner with vocational training appropriate for their interests and the local labor market.",
        ],
        "Refugee camp history": [
            "Schedule a trauma-informed mental health screening — long camp stays often correlate with PTSD or chronic stress.",
            "Consider an extended R&P period if available, to allow more recovery time before employment placement.",
        ],
        "Age": [
            "If older respondent (50+): connect to senior-specific resources and SSI eligibility review.",
            "If very young respondent (<22): prioritize education over immediate employment placement.",
        ],
        "Work history (home country)": [
            "Conduct a detailed skills assessment to identify transferable work experience.",
            "Consider job-readiness training before placement to address employment gaps.",
        ],
        "Marital status": [
            "If single parent: ensure childcare arrangements are in place before employment placement.",
            "Connect to family support networks in the resettlement community.",
        ],
    }

    for f in risk_factors:
        if f["factor"] in factor_to_recs:
            recs.extend(factor_to_recs[f["factor"]][:2])

    # Risk-level recommendations
    if prob >= HIGH_THRESHOLD:
        recs.insert(0, "🚨 **High priority case** — schedule a follow-up case-management meeting within the first 2 weeks of arrival.")
    elif prob >= LOW_THRESHOLD:
        recs.insert(0, "⚠️ **Elevated priority** — schedule a 30-day check-in to monitor early adjustment.")

    # Dedupe while preserving order
    seen = set()
    deduped = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped[:8]


def build_report(intake_input: dict, prob: float, risk_label: str,
                 top_factors: list, recommendations: list) -> bytes:
    """Generate a downloadable text report for the case file."""
    lines = [
        "=" * 70,
        "REFUGEEMATCH FINANCIAL RISK ASSESSMENT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
        "CASE INFORMATION",
        "-" * 70,
        f"Family / Case ID: {intake_input.get('family_name') or '(not provided)'}",
        "",
        "INTAKE PROFILE",
        "-" * 70,
        f"Age: {intake_input['age']}",
        f"Marital status: {intake_input['marital_status']}",
        f"Education: {intake_input['education']}",
        f"Work in home country: {intake_input['work_status']}",
        f"Refugee camp: {intake_input['lived_in_camp']}",
    ]
    if intake_input["lived_in_camp"] == "Yes":
        lines.append(f"  Duration: {intake_input['years_in_camp_label']}")

    lines += [
        "",
        "RISK ASSESSMENT",
        "-" * 70,
        f"Predicted probability of financial difficulty: {prob*100:.1f}%",
        f"Population base rate for comparison: {BASE_RATE*100:.1f}%",
        f"Risk tier: {risk_label}",
        f"Decision threshold (used for binary flagging): {THRESHOLD*100:.0f}%",
        "",
        "TOP CONTRIBUTING FACTORS",
        "-" * 70,
        "(How each piece of intake info shifted risk vs. a low-risk baseline profile)",
        "",
    ]
    for i, f in enumerate(top_factors, 1):
        if abs(f["contribution"]) < 0.005:
            continue
        direction = "increases risk" if f["contribution"] > 0 else "decreases risk"
        lines.append(f"{i}. {f['factor']}: {f['value']}")
        lines.append(f"   Effect: {direction} ({abs(f['contribution'])*100:.1f} percentage points vs. baseline)")

    lines += [
        "",
        "RECOMMENDED CASEWORKER ACTIONS",
        "-" * 70,
    ]
    for i, r in enumerate(recommendations, 1):
        clean_r = r.replace("**", "").replace("🚨", "").replace("⚠️", "").strip()
        lines.append(f"{i}. {clean_r}")

    lines += [
        "",
        "CASEWORKER NOTES (for case file)",
        "-" * 70,
        "Do you agree with this assessment? [ ] Agree   [ ] Disagree",
        "If you disagree, please note your reasoning:",
        "",
        "_" * 70,
        "_" * 70,
        "_" * 70,
        "",
        "Caseworker name: ____________________     Date: ____________",
        "",
        "=" * 70,
        "DISCLAIMER",
        "-" * 70,
        "This is a decision-support tool, not a decision-making tool. The model is",
        "trained on the 2022 Annual Survey of Refugees and has known limitations:",
        "small sample size, single-year coverage, and self-reported outcomes. The",
        "model surfaces patterns; it does not reliably identify individual at-risk",
        "families. All flags should be reviewed by a qualified caseworker.",
        "",
        f"Model: {META.get('model_type', 'Calibrated Logistic Regression')} | "
        f"Test F1: {META['test_f1']:.3f} | Test AUC: {META['test_auc']:.3f}",
        "=" * 70,
    ]

    return "\n".join(lines).encode("utf-8")


# ============================================================
# RESULTS SECTION
# ============================================================
if submitted:
    intake_input = {
        "family_name": family_name,
        "age": age,
        "marital_status": marital_status,
        "lived_in_camp": lived_in_camp,
        "years_in_camp_label": years_in_camp_label,
        "education": education,
        "work_status": work_status,
    }

    feature_row = build_feature_row(intake_input)
    prob = float(MODEL.predict_proba(feature_row)[0, 1])

    risk_class, risk_label, msg_kind = risk_band(prob)
    pp_vs_baseline = (prob - BASE_RATE) * 100

    st.markdown("---")
    st.markdown("## Risk Assessment Results")

    # Risk card + interpretation
    col_risk, col_info = st.columns([2, 3])
    with col_risk:
        sign = "+" if pp_vs_baseline >= 0 else ""
        st.markdown(f"""
            <div class="risk-card {risk_class}">
                <div class="risk-percent">{prob*100:.0f}%</div>
                <div class="risk-label">{risk_label}</div>
                <div style="margin-top: 12px; font-size: 14px; color: #333;">
                    Probability of financial difficulty in first year
                </div>
                <div class="baseline-note">
                    {sign}{pp_vs_baseline:.1f} pp vs. {BASE_RATE*100:.0f}% population baseline
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown("### What this means")
        if msg_kind == "low":
            st.success(
                "This family's intake profile is **at or below the population baseline** "
                "for financial difficulty. Standard case management is likely sufficient. "
                "Routine 90-day check-in recommended."
            )
        elif msg_kind == "moderate":
            st.warning(
                "This family's intake profile shows **modestly elevated risk** compared "
                "to the population baseline. Review the contributing factors below — a "
                "30-day check-in and targeted support may help prevent early difficulties."
            )
        else:
            st.error(
                "This family's intake profile shows **substantially elevated risk** of "
                "financial difficulty. Proactive intervention is recommended — see action "
                "list below."
            )

        st.caption(
            f"Risk bands are anchored to the {BASE_RATE*100:.1f}% population base rate: "
            f"Low <{LOW_THRESHOLD*100:.0f}%, Elevated {LOW_THRESHOLD*100:.0f}–{HIGH_THRESHOLD*100:.0f}%, "
            f"High >{HIGH_THRESHOLD*100:.0f}%. Model AUC: {META['test_auc']:.2f} — "
            f"this tool surfaces patterns rather than identifying individual cases with high precision."
        )

    # Top contributing factors
    st.markdown("### Top contributing factors")
    st.caption(
        "How each piece of intake information shifted the risk assessment compared to a "
        "low-risk baseline profile (university degree, employed, working-age, married, no camp). "
        "Red bars push risk up; green bars push risk down."
    )

    factors = explain_prediction(feature_row, intake_input)
    factors_to_show = [f for f in factors if abs(f["contribution"]) > 0.005]

    if factors_to_show:
        for f in factors_to_show:
            direction = "↑ Increases" if f["contribution"] > 0 else "↓ Decreases"
            color_class = "factor-positive" if f["contribution"] > 0 else "factor-negative"
            magnitude = abs(f["contribution"]) * 100

            col1, col2, col3 = st.columns([2, 3, 2])
            with col1:
                st.markdown(f"**{f['factor']}**")
            with col2:
                st.caption(f"Value: *{f['value']}*")
            with col3:
                st.markdown(
                    f"<span class='{color_class}'>{direction} risk by {magnitude:.1f} pp</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.info(
            "This family's profile is very close to the low-risk baseline — no single "
            "factor moved the assessment substantially in either direction."
        )

    # Recommendations
    st.markdown("### Recommended caseworker actions")
    recommendations = get_recommendations(factors, prob)
    if recommendations:
        for r in recommendations:
            st.markdown(f"- {r}")
    else:
        st.info(
            "No specific actions flagged — this family appears well-positioned for standard "
            "resettlement support."
        )

    # Caseworker disagreement capture (placeholder — future v2 feature)
    st.markdown("### Caseworker review")
    st.caption(
        "Your judgment takes precedence over the model. If you disagree with this assessment, "
        "note your reasoning below before downloading the report — it becomes part of the case file "
        "and helps us improve the model in future versions."
    )
    agree = st.radio(
        "Do you agree with this assessment?",
        options=["Agree", "Partially agree", "Disagree"],
        horizontal=True,
        key="agreement",
    )
    disagreement_note = ""
    if agree != "Agree":
        disagreement_note = st.text_area(
            "Your reasoning (optional)",
            placeholder="e.g. Family has a brother already established locally who can provide housing and job leads — model can't see this.",
            key="disagreement_note",
        )

    # Download report
    st.markdown("---")
    st.markdown("### Save this assessment")
    report_bytes = build_report(intake_input, prob, risk_label, factors, recommendations)
    st.download_button(
        label="📄 Download Assessment Report",
        data=report_bytes,
        file_name=f"risk_assessment_{(family_name or 'case').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.caption(
        "⚠️ **Disclaimer:** This assessment is a decision-support tool only. "
        "All flagged cases should be reviewed by a qualified caseworker. "
        "The model has known limitations including small training data and single-year coverage."
    )

else:
    st.info("👆 Fill in the intake form above and click **Assess Risk** to see the assessment.")
