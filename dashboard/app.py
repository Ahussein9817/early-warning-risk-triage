"""Section 7, Dashboard.

Streamlit app: a complaint (picked from a real example, or typed in) goes
through the pipeline live, structured fields one-hot encoded exactly as at
training time, narrative embedded live with MiniLM, and both the baseline
(structured-only) and fused (structured + text) models' predictions and
confidence are shown side by side.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import yaml
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RISK_COLOR = "#c0392b"    # escalated, a warning, not good news. Distinct from theme.primaryColor.
SAFE_COLOR = "#2a7a4f"    # not escalated
MUTED_COLOR = "#6b7280"
CALIBRATION_NOTE = (
    "Not calibrated against the hand-labeled evaluation rate (78.3% true "
    "escalation vs. this model's ~26.6% training rate), read as a ranking "
    "signal, not a probability. See README Evaluation for why."
)
NO_NARRATIVE_NOTE = (
    "No narrative, this is the subgroup where evaluation found fused offers "
    "little advantage over baseline."
)


@st.cache_resource
def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_models(_config: dict) -> tuple:
    models_dir = ROOT / _config["models"]["dir"]
    baseline = joblib.load(models_dir / _config["models"]["baseline_filename"])
    fused = joblib.load(models_dir / _config["models"]["fused_filename"])
    return baseline, fused


@st.cache_resource
def load_embedding_model(_config: dict) -> SentenceTransformer:
    return SentenceTransformer(_config["embedding"]["model_name"])


@st.cache_resource
def load_categories(_config: dict) -> dict:
    path = ROOT / _config["data"]["processed_dir"] / "dashboard_categories.json"
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_examples(_config: dict) -> pd.DataFrame:
    path = ROOT / _config["data"]["labeled_path"]
    return pd.read_csv(path, dtype={"complaint_id": str})


def bucket_company(company: str, top_companies: list[str]) -> str:
    return company if company in top_companies else "Other"


def build_baseline_vector(selections: dict, feature_names: pd.Index) -> pd.DataFrame:
    vec = pd.Series(0.0, index=feature_names)
    for field in ["sub_product", "issue", "sub_issue", "state"]:
        value = selections.get(field)
        if value:
            col = f"{field}_{value}"
            if col in vec.index:
                vec[col] = 1.0
    company_col = f"company_bucketed_{selections.get('company')}"
    if company_col in vec.index:
        vec[company_col] = 1.0
    return vec.to_frame().T


def build_fused_vector(
    baseline_vec: pd.DataFrame, narrative: str, embed_model: SentenceTransformer,
    fused_feature_names: pd.Index, embedding_dim: int,
) -> pd.DataFrame:
    if narrative and narrative.strip():
        embedding = embed_model.encode(narrative)
    else:
        embedding = [0.0] * embedding_dim
    emb_series = pd.Series(embedding, index=[f"emb_{i}" for i in range(embedding_dim)])
    row = pd.concat([baseline_vec.iloc[0], emb_series])
    return row.reindex(fused_feature_names).to_frame().T


def predict(model, X: pd.DataFrame) -> tuple[int, float]:
    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= 0.5)
    return pred, proba


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Manrope', sans-serif !important; }

        .twt-hero {
            background: linear-gradient(135deg, #EEF1FB 0%, #FFFFFF 100%);
            border-left: 4px solid #3457D5;
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
        }

        [data-testid="stHeading"] h3 {
            border-left: 4px solid #3457D5;
            padding-left: 0.6rem;
        }

        .st-key-form-card, .st-key-about-card,
        .st-key-baseline-card, .st-key-fused-card {
            border-radius: 14px !important;
            box-shadow: 0 1px 4px rgba(31,41,55,0.06), 0 8px 20px rgba(31,41,55,0.05) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_model_result(name: str, pred: int, proba: float, delta: float | None = None) -> None:
    label = "Escalated" if pred else "Not Escalated"
    color = RISK_COLOR if pred else SAFE_COLOR
    st.markdown(
        f"<div style='border-left:4px solid {color}; padding-left:0.75rem; margin-bottom:0.6rem;'>"
        f"<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.06em;"
        f"text-transform:uppercase;color:{MUTED_COLOR}'>{name}</div>"
        f"<div style='font-size:1.35rem;font-weight:800;color:{color}'>{label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if delta is not None:
        st.metric("Confidence", f"{proba:.1%}", delta=f"{delta:+.1%} vs baseline", delta_color="off")
    else:
        st.metric("Confidence", f"{proba:.1%}")
    st.progress(proba)
    st.caption(CALIBRATION_NOTE)


def main() -> None:
    st.set_page_config(page_title="Early-Warning Risk Triage", layout="wide")
    inject_css()
    st.markdown(
        "<div class='twt-hero'>"
        "<h1 style='font-size:2rem; margin:0 0 0.4rem 0; color:#1F2937;'>"
        "Early-Warning Risk Triage</h1>"
        "<div style='font-size:0.95rem; color:#4b5563;'>Debt Collection Complaints</div>"
        "<div style='font-size:1.05rem; font-weight:500; color:#374151; margin-top:0.75rem;'>"
        "Does adding complaint-narrative text to a structured-feature classifier "
        "improve prediction of case escalation? A complaint goes through both a "
        "structured-only baseline model and a structured+text fused model, live."
        "</div></div>",
        unsafe_allow_html=True,
    )

    config = load_config()
    baseline_model, fused_model = load_models(config)
    embed_model = load_embedding_model(config)
    categories = load_categories(config)
    examples = load_examples(config)

    if "selections" not in st.session_state:
        st.session_state.selections = {
            "sub_product": categories["sub_product"][0],
            "issue": categories["issue"][0],
            "sub_issue": categories["sub_issue"][0],
            "state": categories["state"][0],
            "company": "Other",
        }
        st.session_state.narrative = ""
        st.session_state.true_label = None
        st.session_state.loaded_snapshot = None

    if st.button("Load a random real example"):
        row = examples.sample(1).iloc[0]
        top_companies = [c for c in categories["company_options"] if c != "Other"]
        st.session_state.selections = {
            "sub_product": row["sub_product"],
            "issue": row["issue"],
            "sub_issue": row["sub_issue"],
            "state": row["state"] if pd.notna(row["state"]) else categories["state"][0],
            "company": bucket_company(row["company"], top_companies),
        }
        st.session_state.narrative = row["complaint_what_happened"] if pd.notna(row["complaint_what_happened"]) else ""
        st.session_state.true_label = int(row["escalated"])
        st.session_state.loaded_snapshot = {**st.session_state.selections, "narrative": st.session_state.narrative}

    col_in, col_out = st.columns([1, 1])

    with col_in:
        with st.container(border=True, key="form-card"):
            st.subheader("Complaint going in")
            sel = st.session_state.selections
            sel["sub_product"] = st.selectbox("Sub-product", categories["sub_product"],
                                               index=categories["sub_product"].index(sel["sub_product"]))
            sel["issue"] = st.selectbox("Issue", categories["issue"],
                                         index=categories["issue"].index(sel["issue"]))
            sel["sub_issue"] = st.selectbox("Sub-issue", categories["sub_issue"],
                                             index=categories["sub_issue"].index(sel["sub_issue"]))
            sel["state"] = st.selectbox("State", categories["state"],
                                         index=categories["state"].index(sel["state"]))
            sel["company"] = st.selectbox("Company", categories["company_options"],
                                           index=categories["company_options"].index(sel["company"]))
            narrative = st.text_area("Complaint narrative (optional, leave blank to simulate "
                                      "the ~56% of complaints with no narrative)",
                                      value=st.session_state.narrative, height=200)
            st.session_state.narrative = narrative

            current_snapshot = {**sel, "narrative": narrative}
            if current_snapshot != st.session_state.loaded_snapshot:
                st.session_state.true_label = None

            if st.session_state.true_label is not None:
                label_text = "Escalated" if st.session_state.true_label == 1 else "Not escalated"
                st.info(f"Hand-labeled ground truth for this example: **{label_text}**")

            predict_clicked = st.button("Predict", type="primary")

    with col_out:
        st.subheader("Model predictions")
        if predict_clicked:
            baseline_vec = build_baseline_vector(sel, baseline_model.feature_names_in_)
            fused_vec = build_fused_vector(
                baseline_vec, narrative, embed_model,
                fused_model.feature_names_in_, config["embedding"]["dimension"],
            )

            b_pred, b_proba = predict(baseline_model, baseline_vec)
            f_pred, f_proba = predict(fused_model, fused_vec)

            if b_pred == f_pred:
                label = "Escalated" if b_pred else "Not Escalated"
                st.info(f"**Models agree:** {label}")
            else:
                st.warning(
                    f"**Models disagree**, baseline: "
                    f"{'Escalated' if b_pred else 'Not Escalated'}, fused: "
                    f"{'Escalated' if f_pred else 'Not Escalated'}"
                )

            b_col, f_col = st.columns(2)
            with b_col:
                with st.container(border=True, key="baseline-card"):
                    render_model_result("Baseline (structured only)", b_pred, b_proba)
            with f_col:
                with st.container(border=True, key="fused-card"):
                    render_model_result("Fused (structured + text)", f_pred, f_proba, delta=f_proba - b_proba)

            if not narrative.strip():
                st.caption(NO_NARRATIVE_NOTE)
        else:
            st.write("Fill in the complaint and click **Predict**.")

        with st.container(border=True, key="about-card"):
            with st.expander("About these numbers", expanded=not predict_clicked):
                st.markdown(
                    "- **Confidence is not calibrated.** These models were trained "
                    "on a proxy label with a 26.6% escalation rate; the hand-labeled "
                    "evaluation sample's true rate is 78.3%. A raw confidence score "
                    "should be read as a relative ranking signal, not a real-world "
                    "probability.\n"
                    "- **Baseline vs. fused, on the deconfounded evaluation subgroup "
                    "(has_narrative, n=51): fused AUC 0.6875 vs. baseline 0.6701** "
                    "a modest, directionally positive difference, not a decisive "
                    "one at this sample size.\n"
                    "- Full methodology, the base-rate divergence finding, and all "
                    "limitations are in the project README."
                )

    st.caption(
        "Research prototype, not deployment-ready. Trained on a proxy label "
        "that is not verified ground truth; see the README for full "
        "methodology and limitations."
    )


if __name__ == "__main__":
    main()
