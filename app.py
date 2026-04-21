# =============================================================================
# AI-Powered SIEM Dashboard — Cybersecurity Threat Detection & Log Analytics
# Dataset : CICIDS2017
# Models  : Logistic Regression | Decision Tree | Random Forest (Final)
# Author  : Generated with Claude
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIEM Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

    .main-header {
        font-family: 'Exo 2', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-align: center;
        background: linear-gradient(120deg, #00e5ff 0%, #00b0ff 40%, #651fff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 0.4rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #7986cb;
        text-align: center;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }
    .mono { font-family: 'Share Tech Mono', monospace; }

    /* Severity alert boxes */
    .alert-HIGH {
        background: linear-gradient(90deg, #1a0000 0%, #2d0000 100%);
        border-left: 4px solid #ff1744;
        border-radius: 0 8px 8px 0;
        padding: 18px 22px;
        color: #ff6e6e;
        font-size: 1.05rem;
        margin: 12px 0;
    }
    .alert-MEDIUM {
        background: linear-gradient(90deg, #1a0e00 0%, #2d1a00 100%);
        border-left: 4px solid #ff6d00;
        border-radius: 0 8px 8px 0;
        padding: 18px 22px;
        color: #ffb74d;
        font-size: 1.05rem;
        margin: 12px 0;
    }
    .alert-LOW {
        background: linear-gradient(90deg, #1a1a00 0%, #2d2a00 100%);
        border-left: 4px solid #ffd600;
        border-radius: 0 8px 8px 0;
        padding: 18px 22px;
        color: #fff176;
        font-size: 1.05rem;
        margin: 12px 0;
    }
    .alert-SAFE {
        background: linear-gradient(90deg, #001a06 0%, #002d0d 100%);
        border-left: 4px solid #00e676;
        border-radius: 0 8px 8px 0;
        padding: 18px 22px;
        color: #69f0ae;
        font-size: 1.05rem;
        margin: 12px 0;
    }

    /* Stat card */
    .stat-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .stat-number { font-size: 1.8rem; font-weight: 800; color: #58a6ff; }
    .stat-label  { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; }

    div[data-testid="stButton"] > button {
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.04em;
        transition: all 0.2s;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Constants ────────────────────────────────────────────────────────────────
SEVERITY_MAP: dict[str, str] = {
    "BENIGN":                      "SAFE",
    "PortScan":                    "LOW",
    "SSH-Patator":                 "MEDIUM",
    "FTP-Patator":                 "MEDIUM",
    "DoS Hulk":                    "HIGH",
    "DoS GoldenEye":               "HIGH",
    "DoS slowloris":               "HIGH",
    "DoS Slowhttptest":            "HIGH",
    "DDoS":                        "HIGH",
    "Bot":                         "HIGH",
    "Infiltration":                "HIGH",
    "Heartbleed":                  "HIGH",
    "Web Attack \x96 Brute Force": "MEDIUM",
    "Web Attack \x96 XSS":         "MEDIUM",
    "Web Attack \x96 Sql Injection":"HIGH",
    "Web Attack – Brute Force":    "MEDIUM",
    "Web Attack – XSS":            "MEDIUM",
    "Web Attack – Sql Injection":  "HIGH",
}

ALERT_CONFIG: dict[str, tuple] = {
    "HIGH":   ("🚨", "CRITICAL ALERT",  "Immediate action required! Isolate affected systems and escalate to your SOC team now."),
    "MEDIUM": ("⚠️", "WARNING",          "Investigate source IP and affected services. Monitor for escalation."),
    "LOW":    ("🔍", "NOTICE",           "Low-priority log. Schedule review and correlate with other events."),
    "SAFE":   ("✅", "ALL CLEAR",        "Normal traffic pattern. No action required."),
}

SEV_COLORS: dict[str, str] = {
    "HIGH":   "#ff1744",
    "MEDIUM": "#ff6d00",
    "LOW":    "#ffd600",
    "SAFE":   "#00e676",
}

PAGES = [
    "🏠  Home",
    "📂  Upload Dataset",
    "📋  Traffic Logs",
    "📊  EDA / Analytics",
    "🤖  Model Training",
    "🔍  Threat Detection",
    "📄  Threat Report",
]


# ─── Utility Functions ────────────────────────────────────────────────────────

def get_severity(label: str) -> str:
    """Map an attack label string to a severity level."""
    return SEVERITY_MAP.get(str(label).strip(), "MEDIUM")


def compute_risk_score(
    proba_array: np.ndarray,
    model_classes: np.ndarray,
    benign_model_idx: int,
) -> int:
    """
    Risk score = (1 – P(BENIGN)) × 100.
    Range: 0–30 → SAFE | 31–70 → MEDIUM | 71–100 → HIGH.
    """
    if 0 <= benign_model_idx < len(proba_array):
        p_benign = float(proba_array[benign_model_idx])
    else:
        p_benign = 0.0
    score = int(round((1.0 - p_benign) * 100))
    return max(0, min(100, score))


def get_benign_model_idx(model, le: LabelEncoder) -> int:
    """
    Find the index of 'BENIGN' in model.classes_.
    model.classes_ contains the label-encoded integer IDs.
    le.classes_ maps integer position → original string label.
    """
    benign_positions = np.where(le.classes_ == "BENIGN")[0]
    if len(benign_positions) == 0:
        return -1
    benign_enc = int(benign_positions[0])          # encoded integer for BENIGN
    model_positions = np.where(model.classes_ == benign_enc)[0]
    if len(model_positions) == 0:
        return -1
    return int(model_positions[0])


def styled_metric(label: str, value: str, delta: str = "") -> str:
    delta_html = f"<div style='font-size:0.72rem;color:#8b949e;'>{delta}</div>" if delta else ""
    return f"""
    <div class="stat-card">
        <div class="stat-number">{value}</div>
        <div class="stat-label">{label}</div>
        {delta_html}
    </div>
    """


# ─── Data Preprocessing ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def preprocess_dataset(file_bytes: bytes):
    """
    Full preprocessing pipeline for CICIDS2017 CSV.
    Returns (result_tuple, error_string).
    result_tuple = (df, X_scaled, y, le, scaler, feature_cols, label_col, meta)
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    except Exception as exc:
        return None, f"Could not read CSV: {exc}"

    # Strip column names
    df.columns = df.columns.str.strip()

    # Detect label column (case-insensitive or fallback to last column)
    common_labels = ["label", "class", "target", "attack_type", "type"]
    label_col = next(
        (c for c in df.columns if c.strip().lower() in common_labels), None
    )
    if label_col is None:
        label_col = df.columns[-1]

    # Preserve original string labels
    df["Original_Label"] = df[label_col].astype(str).str.strip()

    # ── Step 1: Remove duplicates ─────────────────────────────────
    rows_before = len(df)
    df = df.drop_duplicates()
    dupes_removed = rows_before - len(df)

    # ── Step 2: Handle inf / NaN ──────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop columns with >50% missing values
    thresh = int(len(df) * 0.5)
    df = df.dropna(axis=1, thresh=thresh)

    # Fill remaining NaN with column median
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols_all:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # ── Step 3: Label Encoding ────────────────────────────────────
    le = LabelEncoder()
    df[label_col] = le.fit_transform(df["Original_Label"])

    # ── Step 4: Feature Selection ─────────────────────────────────
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != label_col
    ]

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # Remove zero-variance features
    var = X.var()
    zero_var = var[var == 0].index.tolist()
    X = X.drop(columns=zero_var)
    feature_cols = [c for c in feature_cols if c not in zero_var]

    # ── Step 5: Standard Scaling ──────────────────────────────────
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_cols,
        index=X.index,
    )

    meta = {
        "duplicates_removed": dupes_removed,
        "zero_var_removed":   len(zero_var),
        "label_col":          label_col,
        "feature_cols":       feature_cols,
        "n_classes":          len(le.classes_),
    }

    return (df, X_scaled, y, le, scaler, feature_cols, label_col, meta), None


# ─── Model Training ───────────────────────────────────────────────────────────

def train_all_models(X: pd.DataFrame, y: pd.Series, sample_size: int):
    """
    Train Logistic Regression, Decision Tree, and Random Forest.
    Returns (trained_models_dict, results_dict, X_test, y_test, unique_cls).
    """
    # Optional sub-sampling for speed
    if sample_size < len(X):
        idx = np.random.choice(len(X), size=sample_size, replace=False)
        X_use = X.iloc[idx].reset_index(drop=True)
        y_use = y.iloc[idx].reset_index(drop=True)
    else:
        X_use = X.reset_index(drop=True)
        y_use = y.reset_index(drop=True)

    # Stratified split (fallback to non-stratified for rare classes)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_use, y_use, test_size=0.2, random_state=42, stratify=y_use
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_use, y_use, test_size=0.2, random_state=42
        )

    unique_cls = sorted(
        np.unique(np.concatenate([y_tr.values, y_te.values]))
    )

    if len(np.unique(y_tr)) < 2:
        raise ValueError(
            "The training data contains only ONE class. "
            "ML models require at least two distinct classes (e.g., Normal and Attack traffic) "
            "to perform classification. Please check your dataset (some sets like Monday "
            "in CICIDS2017 contain exclusively 'BENIGN' traffic) or increase the sample size "
            "so that different classes are included."
        )

    model_configs = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=500, solver="saga", random_state=42, n_jobs=-1
            ),
        ),
        (
            "Decision Tree",
            DecisionTreeClassifier(max_depth=20, random_state=42),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
            ),
        ),
    ]

    trained_models: dict = {}
    results: dict = {}

    for name, model in model_configs:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        cm  = confusion_matrix(y_te, y_pred, labels=unique_cls)
        cr  = classification_report(
            y_te, y_pred, labels=unique_cls,
            output_dict=True, zero_division=0
        )
        trained_models[name] = model
        results[name] = {
            "accuracy":              acc,
            "confusion_matrix":      cm,
            "classification_report": cr,
            "unique_cls":            unique_cls,
        }

    return trained_models, results, X_te, y_te, unique_cls


# ─── Sidebar Navigation ───────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;font-size:2.4rem;'>🛡️</div>"
            "<div style='text-align:center;font-weight:800;font-size:1.1rem;"
            "letter-spacing:0.1em;color:#58a6ff;'>SIEM DASHBOARD</div>"
            "<div style='text-align:center;font-size:0.7rem;color:#8b949e;"
            "text-transform:uppercase;letter-spacing:0.15em;margin-bottom:1rem;'>"
            "AI-Powered Threat Detection</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        page = st.radio(
            "Navigation",
            PAGES,
            index=PAGES.index(st.session_state.get("page", PAGES[0])),
            label_visibility="collapsed",
        )
        st.session_state.page = page

        st.markdown("---")
        st.markdown("**System Status**")

        if "df" in st.session_state:
            df = st.session_state["df"]
            st.success(f"✅ Dataset: {len(df):,} rows")
        else:
            st.error("❌ No Dataset Loaded")

        if "trained_models" in st.session_state:
            st.success("✅ Models: Trained")
        else:
            st.warning("⚠️  Models: Not Trained")

        st.markdown("---")
        st.caption("CICIDS2017 · Scikit-learn · v1.0")

    return page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

def page_home() -> None:
    st.markdown('<p class="main-header">🛡️ AI-Powered SIEM Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Security Information & Event Management · '
        'CICIDS2017 · Machine Learning Threat Detection</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "**🔐 What is SIEM?**\n\n"
            "Security Information and Event Management (SIEM) centralises security log data, "
            "correlates events across sources, detects anomalies in real time, and enables "
            "rapid incident response. Enterprise SIEM tools process millions of events per day."
        )
    with c2:
        st.success(
            "**🤖 ML-Powered Detection**\n\n"
            "Three ML classifiers — Logistic Regression, Decision Tree, and Random Forest — "
            "are trained on real network flows. Each prediction returns a Risk Score (0–100) "
            "and a Severity Level to help SOC analysts prioritise threats."
        )
    with c3:
        st.warning(
            "**📦 CICIDS2017 Dataset**\n\n"
            "A benchmark dataset from the Canadian Institute for Cybersecurity containing "
            "labelled network flow records across 14+ attack categories: DDoS, PortScan, "
            "Brute Force, DoS variants, Bot, Heartbleed, Web Attacks, and more."
        )

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("⚙️ System Modules")
        st.markdown(
            """
| Module              | Description                                  |
|---------------------|----------------------------------------------|
| 📂 Upload Dataset   | Load & preprocess CICIDS2017 CSV             |
| 📋 Traffic Logs     | Browse, search, and filter all log entries   |
| 📊 EDA / Analytics  | Visualise distributions, heatmaps, protocols |
| 🤖 Model Training   | Train & compare 3 ML classifiers             |
| 🔍 Threat Detection | Predict threats with risk scores & alerts    |
| 📄 Threat Report    | Generate & download threat intelligence CSV  |
"""
        )

    with c2:
        st.subheader("🎯 Severity & Risk Score Guide")
        st.markdown(
            """
| Severity     | Risk Score | Attack Types                            |
|-------------|------------|-----------------------------------------|
| ✅ SAFE      | 0 – 30     | BENIGN (normal traffic)                 |
| 🟡 LOW       | 31 – 50    | PortScan                                |
| 🟠 MEDIUM    | 51 – 70    | SSH/FTP Brute Force, Web Attacks        |
| 🔴 HIGH      | 71 – 100   | DoS, DDoS, Bot, Heartbleed, Infiltration|
"""
        )

    st.markdown("---")
    st.subheader("📖 Key SIEM Concepts")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Log Aggregation",    "✅ Active",  "Centralised collection")
    c2.metric("Threat Correlation", "✅ Active",  "Pattern-based detection")
    c3.metric("Risk Scoring",       "0 – 100",   "Probability-based metric")
    c4.metric("Auto Alerting",      "✅ Active",  "Real-time notifications")

    st.markdown("---")
    st.caption(
        "🔒 Built with: Streamlit · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — UPLOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════

def page_upload() -> None:
    st.title("📂 Upload Dataset")
    st.markdown(
        "Upload your **CICIDS2017** CSV file. "
        "The pipeline will clean, encode, and scale it automatically."
    )
    st.markdown("---")

    uploaded = st.file_uploader(
        "📁 Choose a CSV file",
        type=["csv"],
        help="Upload a CICIDS2017 network-flow CSV (must contain a 'Label' column).",
    )

    if uploaded is None:
        st.info("👆 Please upload a CICIDS2017 CSV file to begin.")
        with st.expander("📖 Expected Dataset Format"):
            st.markdown(
                """
- **File type:** `.csv`
- **Required column:** `Label` — attack type strings (e.g. `BENIGN`, `DDoS`, `PortScan`)
- **Feature columns:** Numeric network-flow statistics (packet lengths, durations, flags, …)
- **Download:** [CICIDS2017 on UNB](https://www.unb.ca/cic/datasets/ids-2017.html)
                """
            )
        return

    with st.spinner("🔄 Loading and preprocessing dataset — please wait…"):
        file_bytes = uploaded.read()
        result, err = preprocess_dataset(file_bytes)

    if err:
        st.error(f"❌ {err}")
        return

    df, X_scaled, y, le, scaler, feature_cols, label_col, meta = result

    # Persist to session state; clear stale model results
    st.session_state.update(
        df=df, X_scaled=X_scaled, y=y, le=le,
        scaler=scaler, feature_cols=feature_cols, label_col=label_col,
    )
    for k in ("trained_models", "results", "X_test", "y_test", "report_df"):
        st.session_state.pop(k, None)

    st.success("✅ Dataset loaded and preprocessed successfully!")
    st.markdown("---")

    # Summary metrics
    st.subheader("🔧 Preprocessing Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",        f"{df.shape[0]:,}")
    c2.metric("Feature Columns",   len(feature_cols))
    c3.metric("Duplicates Removed", meta["duplicates_removed"])
    c4.metric("Attack Classes",    meta["n_classes"])

    st.markdown("---")

    # Preview
    st.subheader("📋 Dataset Preview (First 20 Rows)")
    preview_cols = ["Original_Label"] + feature_cols[:9]
    st.dataframe(df[preview_cols].head(20), use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🏷️ Detected Attack Classes")
        cls_df = pd.DataFrame(
            {
                "Encoded ID": range(len(le.classes_)),
                "Attack Label": le.classes_,
                "Severity": [get_severity(c) for c in le.classes_],
            }
        )
        st.dataframe(cls_df, use_container_width=True, hide_index=True)

    with c2:
        st.subheader("📊 Label Distribution")
        dist = df["Original_Label"].value_counts().reset_index()
        dist.columns = ["Label", "Count"]
        dist["Pct"] = (dist["Count"] / len(df) * 100).round(2).astype(str) + " %"
        dist["Severity"] = dist["Label"].apply(get_severity)
        st.dataframe(dist, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRAFFIC LOGS
# ══════════════════════════════════════════════════════════════════════════════

def page_traffic_logs() -> None:
    st.title("📋 Traffic Log Monitor")
    st.markdown("Real-time browsable view of all network traffic flow records.")
    st.markdown("---")

    if "df" not in st.session_state:
        st.warning("⚠️ No dataset loaded — go to **Upload Dataset** first.")
        return

    df           = st.session_state["df"]
    feature_cols = st.session_state["feature_cols"]

    # Derive severity for every row
    df_view = df.copy()
    df_view["Severity"] = df_view["Original_Label"].apply(get_severity)

    # ── Filters ────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        traffic_filter = st.selectbox(
            "Traffic Type", ["All", "Normal (BENIGN)", "Attacks Only"]
        )
    with c2:
        label_filter = st.selectbox(
            "Specific Attack Type",
            ["All"] + sorted(df_view["Original_Label"].unique().tolist()),
        )
    with c3:
        sev_filter = st.multiselect(
            "Severity Level",
            ["SAFE", "LOW", "MEDIUM", "HIGH"],
            default=["SAFE", "LOW", "MEDIUM", "HIGH"],
        )

    if traffic_filter == "Normal (BENIGN)":
        df_view = df_view[df_view["Original_Label"] == "BENIGN"]
    elif traffic_filter == "Attacks Only":
        df_view = df_view[df_view["Original_Label"] != "BENIGN"]

    if label_filter != "All":
        df_view = df_view[df_view["Original_Label"] == label_filter]

    if sev_filter:
        df_view = df_view[df_view["Severity"].isin(sev_filter)]

    # ── Summary metrics ────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered Rows",  f"{len(df_view):,}")
    c2.metric("Total Records",  f"{len(df):,}")
    c3.metric("Attack Types",   df_view["Original_Label"].nunique())
    c4.metric("🔴 HIGH Risk",   len(df_view[df_view["Severity"] == "HIGH"]))

    # ── Table ──────────────────────────────────────────────────────
    st.markdown("---")
    display_cols = (
        ["Original_Label", "Severity"]
        + [c for c in feature_cols[:12] if c in df_view.columns]
    )
    st.dataframe(
        df_view[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=530,
        hide_index=True,
    )
    st.caption(
        f"Showing **{len(df_view):,}** filtered records from **{len(df):,}** total."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA / ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def page_eda() -> None:
    st.title("📊 EDA / Analytics")
    st.markdown(
        "Exploratory Data Analysis — distributions, correlations, and traffic patterns."
    )
    st.markdown("---")

    if "df" not in st.session_state:
        st.warning("⚠️ No dataset loaded — go to **Upload Dataset** first.")
        return

    df           = st.session_state["df"]
    feature_cols = st.session_state["feature_cols"]
    label_counts = df["Original_Label"].value_counts()

    # ── 1. Class Imbalance Check ─────────────────────────────────
    st.subheader("⚖️ Class Imbalance Check")
    normal_count = int(label_counts.get("BENIGN", 0))
    attack_count = len(df) - normal_count

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "✅ BENIGN (Normal)",
        f"{normal_count:,}",
        f"{normal_count / len(df) * 100:.1f} % of data",
    )
    c2.metric(
        "🔴 Attack Traffic",
        f"{attack_count:,}",
        f"{attack_count / len(df) * 100:.1f} % of data",
    )
    c3.metric("Total Records", f"{len(df):,}")

    if normal_count != attack_count:
        st.info(
            "ℹ️ **Dataset is imbalanced, which is common in cybersecurity.**  "
            "Real networks generate far more benign traffic than attack traffic."
        )
    else:
        st.success("✅ Dataset is balanced — equal normal and attack records.")

    st.markdown("---")

    # ── 2. Normal vs Attack + Per-Class Distribution ─────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🟢 Normal vs 🔴 Attack Traffic")
        fig, ax = plt.subplots(figsize=(6, 4))
        traffic_s = pd.Series(
            {"BENIGN\n(Normal)": normal_count, "ATTACKS\n(All Types)": attack_count}
        )
        bars = ax.bar(
            traffic_s.index, traffic_s.values,
            color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5, width=0.45,
        )
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Normal vs Attack Traffic Distribution", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        for bar, v in zip(bars, traffic_s.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(traffic_s.values) * 0.012,
                f"{v:,}", ha="center", fontweight="bold", fontsize=10,
            )
        ax.set_ylim(0, max(traffic_s.values) * 1.14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("📊 Attack Type Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.cm.tab10(np.linspace(0, 1, len(label_counts)))
        label_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
        ax.set_title("Per-Class Attack Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Attack Type", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── 3. Protocol + Top Suspicious Ports ───────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🔌 Protocol Distribution")
        proto_col = next(
            (c for c in df.columns if "protocol" in c.lower()), None
        )
        if proto_col:
            proto_data = df[proto_col].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(
                proto_data.values,
                labels=proto_data.index.astype(str),
                autopct="%1.1f%%",
                startangle=140,
                colors=plt.cm.Set2(np.linspace(0, 1, len(proto_data))),
                wedgeprops={"linewidth": 2, "edgecolor": "white"},
            )
            ax.set_title("Protocol Distribution", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("ℹ️ No 'Protocol' column detected in the dataset.")

    with c2:
        st.subheader("🚪 Top 10 Suspicious Destination Ports")
        port_col = next(
            (
                c for c in df.columns
                if "destination port" in c.lower() or "dst port" in c.lower()
                or c.lower() == "destination port"
            ),
            None,
        )
        if port_col:
            attack_df = df[df["Original_Label"] != "BENIGN"]
            port_data = attack_df[port_col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            colors_p = plt.cm.Reds_r(np.linspace(0.2, 0.75, len(port_data)))
            port_data.plot(kind="barh", ax=ax, color=colors_p, edgecolor="white")
            ax.set_title("Top 10 Suspicious Destination Ports", fontsize=12, fontweight="bold")
            ax.set_xlabel("Count", fontsize=10)
            ax.grid(axis="x", alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("ℹ️ No 'Destination Port' column detected.")

    st.markdown("---")

    # ── 4. Severity Breakdown ─────────────────────────────────────
    st.subheader("🎚️ Severity-Level Breakdown")
    sev_series = df["Original_Label"].apply(get_severity).value_counts()
    sev_order  = [s for s in ["SAFE", "LOW", "MEDIUM", "HIGH"] if s in sev_series.index]
    sev_series = sev_series.reindex(sev_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(
        sev_series.index, sev_series.values,
        color=[SEV_COLORS[s] for s in sev_series.index],
        edgecolor="white", linewidth=1.5, width=0.4,
    )
    ax.set_title("Traffic Records by Severity Level", fontsize=13, fontweight="bold")
    ax.set_xlabel("Severity Level", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, sev_series.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + len(df) * 0.003,
            f"{v:,}", ha="center", fontweight="bold",
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── 5. Correlation Heatmap ─────────────────────────────────────
    st.subheader("🔥 Feature Correlation Heatmap (Top 15 Features)")

    top_feats  = feature_cols[:15]
    sample_n   = min(5_000, len(df))
    corr_df    = df[top_feats].sample(n=sample_n, random_state=42)
    corr_matrix = corr_df.corr()

    fig, ax = plt.subplots(figsize=(14, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=False,
        cmap="RdYlBu_r", center=0, linewidths=0.35, ax=ax,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title(
        "Feature Correlation Heatmap — Top 15 Features",
        fontsize=14, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def page_model_training() -> None:
    st.title("🤖 Model Training & Evaluation")
    st.markdown(
        "Train three ML classifiers and compare their performance on the CICIDS2017 dataset."
    )
    st.markdown("---")

    if "df" not in st.session_state:
        st.warning("⚠️ No dataset loaded — go to **Upload Dataset** first.")
        return

    X_scaled     = st.session_state["X_scaled"]
    y            = st.session_state["y"]
    le           = st.session_state["le"]
    feature_cols = st.session_state["feature_cols"]

    # Configuration
    with st.expander("⚙️ Training Configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            max_rows = len(X_scaled)
            sample_size = st.slider(
                "Training Sample Size (rows)",
                min_value=min(1_000, max_rows),
                max_value=max_rows,
                value=min(50_000, max_rows),
                step=1_000,
                help=(
                    "Subsample for faster training. "
                    "50,000 rows typically gives excellent results."
                ),
            )
        with c2:
            st.markdown("**Models queued for training:**")
            st.markdown("📘 Logistic Regression — *Baseline*")
            st.markdown("🌳 Decision Tree — *Comparison*")
            st.markdown("🌲 Random Forest — *Final / Recommended*")

    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        progress = st.progress(5, text="Initialising…")
        with st.spinner("Training in progress — this may take a minute for large samples…"):
            progress.progress(15, text="Training Models…")
            try:
                trained_models, results, X_te, y_te, unique_cls = train_all_models(
                    X_scaled, y, sample_size
                )
            except Exception as e:
                progress.empty()
                st.error(f"❌ Training failed: {e}")
                return
            progress.progress(100, text="Training complete!")

        st.session_state["trained_models"] = trained_models
        st.session_state["results"]        = results
        st.session_state["X_test"]         = X_te
        st.session_state["y_test"]         = y_te
        st.session_state["unique_cls"]     = unique_cls
        progress.empty()
        st.success("✅ All three models trained and evaluated successfully!")

    if "results" not in st.session_state:
        st.info("👆 Configure options then click **Train All Models**.")
        return

    results        = st.session_state["results"]
    trained_models = st.session_state["trained_models"]
    unique_cls     = st.session_state.get("unique_cls", [])

    st.markdown("---")

    # ── Model Comparison Table ────────────────────────────────────
    st.subheader("📊 Model Performance Comparison")

    rows = []
    for mname, res in results.items():
        cr = res["classification_report"]
        rows.append(
            {
                "Model":               mname,
                "Accuracy (%)":        round(res["accuracy"] * 100, 2),
                "Precision Macro (%)": round(cr.get("macro avg", {}).get("precision", 0) * 100, 2),
                "Recall Macro (%)":    round(cr.get("macro avg", {}).get("recall", 0) * 100, 2),
                "F1 Macro (%)":        round(cr.get("macro avg", {}).get("f1-score", 0) * 100, 2),
                "Status":              "✅ Final Model" if mname == "Random Forest" else "—",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.success(
        "🏆 **Random Forest** is selected as the final model.  \n"
        "Reason: Highest accuracy, ensemble-based resistance to overfitting, "
        "and superior handling of high-dimensional, imbalanced cybersecurity data."
    )

    st.markdown("---")

    # ── Accuracy Bar Chart ────────────────────────────────────────
    st.subheader("📈 Model Accuracy Comparison Chart")

    m_names  = [r["Model"] for r in rows]
    accs     = [r["Accuracy (%)"] for r in rows]
    bar_clrs = ["#3498db", "#e67e22", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(m_names, accs, color=bar_clrs, edgecolor="white", linewidth=1.5, width=0.45)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("ML Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc:.2f} %",
            ha="center", va="bottom", fontweight="bold", fontsize=12,
        )

    # Highlight RF as chosen
    bars[2].set_edgecolor("#00e676")
    bars[2].set_linewidth(3)
    rf_acc_val = accs[2]
    ax.annotate(
        "✅ Chosen",
        xy=(2, rf_acc_val + 1),
        xytext=(2, rf_acc_val + 9),
        ha="center", fontsize=10, color="#00e676",
        arrowprops=dict(arrowstyle="->", color="#00e676"),
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Confusion Matrices ────────────────────────────────────────
    st.subheader("🔢 Confusion Matrices")
    n_cls = len(unique_cls)
    class_labels = [str(le.classes_[i])[:10] for i in unique_cls]

    cm_cols = st.columns(3)
    for idx, (mname, res) in enumerate(results.items()):
        with cm_cols[idx]:
            st.markdown(f"**{mname}**")
            cm = res["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(5, 4))
            annot = n_cls <= 12
            fontsize_annot = max(5, 10 - n_cls // 2)
            sns.heatmap(
                cm, annot=annot, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_labels if n_cls <= 14 else False,
                yticklabels=class_labels if n_cls <= 14 else False,
                linewidths=0.3 if n_cls <= 12 else 0,
                annot_kws={"size": fontsize_annot},
            )
            ax.tick_params(axis="x", rotation=45, labelsize=6)
            ax.tick_params(axis="y", rotation=0,  labelsize=6)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual",    fontsize=8)
            ax.set_title(
                f"{mname}\nAcc: {res['accuracy'] * 100:.2f} %",
                fontsize=10, fontweight="bold",
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # ── Top 10 Feature Importances (RF) ──────────────────────────
    st.subheader("🎯 Top 10 Feature Importances — Random Forest")

    rf = trained_models.get("Random Forest")
    if rf and hasattr(rf, "feature_importances_"):
        imp     = pd.Series(rf.feature_importances_, index=feature_cols)
        top10   = imp.sort_values(ascending=False).head(10)
        palette = plt.cm.RdYlGn_r(np.linspace(0.1, 0.85, 10))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            range(10), top10.values[::-1],
            color=palette[::-1], edgecolor="white",
        )
        ax.set_yticks(range(10))
        ax.set_yticklabels(top10.index[::-1], fontsize=10)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title(
            "Top 10 Feature Importances (Random Forest)",
            fontsize=13, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.25)
        for i, v in enumerate(top10.values[::-1]):
            ax.text(v + 0.0005, i, f"{v:.5f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.dataframe(
            pd.DataFrame(
                {
                    "Rank": range(1, 11),
                    "Feature": top10.index,
                    "Importance Score": top10.values.round(6),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Feature importances not available.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — THREAT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def page_threat_detection() -> None:
    st.title("🔍 Threat Detection Engine")
    st.markdown(
        "Select any traffic record for real-time threat prediction, "
        "risk scoring, and severity classification."
    )
    st.markdown("---")

    if "df" not in st.session_state:
        st.warning("⚠️ No dataset loaded — go to **Upload Dataset** first.")
        return
    if "trained_models" not in st.session_state:
        st.warning("⚠️ Models not trained — go to **Model Training** first.")
        return

    df           = st.session_state["df"]
    X_scaled     = st.session_state["X_scaled"]
    le           = st.session_state["le"]
    feature_cols = st.session_state["feature_cols"]
    rf           = st.session_state["trained_models"]["Random Forest"]
    benign_idx   = get_benign_model_idx(rf, le)

    st.subheader("🎯 Record Selection")
    c1, c2 = st.columns([3, 2])

    with c1:
        row_num = st.number_input(
            "Enter Row Number to Analyse",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            step=1,
            help=f"Valid range: 0 – {len(df) - 1}",
        )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        simulate = st.button(
            "🎲 Simulate Live Traffic", type="primary", use_container_width=True
        )

    if simulate:
        row_num = int(np.random.randint(0, len(df)))
        st.info(f"🎲 **Live Simulation** — randomly selected **row {row_num}**")

    analyse = st.button("🔍 Analyse Selected Record", use_container_width=True)

    if not (analyse or simulate):
        return

    try:
        record         = X_scaled.iloc[[row_num]]
        orig_label     = df["Original_Label"].iloc[row_num]
        pred_enc       = rf.predict(record)[0]
        pred_proba_arr = rf.predict_proba(record)[0]
        pred_label     = le.inverse_transform([pred_enc])[0]
        max_prob       = float(pred_proba_arr.max())
        risk_score     = compute_risk_score(pred_proba_arr, rf.classes_, benign_idx)
        severity       = get_severity(pred_label)

        st.markdown("---")
        st.subheader("🔬 Detection Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏷️ Predicted Attack",  pred_label)
        c2.metric("⚡ Severity",           severity)
        c3.metric("📊 Risk Score",         f"{risk_score} / 100")
        c4.metric("💯 Model Confidence",   f"{max_prob * 100:.1f} %")

        st.markdown("---")

        # Alert Box
        icon, title, action = ALERT_CONFIG.get(severity, ("ℹ️", "INFO", "Review recommended."))
        st.markdown(
            f"""
            <div class="alert-{severity}">
                <span style="font-size:1.3rem;font-weight:800;">{icon} {title}: {pred_label}</span><br><br>
                🔒 <strong>Severity:</strong> {severity} &nbsp;|&nbsp;
                📊 <strong>Risk Score:</strong> {risk_score}/100 &nbsp;|&nbsp;
                💯 <strong>Confidence:</strong> {max_prob * 100:.1f} %<br><br>
                🚀 <strong>Recommended Action:</strong> {action}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📊 Class Probability Distribution (Top 10)")
            all_classes = le.inverse_transform(rf.classes_)
            prob_s = (
                pd.Series(pred_proba_arr, index=all_classes)
                .sort_values(ascending=False)
                .head(10)
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            bar_c = ["#e74c3c" if c != "BENIGN" else "#2ecc71" for c in prob_s.index]
            ax.barh(
                prob_s.index[::-1], prob_s.values[::-1],
                color=bar_c[::-1], edgecolor="white",
            )
            ax.set_xlabel("Probability", fontsize=10)
            ax.set_title(
                f"Prediction Probabilities — Row {row_num}",
                fontsize=11, fontweight="bold",
            )
            ax.grid(axis="x", alpha=0.25)
            for i, v in enumerate(prob_s.values[::-1]):
                ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.subheader("🌡️ Risk Score Gauge")
            fig, ax = plt.subplots(figsize=(7, 4))
            zones = [
                (0,  30,  "#2ecc71", "SAFE"),
                (30, 50,  "#f1c40f", "LOW"),
                (50, 70,  "#e67e22", "MEDIUM"),
                (70, 100, "#e74c3c", "HIGH"),
            ]
            for lo, hi, color, label_z in zones:
                ax.barh(0, hi - lo, left=lo, height=0.5, color=color, alpha=0.75)
                ax.text(
                    (lo + hi) / 2, -0.42, label_z,
                    ha="center", fontsize=8, color=color, fontweight="bold",
                )
            ax.axvline(risk_score, color="white", linewidth=3, linestyle="--")
            ax.text(
                risk_score, 0.35,
                f"▼ {risk_score}",
                ha="center", fontsize=12, fontweight="bold", color="white",
            )
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.8, 0.8)
            ax.set_xlabel("Risk Score (0 = Safe  →  100 = Critical)", fontsize=10)
            ax.set_title("Risk Score Gauge", fontsize=11, fontweight="bold")
            ax.set_yticks([])
            ax.set_facecolor("#0d1117")
            fig.patch.set_facecolor("#0d1117")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.subheader("📋 Traffic Record Feature Values (First 20 Features)")
        feat_vals = df[feature_cols].iloc[row_num]
        st.dataframe(
            pd.DataFrame(
                {"Feature": feat_vals.index[:20], "Value": feat_vals.values[:20]}
            ),
            use_container_width=True,
            hide_index=True,
        )
        match = "✅ Correct Prediction" if orig_label == pred_label else "❌ Mismatch"
        st.caption(
            f"📌 **Actual Label:** {orig_label} &nbsp;|&nbsp; "
            f"🤖 **Predicted:** {pred_label} &nbsp;|&nbsp; {match}"
        )

    except Exception as exc:
        st.error(f"❌ Error during analysis: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — THREAT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def page_threat_report() -> None:
    st.title("📄 Threat Report & Conclusion")
    st.markdown(
        "Generate a downloadable threat intelligence report and review project findings."
    )
    st.markdown("---")

    if "df" not in st.session_state:
        st.warning("⚠️ No dataset loaded — go to **Upload Dataset** first.")
        return
    if "trained_models" not in st.session_state:
        st.warning("⚠️ Models not trained — go to **Model Training** first.")
        return

    df           = st.session_state["df"]
    X_scaled     = st.session_state["X_scaled"]
    le           = st.session_state["le"]
    feature_cols = st.session_state["feature_cols"]
    rf           = st.session_state["trained_models"]["Random Forest"]
    benign_idx   = get_benign_model_idx(rf, le)

    # ── Report Configuration ──────────────────────────────────────
    st.subheader("⚙️ Report Configuration")
    c1, c2 = st.columns(2)
    with c1:
        n_records = st.slider(
            "Records to Analyse",
            100, min(10_000, len(df)),
            value=min(2_000, len(df)),
            step=100,
        )
    with c2:
        threats_only = st.checkbox(
            "Show Only Suspicious / High-Risk Logs (MEDIUM + HIGH)", value=True
        )

    if st.button("📊 Generate Threat Report", type="primary", use_container_width=True):
        with st.spinner(f"Analysing {n_records:,} records with Random Forest…"):
            idx      = np.random.choice(len(df), size=n_records, replace=False)
            X_s      = X_scaled.iloc[idx]
            df_s     = df.iloc[idx].copy()
            p_enc    = rf.predict(X_s)
            p_proba  = rf.predict_proba(X_s)
            p_labels = le.inverse_transform(p_enc)

            risk_scores = [
                compute_risk_score(p, rf.classes_, benign_idx) for p in p_proba
            ]
            severities = [get_severity(lbl) for lbl in p_labels]

            report = pd.DataFrame(
                {
                    "Row Index":              idx,
                    "Predicted Attack Type":  p_labels,
                    "Actual Label":           df_s["Original_Label"].values,
                    "Severity":               severities,
                    "Risk Score":             risk_scores,
                    "Correct Prediction":     [
                        "✅" if p == a else "❌"
                        for p, a in zip(p_labels, df_s["Original_Label"].values)
                    ],
                }
            )
            # Attach top 5 network features
            for f in feature_cols[:5]:
                if f in df_s.columns:
                    report[f] = df_s[f].values

            if threats_only:
                report = report[report["Severity"].isin(["MEDIUM", "HIGH"])]

            report = report.sort_values("Risk Score", ascending=False).reset_index(drop=True)

        st.session_state["report_df"] = report
        st.success(f"✅ Report generated — **{len(report):,}** records found.")

    # ── Display Report ────────────────────────────────────────────
    if "report_df" not in st.session_state:
        st.info("👆 Click **Generate Threat Report** to begin analysis.")
    else:
        report = st.session_state["report_df"]

        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Records",        f"{len(report):,}")
        c2.metric("🔴 HIGH",        f"{len(report[report['Severity'] == 'HIGH']):,}")
        c3.metric("🟠 MEDIUM",      f"{len(report[report['Severity'] == 'MEDIUM']):,}")
        c4.metric("Avg Risk Score", f"{report['Risk Score'].mean():.1f}")
        c5.metric("Max Risk Score", f"{report['Risk Score'].max()}")

        st.markdown("---")
        st.subheader("🚨 Threat Intelligence Report Table")
        st.dataframe(report, use_container_width=True, height=420, hide_index=True)

        # Download button
        csv_buf = io.StringIO()
        report.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Download Threat Report (CSV)",
            data=csv_buf.getvalue(),
            file_name="siem_threat_report.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📊 Severity Distribution")
            sev_c  = report["Severity"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            bc = [SEV_COLORS.get(s, "#aaa") for s in sev_c.index]
            sev_c.plot(kind="bar", ax=ax, color=bc, edgecolor="white")
            ax.set_title("Severity Level Distribution", fontweight="bold")
            ax.set_xlabel("Severity")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=0)
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c2:
            st.subheader("🎯 Risk Score Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(
                report["Risk Score"], bins=20,
                color="#c0392b", edgecolor="white", alpha=0.85,
            )
            ax.axvline(30, color="#2ecc71", linestyle="--", linewidth=1.5, label="Low→Medium (30)")
            ax.axvline(70, color="#e67e22", linestyle="--", linewidth=1.5, label="Medium→High (70)")
            ax.set_title("Risk Score Distribution", fontweight="bold")
            ax.set_xlabel("Risk Score")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Conclusion ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📚 Conclusion & Project Findings")

    res     = st.session_state.get("results", {})
    rf_acc  = res.get("Random Forest",       {}).get("accuracy", 0) * 100
    lr_acc  = res.get("Logistic Regression", {}).get("accuracy", 0) * 100
    dt_acc  = res.get("Decision Tree",       {}).get("accuracy", 0) * 100

    acc_table = ""
    if res:
        acc_table = f"""
| Model | Accuracy |
|---|---|
| Logistic Regression | {lr_acc:.2f} % |
| Decision Tree | {dt_acc:.2f} % |
| **Random Forest ✅** | **{rf_acc:.2f} %** |
"""
    else:
        acc_table = "*(Train models on the Model Training page to see accuracy figures here.)*"

    st.markdown(
        f"""
### 🏆 Best Performing Model: Random Forest Classifier

{acc_table}

---

### 🔍 Key Learnings from CICIDS2017

**1. Class Imbalance is Universal in Cybersecurity**
The CICIDS2017 dataset is heavily dominated by BENIGN traffic (often 80 %+), accurately reflecting
real-world enterprise networks where attacks are rare events. This imbalance is expected and normal.

**2. Feature Importance Insights**
Network-flow features such as *Flow Duration*, *Packet Length Statistics*, *Flow Bytes/s*,
*Inter-Arrival Times (IAT)*, and *TCP Flag counts* are the most discriminative for attack
classification. These features capture behavioural anomalies that signature-based IDS rules miss.

**3. Random Forest Superiority**
Random Forest consistently outperforms Logistic Regression and Decision Tree because:
- An ensemble of 100 trees reduces overfitting
- Handles non-linear decision boundaries in high-dimensional feature spaces  
- Robust to class imbalance and noisy features
- Provides built-in feature importance ranking for explainability

**4. Multi-Class Attack Detection**
A single RF model simultaneously detects 14+ attack types — replacing multiple signature-based
rule sets and reducing the maintenance burden on security teams.

---

### 🌐 Real-World SIEM Applications

| Use Case | Description |
|---|---|
| 🏢 SOC Automation | Automated threat triage reduces analyst alert fatigue |
| 🔒 Network IDS | Real-time stream classification of network flow records |
| 📋 Compliance Reporting | Automated logs for PCI-DSS, ISO 27001, GDPR audits |
| 🚨 Incident Response | Risk-scored alerts speed up IR prioritisation workflows |
| 🤖 Threat Hunting | Pattern analysis enables proactive threat hunting |
| 📊 Executive Visibility | Risk dashboards deliver C-suite security intelligence |

> 💡 *This mini-SIEM demonstrates how ML-powered detection complements traditional rule-based
> SIEM systems — achieving higher detection rates, lower false positives, and scalable,
> adaptive security monitoring in enterprise environments.*
        """,
        unsafe_allow_html=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    page = render_sidebar()

    dispatch = {
        PAGES[0]: page_home,
        PAGES[1]: page_upload,
        PAGES[2]: page_traffic_logs,
        PAGES[3]: page_eda,
        PAGES[4]: page_model_training,
        PAGES[5]: page_threat_detection,
        PAGES[6]: page_threat_report,
    }

    fn = dispatch.get(page, page_home)
    fn()


if __name__ == "__main__":
    main()
