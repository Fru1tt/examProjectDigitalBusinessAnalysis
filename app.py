"""Professional dashboard for channel-preference decision support."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "shopping_behavior_clean.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "modeling_summary.json"
METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "model_metrics.csv"
TOP_DRIVERS_PATH = PROJECT_ROOT / "outputs" / "tables" / "executive_top_drivers.csv"
TARGET_COLUMN = "shopping_preference"

CATEGORY_COLORS = {
    "online": "#0B8F8C",
    "store": "#2563EB",
    "hybrid": "#D97706",
}

BASE_INPUT_FEATURES = [
    "age",
    "monthly_income",
    "daily_internet_hours",
    "smartphone_usage_years",
    "social_media_hours",
    "online_payment_trust_score",
    "tech_savvy_score",
    "monthly_online_orders",
    "monthly_store_visits",
    "avg_online_spend",
    "avg_store_spend",
    "discount_sensitivity",
    "return_frequency",
    "avg_delivery_days",
    "delivery_fee_sensitivity",
    "free_return_importance",
    "product_availability_online",
    "impulse_buying_score",
    "need_touch_feel_score",
    "brand_loyalty_score",
    "environmental_awareness",
    "time_pressure_level",
    "gender",
    "city_tier",
]

ENGINEERED_FEATURES = [
    "total_monthly_orders",
    "online_order_share",
    "total_avg_spend",
    "online_spend_share",
    "digital_engagement_score",
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Manrope', 'Avenir Next', 'Segoe UI', sans-serif;
        }
        .stApp {
            background:
              radial-gradient(circle at 90% 0%, #e0f2fe 0%, transparent 30%),
              radial-gradient(circle at 0% 100%, #dcfce7 0%, transparent 30%),
              #f8fafc;
        }
        .hero {
            background: linear-gradient(120deg, #0f172a, #1e293b);
            border-radius: 14px;
            padding: 20px 24px;
            color: #f8fafc;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.22);
            margin-bottom: 14px;
        }
        .hero h1 {
            margin: 0;
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: 0.2px;
        }
        .hero p {
            margin: 8px 0 0 0;
            color: #cbd5e1;
            font-size: 0.95rem;
        }
        .metric-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 14px 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.07);
            min-height: 118px;
        }
        .metric-title {
            color: #475569;
            font-size: 0.78rem;
            letter-spacing: 0.4px;
            text-transform: uppercase;
            margin-bottom: 6px;
            font-weight: 700;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.45rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.2;
        }
        .metric-note {
            color: #64748b;
            margin-top: 6px;
            font-size: 0.84rem;
        }
        .rec-card {
            border-radius: 12px;
            padding: 14px 16px;
            color: #0f172a;
            border: 1px solid #dbeafe;
            background: #f8fbff;
        }
        .legend-wrap {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 8px;
            margin-bottom: 12px;
        }
        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 5px 10px;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            background: #ffffff;
            font-size: 0.84rem;
            color: #334155;
            font-weight: 600;
        }
        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }
        .section-title {
            font-weight: 800;
            color: #0f172a;
            margin-top: 6px;
            margin-bottom: 8px;
            font-size: 1.12rem;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            background: #ffffff;
            box-shadow: 0 4px 14px rgba(30, 41, 59, 0.08);
        }
        div[data-testid="stExpander"] details {
            border-radius: 12px;
            overflow: hidden;
        }
        div[data-testid="stExpander"] details summary {
            background: linear-gradient(90deg, #1e40af, #2563eb);
            padding: 0.55rem 0.8rem;
        }
        div[data-testid="stExpander"] details summary p {
            color: #ffffff !important;
            font-weight: 700;
            letter-spacing: 0.15px;
        }
        div[data-testid="stExpander"] details summary svg {
            fill: #ffffff !important;
        }
        div[data-testid="stExpander"] details[open] summary {
            border-bottom: 1px solid #bfdbfe;
        }
        div[data-testid="stExpander"] .stSlider label p {
            color: #0f172a !important;
            font-weight: 700 !important;
        }
        div[data-testid="stExpander"] .stSlider div[data-baseweb="slider"] + div p {
            color: #334155 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stExpander"] p {
            color: #1e293b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PROCESSED_PATH}. Run scripts/prepare_data.py first."
        )
    return pd.read_csv(PROCESSED_PATH)


@st.cache_data
def load_model_summary() -> dict[str, object]:
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()


@st.cache_data
def load_top_drivers() -> pd.DataFrame:
    if TOP_DRIVERS_PATH.exists():
        return pd.read_csv(TOP_DRIVERS_PATH)
    return pd.DataFrame()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


@st.cache_resource
def train_prediction_pipeline(df: pd.DataFrame) -> Pipeline:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(str)
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def choose_hybrid_threshold(summary: dict[str, object]) -> float:
    thresholds = summary.get("hybrid_thresholds", {})
    if isinstance(thresholds, dict):
        tuned = thresholds.get("logistic_regression_hybrid_tuned")
        if isinstance(tuned, (int, float)):
            return float(tuned)
    return 0.76


def default_value(df: pd.DataFrame, column: str) -> float | int | str:
    if pd.api.types.is_numeric_dtype(df[column]):
        return float(df[column].median())
    mode = df[column].mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else str(df[column].dropna().iloc[0])


def build_default_profile(df: pd.DataFrame) -> dict[str, object]:
    return {col: default_value(df, col) for col in BASE_INPUT_FEATURES}


def engineer_features(profile: dict[str, object]) -> dict[str, object]:
    online_orders = float(profile["monthly_online_orders"])
    store_visits = float(profile["monthly_store_visits"])
    online_spend = float(profile["avg_online_spend"])
    store_spend = float(profile["avg_store_spend"])
    total_orders = online_orders + store_visits
    total_spend = online_spend + store_spend

    profile["total_monthly_orders"] = total_orders
    profile["online_order_share"] = online_orders / total_orders if total_orders > 0 else 0.0
    profile["total_avg_spend"] = total_spend
    profile["online_spend_share"] = online_spend / total_spend if total_spend > 0 else 0.0
    profile["digital_engagement_score"] = float(
        np.mean(
            [
                float(profile["daily_internet_hours"]),
                float(profile["social_media_hours"]),
                float(profile["tech_savvy_score"]),
                float(profile["online_payment_trust_score"]),
            ]
        )
    )
    return profile


def format_category(value: str) -> str:
    return value.replace("_", " ").title()


def digital_susceptibility_index(row: dict[str, object]) -> float:
    score = (
        0.18 * ((float(row["daily_internet_hours"]) - 1.0) / 11.0)
        + 0.12 * (float(row["social_media_hours"]) / 6.0)
        + 0.17 * (float(row["tech_savvy_score"]) / 10.0)
        + 0.13 * (float(row["online_payment_trust_score"]) / 10.0)
        + 0.20 * float(row["online_order_share"])
        + 0.20 * float(row["online_spend_share"])
    )
    return float(np.clip(score, 0.0, 1.0) * 100.0)


def channel_flexibility_index(probability_map: dict[str, float]) -> float:
    probs = np.array(
        [
            probability_map.get("online", 0.0),
            probability_map.get("store", 0.0),
            probability_map.get("hybrid", 0.0),
        ],
        dtype=float,
    )
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()
    entropy = -np.sum(probs * np.log(probs)) / np.log(3.0)
    return float(np.clip(entropy, 0.0, 1.0) * 100.0)


def build_adaptive_strategy(
    *,
    predicted: str,
    probability_map: dict[str, float],
    row: dict[str, object],
) -> dict[str, object]:
    digital_index = digital_susceptibility_index(row)
    flexibility_index = channel_flexibility_index(probability_map)

    if predicted == "store":
        if probability_map["online"] >= 0.22 or digital_index >= 60:
            strategy = "Store-First, Digitally Activated"
            action = (
                "Keep in-store as the main channel, but use digital ads to drive footfall "
                "(location-based ads, click-to-store, local inventory highlights)."
            )
        elif probability_map["hybrid"] >= 0.20 or flexibility_index >= 55:
            strategy = "Store-First with Omni Bridge"
            action = (
                "Keep store priority, but build cross-channel support (reserve online, pickup in store, "
                "consistent offers across channels)."
            )
        else:
            strategy = "Store-Centric"
            action = (
                "Focus budget on in-store promotions, product experience, and local retail conversion."
            )
    elif predicted == "online":
        if probability_map["store"] >= 0.24 and flexibility_index >= 50:
            strategy = "Online-First with Store Backup"
            action = (
                "Run digital-first campaigns, while supporting conversion with nearest-store options "
                "and pickup/return convenience."
            )
        else:
            strategy = "Online-Centric"
            action = (
                "Prioritize digital channels: paid social/search, app retention, and checkout optimization."
            )
    else:
        strategy = "Omni-Channel Priority"
        action = (
            "Coordinate online and store channels: unified promotions, cross-channel loyalty, "
            "and seamless switch between browsing and buying."
        )

    return {
        "strategy": strategy,
        "action": action,
        "digital_index": round(digital_index, 1),
        "flexibility_index": round(flexibility_index, 1),
    }


def behavior_signals(row: dict[str, object], probability_map: dict[str, float]) -> list[str]:
    signals: list[str] = []
    if float(row["daily_internet_hours"]) >= 8 or float(row["social_media_hours"]) >= 4:
        signals.append("High daily digital exposure suggests stronger online ad reach.")
    if float(row["online_order_share"]) >= 0.65 or float(row["online_spend_share"]) >= 0.6:
        signals.append("Online purchase pattern indicates responsiveness to digital campaigns.")
    if float(row["need_touch_feel_score"]) >= 7:
        signals.append("Strong touch-and-feel need supports physical store conversion tactics.")
    if float(row["online_payment_trust_score"]) >= 7 and float(row["tech_savvy_score"]) >= 7:
        signals.append("High trust and tech comfort support online and hybrid adoption.")
    if probability_map["hybrid"] >= 0.2:
        signals.append("Non-trivial hybrid probability supports omni-channel experimentation.")
    if not signals:
        signals.append("No single dominant behavioral driver detected; use balanced testing.")
    return signals[:4]


def render_metric_card(title: str, value: str, note: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top: 4px solid {color};">
          <div class="metric-title">{title}</div>
          <p class="metric-value">{value}</p>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_legend() -> None:
    legend = "".join(
        [
            (
                f"<span class='legend-item'><span class='legend-dot' "
                f"style='background:{CATEGORY_COLORS[key]};'></span>{format_category(key)}</span>"
            )
            for key in ["online", "store", "hybrid"]
        ]
    )
    st.markdown(f"<div class='legend-wrap'>{legend}</div>", unsafe_allow_html=True)


def build_profile_input(df: pd.DataFrame) -> dict[str, object]:
    defaults = build_default_profile(df)
    if "profile" not in st.session_state:
        st.session_state["profile"] = defaults.copy()

    profile = st.session_state["profile"]
    with st.sidebar:
        st.markdown("## Customer Profile Input")
        st.caption("Enter one customer profile to predict the preferred shopping channel.")

        with st.form("profile_form", clear_on_submit=False):
            st.markdown("### Demographics")
            age = st.slider("Age", 18, 79, int(profile.get("age", defaults["age"])))
            monthly_income = st.slider(
                "Monthly income",
                int(df["monthly_income"].min()),
                int(df["monthly_income"].max()),
                int(profile.get("monthly_income", defaults["monthly_income"])),
                step=500,
            )
            gender = st.selectbox(
                "Gender",
                ["male", "female", "other"],
                index=["male", "female", "other"].index(
                    str(profile.get("gender", defaults["gender"]))
                ),
            )
            city_tier = st.selectbox(
                "City tier",
                ["tier_1", "tier_2", "tier_3"],
                index=["tier_1", "tier_2", "tier_3"].index(
                    str(profile.get("city_tier", defaults["city_tier"]))
                ),
            )

            st.markdown("### Digital Behavior")
            daily_internet_hours = st.slider(
                "Daily internet hours", 1.0, 12.0, float(profile.get("daily_internet_hours", 6.0)), 0.1
            )
            smartphone_usage_years = st.slider(
                "Smartphone usage years", 1, 14, int(profile.get("smartphone_usage_years", 8))
            )
            social_media_hours = st.slider(
                "Social media hours", 0.0, 6.0, float(profile.get("social_media_hours", 2.5)), 0.1
            )
            online_payment_trust_score = st.slider(
                "Online payment trust score (1-10)",
                1,
                10,
                int(profile.get("online_payment_trust_score", 6)),
            )
            tech_savvy_score = st.slider(
                "Tech savvy score (1-10)",
                1,
                10,
                int(profile.get("tech_savvy_score", 6)),
            )

            st.markdown("### Shopping Behavior")
            monthly_online_orders = st.slider(
                "Monthly online orders",
                0,
                49,
                int(profile.get("monthly_online_orders", 8)),
            )
            monthly_store_visits = st.slider(
                "Monthly store visits",
                0,
                19,
                int(profile.get("monthly_store_visits", 7)),
            )
            avg_online_spend = st.slider(
                "Avg online spend",
                int(df["avg_online_spend"].min()),
                int(df["avg_online_spend"].max()),
                int(profile.get("avg_online_spend", defaults["avg_online_spend"])),
                step=500,
            )
            avg_store_spend = st.slider(
                "Avg store spend",
                int(df["avg_store_spend"].min()),
                int(df["avg_store_spend"].max()),
                int(profile.get("avg_store_spend", defaults["avg_store_spend"])),
                step=500,
            )
            return_frequency = st.slider(
                "Return frequency",
                0,
                9,
                int(profile.get("return_frequency", 3)),
            )
            avg_delivery_days = st.slider(
                "Average delivery days",
                1,
                7,
                int(profile.get("avg_delivery_days", 3)),
            )

            st.markdown("### Preference and Attitude Scores")
            discount_sensitivity = st.slider(
                "Discount sensitivity (1-10)",
                1,
                10,
                int(profile.get("discount_sensitivity", 5)),
            )
            delivery_fee_sensitivity = st.slider(
                "Delivery fee sensitivity (1-10)",
                1,
                10,
                int(profile.get("delivery_fee_sensitivity", 5)),
            )
            free_return_importance = st.slider(
                "Free return importance (1-10)",
                1,
                10,
                int(profile.get("free_return_importance", 5)),
            )
            product_availability_online = st.slider(
                "Product availability online (1-10)",
                1,
                10,
                int(profile.get("product_availability_online", 5)),
            )
            impulse_buying_score = st.slider(
                "Impulse buying score (1-10)",
                1,
                10,
                int(profile.get("impulse_buying_score", 5)),
            )
            need_touch_feel_score = st.slider(
                "Need touch and feel score (1-10)",
                1,
                10,
                int(profile.get("need_touch_feel_score", 5)),
            )
            brand_loyalty_score = st.slider(
                "Brand loyalty score (1-10)",
                1,
                10,
                int(profile.get("brand_loyalty_score", 5)),
            )
            environmental_awareness = st.slider(
                "Environmental awareness (1-10)",
                1,
                10,
                int(profile.get("environmental_awareness", 5)),
            )
            time_pressure_level = st.slider(
                "Time pressure level (1-10)",
                1,
                10,
                int(profile.get("time_pressure_level", 5)),
            )

            submitted = st.form_submit_button("Update Prediction")

        if submitted:
            st.session_state["profile"] = {
                "age": age,
                "monthly_income": monthly_income,
                "daily_internet_hours": daily_internet_hours,
                "smartphone_usage_years": smartphone_usage_years,
                "social_media_hours": social_media_hours,
                "online_payment_trust_score": online_payment_trust_score,
                "tech_savvy_score": tech_savvy_score,
                "monthly_online_orders": monthly_online_orders,
                "monthly_store_visits": monthly_store_visits,
                "avg_online_spend": avg_online_spend,
                "avg_store_spend": avg_store_spend,
                "discount_sensitivity": discount_sensitivity,
                "return_frequency": return_frequency,
                "avg_delivery_days": avg_delivery_days,
                "delivery_fee_sensitivity": delivery_fee_sensitivity,
                "free_return_importance": free_return_importance,
                "product_availability_online": product_availability_online,
                "impulse_buying_score": impulse_buying_score,
                "need_touch_feel_score": need_touch_feel_score,
                "brand_loyalty_score": brand_loyalty_score,
                "environmental_awareness": environmental_awareness,
                "time_pressure_level": time_pressure_level,
                "gender": gender,
                "city_tier": city_tier,
            }

    return st.session_state["profile"]


def predict_customer(
    *,
    pipeline: Pipeline,
    profile: dict[str, object],
    threshold: float,
) -> tuple[str, dict[str, float], float, dict[str, object], str, float]:
    row = engineer_features(profile.copy())
    ordered_cols = BASE_INPUT_FEATURES + ENGINEERED_FEATURES
    input_df = pd.DataFrame([row])[ordered_cols]

    probabilities = pipeline.predict_proba(input_df)[0]
    class_labels = pipeline.named_steps["model"].classes_
    probability_map = {
        label: float(probabilities[idx])
        for idx, label in enumerate(class_labels)
    }
    for label in ["online", "store", "hybrid"]:
        probability_map.setdefault(label, 0.0)

    if probability_map["hybrid"] >= threshold:
        predicted = "hybrid"
    else:
        predicted = "online" if probability_map["online"] >= probability_map["store"] else "store"

    confidence = probability_map[predicted]
    ranked = sorted(probability_map.items(), key=lambda x: x[1], reverse=True)
    secondary_channel, secondary_probability = ranked[1]
    return predicted, probability_map, confidence, row, secondary_channel, secondary_probability


def clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def build_scenario_profile(
    *,
    base_profile: dict[str, object],
    deltas: dict[str, float],
    df: pd.DataFrame,
) -> dict[str, object]:
    profile = base_profile.copy()
    for feature, delta in deltas.items():
        current = float(profile[feature])
        lower = float(df[feature].min())
        upper = float(df[feature].max())
        if feature in {"monthly_online_orders", "monthly_store_visits"}:
            profile[feature] = int(round(clamp(current + delta, lower, upper)))
        else:
            profile[feature] = clamp(current + delta, lower, upper)
    return profile


def scenario_shift_summary(
    *,
    base_pred: str,
    base_strategy: str,
    scenario_pred: str,
    scenario_strategy: str,
) -> str:
    if base_pred != scenario_pred:
        return (
            f"Preference changed from {format_category(base_pred)} to "
            f"{format_category(scenario_pred)}."
        )
    if base_strategy != scenario_strategy:
        return (
            f"Preference stayed {format_category(base_pred)}, but strategy shifted from "
            f"'{base_strategy}' to '{scenario_strategy}'."
        )
    return "No major strategy shift detected; current profile remains stable under this scenario."


def main() -> None:
    st.set_page_config(
        page_title="Channel Preference Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )
    inject_styles()

    df = load_data()
    summary = load_model_summary()
    metrics_df = load_model_metrics()
    top_drivers_df = load_top_drivers()
    pipeline = train_prediction_pipeline(df)
    threshold = choose_hybrid_threshold(summary)

    st.markdown(
        """
        <div class="hero">
          <h1>Customer Channel Preference Dashboard</h1>
          <p>Decision-support interface for predicting whether a customer is likely to prefer
          online shopping, physical store shopping, or a hybrid channel strategy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_legend()

    profile = build_profile_input(df)
    predicted, probability_map, confidence, engineered_row, secondary_channel, secondary_probability = predict_customer(
        pipeline=pipeline,
        profile=profile,
        threshold=threshold,
    )
    adaptive = build_adaptive_strategy(
        predicted=predicted,
        probability_map=probability_map,
        row=engineered_row,
    )
    signals = behavior_signals(engineered_row, probability_map)

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        render_metric_card(
            "Predicted Preference",
            format_category(predicted),
            "Primary channel recommendation",
            CATEGORY_COLORS[predicted],
        )
    with col_b:
        render_metric_card(
            "Confidence",
            f"{confidence:.1%}",
            "Model confidence for selected class",
            CATEGORY_COLORS[predicted],
        )
    with col_c:
        render_metric_card(
            "Secondary Channel",
            format_category(secondary_channel),
            f"Secondary likelihood: {secondary_probability:.1%}",
            CATEGORY_COLORS[secondary_channel],
        )
    with col_d:
        render_metric_card(
            "Digital Susceptibility",
            f"{adaptive['digital_index']:.1f}/100",
            f"Channel flexibility: {adaptive['flexibility_index']:.1f}/100",
            "#0f172a",
        )

    left, right = st.columns([1.6, 1.1])
    with left:
        st.markdown("<div class='section-title'>Prediction Probability by Category</div>", unsafe_allow_html=True)
        prob_df = pd.DataFrame(
            {
                "category": ["online", "store", "hybrid"],
                "probability": [
                    probability_map["online"],
                    probability_map["store"],
                    probability_map["hybrid"],
                ],
            }
        )
        fig = px.bar(
            prob_df,
            x="category",
            y="probability",
            color="category",
            color_discrete_map=CATEGORY_COLORS,
            text=prob_df["probability"].map(lambda x: f"{x:.1%}"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            yaxis=dict(range=[0, 1]),
            xaxis_title="Category",
            yaxis_title="Probability",
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("<div class='section-title'>Adaptive Recommendation</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="rec-card" style="border-left: 6px solid {CATEGORY_COLORS[predicted]};">
              <strong>Strategy Mode: {adaptive['strategy']}</strong><br><br>
              {adaptive['action']}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown("**Behavior Signals Used**")
        for signal in signals:
            st.markdown(f"- {signal}")

        st.markdown("")
        st.caption(
            f"Hybrid threshold in model decision rule: {threshold:.2f}. "
            "Prediction label remains model-driven; strategy layer adapts tactics."
        )
        st.markdown("**Input Snapshot**")
        snapshot_df = pd.DataFrame(
            {
                "Feature": ["Age", "Monthly Income", "Gender", "City Tier"],
                "Value": [
                    str(profile["age"]),
                    str(profile["monthly_income"]),
                    str(profile["gender"]),
                    str(profile["city_tier"]),
                ],
            }
        )
        st.dataframe(snapshot_df, width="stretch", hide_index=True)

    st.markdown("<div class='section-title'>Scenario Compare</div>", unsafe_allow_html=True)
    with st.expander("Test how behavior changes shift predictions and strategy", expanded=False):
        st.caption(
            "Adjust key behavior variables to compare baseline vs scenario outcomes."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            delta_internet = st.slider(
                "Daily internet hours delta",
                -4.0,
                4.0,
                0.0,
                0.1,
                key="sc_delta_internet",
            )
            delta_social = st.slider(
                "Social media hours delta",
                -3.0,
                3.0,
                0.0,
                0.1,
                key="sc_delta_social",
            )
            delta_tech = st.slider(
                "Tech savvy score delta",
                -4,
                4,
                0,
                1,
                key="sc_delta_tech",
            )
        with c2:
            delta_online_orders = st.slider(
                "Monthly online orders delta",
                -20,
                20,
                0,
                1,
                key="sc_delta_online_orders",
            )
            delta_store_visits = st.slider(
                "Monthly store visits delta",
                -10,
                10,
                0,
                1,
                key="sc_delta_store_visits",
            )
            delta_online_spend = st.slider(
                "Avg online spend delta",
                -50000,
                50000,
                0,
                500,
                key="sc_delta_online_spend",
            )
        with c3:
            delta_store_spend = st.slider(
                "Avg store spend delta",
                -50000,
                50000,
                0,
                500,
                key="sc_delta_store_spend",
            )
            delta_payment_trust = st.slider(
                "Online payment trust delta",
                -4,
                4,
                0,
                1,
                key="sc_delta_trust",
            )
            delta_touch = st.slider(
                "Need touch/feel score delta",
                -4,
                4,
                0,
                1,
                key="sc_delta_touch",
            )

        deltas = {
            "daily_internet_hours": float(delta_internet),
            "social_media_hours": float(delta_social),
            "tech_savvy_score": float(delta_tech),
            "monthly_online_orders": float(delta_online_orders),
            "monthly_store_visits": float(delta_store_visits),
            "avg_online_spend": float(delta_online_spend),
            "avg_store_spend": float(delta_store_spend),
            "online_payment_trust_score": float(delta_payment_trust),
            "need_touch_feel_score": float(delta_touch),
        }
        scenario_profile = build_scenario_profile(
            base_profile=profile,
            deltas=deltas,
            df=df,
        )
        sc_pred, sc_probs, sc_conf, sc_row, sc_secondary, sc_secondary_prob = predict_customer(
            pipeline=pipeline,
            profile=scenario_profile,
            threshold=threshold,
        )
        sc_adaptive = build_adaptive_strategy(
            predicted=sc_pred,
            probability_map=sc_probs,
            row=sc_row,
        )

        left_compare, right_compare = st.columns(2)
        with left_compare:
            st.markdown("**Baseline**")
            st.markdown(
                f"- Preference: **{format_category(predicted)}** ({confidence:.1%})\n"
                f"- Secondary: **{format_category(secondary_channel)}** ({secondary_probability:.1%})\n"
                f"- Strategy: **{adaptive['strategy']}**\n"
                f"- Digital susceptibility: **{adaptive['digital_index']:.1f}/100**"
            )
        with right_compare:
            st.markdown("**Scenario**")
            st.markdown(
                f"- Preference: **{format_category(sc_pred)}** ({sc_conf:.1%})\n"
                f"- Secondary: **{format_category(sc_secondary)}** ({sc_secondary_prob:.1%})\n"
                f"- Strategy: **{sc_adaptive['strategy']}**\n"
                f"- Digital susceptibility: **{sc_adaptive['digital_index']:.1f}/100**"
            )

        compare_df = pd.DataFrame(
            {
                "category": ["online", "store", "hybrid"] * 2,
                "probability": [
                    probability_map["online"],
                    probability_map["store"],
                    probability_map["hybrid"],
                    sc_probs["online"],
                    sc_probs["store"],
                    sc_probs["hybrid"],
                ],
                "profile": ["baseline"] * 3 + ["scenario"] * 3,
            }
        )
        compare_fig = px.bar(
            compare_df,
            x="category",
            y="probability",
            color="profile",
            barmode="group",
            text=compare_df["probability"].map(lambda x: f"{x:.1%}"),
            color_discrete_map={"baseline": "#334155", "scenario": "#0ea5e9"},
        )
        compare_fig.update_traces(textposition="outside")
        compare_fig.update_layout(
            yaxis=dict(range=[0, 1]),
            xaxis_title="Category",
            yaxis_title="Probability",
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(compare_fig, width="stretch")

        st.info(
            scenario_shift_summary(
                base_pred=predicted,
                base_strategy=adaptive["strategy"],
                scenario_pred=sc_pred,
                scenario_strategy=sc_adaptive["strategy"],
            )
        )

    st.markdown("<div class='section-title'>Model Benchmark</div>", unsafe_allow_html=True)
    if not metrics_df.empty:
        benchmark = metrics_df.copy()
        benchmark["model"] = benchmark["model"].str.replace("_", " ").str.title()
        st.dataframe(benchmark, width="stretch", hide_index=True)
    else:
        st.info("Run scripts/analyze.py to populate benchmark metrics.")

    st.markdown("<div class='section-title'>Top Global Drivers</div>", unsafe_allow_html=True)
    if not top_drivers_df.empty:
        st.dataframe(top_drivers_df.head(12), width="stretch", hide_index=True)
    else:
        st.info("Run scripts/make_outputs.py to populate top driver exports.")

    st.caption(
        "Note: predictions are decision-support signals and should be combined with business context."
    )


if __name__ == "__main__":
    main()
