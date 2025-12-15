import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import joblib  # type: ignore
import numpy as np  # type: ignore

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="centered"
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("startup_success_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# App Header
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🚀 Startup Success Predictor</h1>
    <p style='text-align: center; font-size:18px;'>
    AI-powered tool for founders, investors & accelerators
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar Info
# -----------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.write(
    """
    This tool predicts whether a startup is **likely to succeed or fail**
    based on funding, team, and growth indicators.

    Built using **Machine Learning + GridSearchCV**.
    """
)

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("📊 Enter Startup Details")

funding_total_usd = st.number_input(
    "Total Funding Raised (USD)",
    min_value=0,
    step=100000,
    help="Total investment raised so far"
)

funding_rounds = st.slider(
    "Number of Funding Rounds",
    0, 10, 1
)

team_size = st.slider(
    "Team Size",
    1, 500, 10
)

milestones = st.slider(
    "Number of Business Milestones Achieved",
    0, 20, 2
)

avg_participants = st.number_input(
    "Average Participants (Users / Clients)",
    min_value=0,
    step=100
)

relationships = st.slider(
    "Business Relationships / Partnerships",
    0, 50, 5
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Startup Outcome"):

    # Build a full input vector matching features used during training.
    # Load original dataset to reconstruct numeric feature columns (same preprocessing as training)
    df_ref = pd.read_csv('startup data.csv')
    drop_cols = [
        'Unnamed: 0', 'id', 'name', 'city', 'state_code',
        'state_code.1', 'object_id', 'Unnamed: 6'
    ]
    df_ref.drop(columns=[c for c in drop_cols if c in df_ref.columns], inplace=True, errors='ignore')
    # If status exists in reference, drop it to get feature columns
    if 'status' in df_ref.columns:
        try:
            df_ref['status'] = df_ref['status'].map({'acquired': 1, 'closed': 0})
        except Exception:
            pass
        feature_df = df_ref.drop(columns=['status'], errors='ignore')
    else:
        feature_df = df_ref

    # Numeric feature columns used during training
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    # Create full input row initialized with zeros
    input_full = pd.DataFrame([np.zeros(len(numeric_cols))], columns=numeric_cols)

    # Map available user inputs to corresponding training feature names (if present)
    mapping = {
        'funding_total_usd': funding_total_usd,
        'funding_rounds': funding_rounds,
        'milestones': milestones,
        'avg_participants': avg_participants,
        'relationships': relationships
        # Note: 'team_size' was not used during training and is therefore ignored
    }
    for col, val in mapping.items():
        if col in input_full.columns:
            input_full.at[0, col] = val

    # Ensure columns are ordered exactly as the scaler (if scaler has feature names)
    feature_order = getattr(scaler, 'feature_names_in_', None)
    if feature_order is not None:
        # Add any missing columns required by scaler with zeros
        missing = [c for c in feature_order if c not in input_full.columns]
        for c in missing:
            input_full[c] = 0
        # Reorder
        input_full = input_full[feature_order]

    # Scale input
    input_scaled = scaler.transform(input_full)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")

    # -----------------------------
    # Display Results
    # -----------------------------
    if prediction == 1:
        st.success("✅ **Likely to Succeed**")
        st.write(f"📈 **Success Probability:** `{probability[1]*100:.2f}%`")

        st.info(
            """
            🔹 This startup shows strong indicators of success.
            🔹 Consider scaling, partnerships, and long-term strategy.
            """
        )
    else:
        st.error("❌ **High Risk of Failure**")
        st.write(f"⚠️ **Failure Probability:** `{probability[0]*100:.2f}%`")

        st.warning(
            """
            🔹 The model identifies risk factors.
            🔹 Improving funding strategy or team strength may help.
            """
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
    Built with ❤️ using Machine Learning | For educational & decision-support purposes
    </p>
    """,
    unsafe_allow_html=True
)
