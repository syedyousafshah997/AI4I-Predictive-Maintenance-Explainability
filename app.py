"""
Predictive Maintenance — Streamlit Demo App
=============================================
Loads the final trained model + preprocessing pipeline produced by research.ipynb
and serves live predictions with a local SHAP explanation. No training happens here.
"""

import json
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance — AI4I 2020",
    page_icon="🛠️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 2. Load artifacts (cached — loaded once per session, never retrained)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("artifacts/preprocessor.joblib")
    model = joblib.load("artifacts/final_model.joblib")
    with open("artifacts/metadata.json") as f:
        metadata = json.load(f)
    results_df = pd.read_csv("artifacts/results_all_9_configs.csv")
    return preprocessor, model, metadata, results_df


preprocessor, model, metadata, results_df = load_artifacts()

FINAL_MODEL_NAME = metadata["final_model_name"]
FINAL_STRATEGY = metadata["final_strategy"]
NUMERIC_FEATURES = metadata["numeric_features"]
CATEGORICAL_FEATURES = metadata["categorical_features"]
FEATURE_NAMES_OUT = metadata["feature_names_out"]
FINAL_METRICS = metadata["final_metrics"]

# ---------------------------------------------------------------------------
# 3. Navigation
# ---------------------------------------------------------------------------
st.title("🛠️ Predictive Maintenance — AI4I 2020")
st.caption(
    f"Final model: **{FINAL_MODEL_NAME}** trained with **{FINAL_STRATEGY}** imbalance handling"
)

page = st.sidebar.radio(
    "Navigate",
    ["🏠 About", "🔮 Predict", "📊 Model Performance", "🧪 Research Comparison"],
)

# ---------------------------------------------------------------------------
# 4. About / Home
# ---------------------------------------------------------------------------
if page == "🏠 About":
    st.header("About this project")
    st.markdown(
        """
Predictive maintenance uses sensor data from a machine to anticipate a failure **before**
it happens, so maintenance can be scheduled proactively instead of reactively.

This app is a lightweight demo built on top of a larger research notebook
(`predictive_maintainance.ipynb`) that studies the **AI4I 2020 Predictive Maintenance dataset** — 10,000
synthetic milling-machine operating records with a rare `Machine failure` target
(roughly 3–4% of rows).

**Research objective.** The notebook trains Random Forest, SVM, and XGBoost under three
imbalance-handling strategies each (No Handling, Class Weight, SMOTE) — 9 configurations
total — evaluated on the same held-out test set, and asks two questions:

1. Which model/strategy combination best detects real failures (not just accuracy)?
2. Does the *way* you handle class imbalance (class weighting vs. SMOTE) change what a
   SHAP explainer says the model is paying attention to — not just how well it scores?

**This app** loads the final selected model and its exact preprocessing pipeline (both
already fit — nothing is retrained here), lets you enter live sensor readings, and returns
a failure prediction with a SHAP-based explanation of *why*.
"""
    )
    st.info(
        f"Final selected configuration: **{FINAL_MODEL_NAME} + {FINAL_STRATEGY}**, "
        f"chosen for the best balance of Recall, PR-AUC and F1 on the failure class "
        f"(not for raw accuracy alone)."
    )

# ---------------------------------------------------------------------------
# 5. Predict
# ---------------------------------------------------------------------------
elif page == "🔮 Predict":
    st.header("Live failure prediction")
    st.write("Enter the current operating readings for a machine:")

    col1, col2 = st.columns(2)
    with col1:
        product_type = st.selectbox("Product Type", options=["L", "M", "H"], index=0)
        air_temp = st.number_input(
            "Air temperature [K]", min_value=290.0, max_value=320.0, value=300.0, step=0.1
        )
        process_temp = st.number_input(
            "Process temperature [K]", min_value=295.0, max_value=320.0, value=310.0, step=0.1
        )
    with col2:
        rpm = st.number_input(
            "Rotational speed [rpm]", min_value=1000, max_value=3000, value=1500, step=10
        )
        torque = st.number_input(
            "Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.1
        )
        tool_wear = st.number_input(
            "Tool wear [min]", min_value=0, max_value=260, value=100, step=1
        )

    predict_clicked = st.button("Predict", type="primary")

    # -- 6. Input validation -------------------------------------------------
    def validate_inputs():
        errors = []
        if not (200 <= air_temp <= 350):
            errors.append("Air temperature looks out of a plausible range (200–350 K).")
        if not (200 <= process_temp <= 350):
            errors.append("Process temperature looks out of a plausible range (200–350 K).")
        if process_temp < air_temp:
            errors.append("Process temperature is normally higher than air temperature.")
        if rpm <= 0:
            errors.append("Rotational speed must be positive.")
        if torque < 0:
            errors.append("Torque cannot be negative.")
        if tool_wear < 0:
            errors.append("Tool wear cannot be negative.")
        return errors

    if predict_clicked:
        errors = validate_inputs()
        if errors:
            for e in errors:
                st.error(e)
        else:
            # -- 7. Build input row, preprocess exactly like training --------
            input_df = pd.DataFrame([{
                "Type": product_type,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": process_temp,
                "Rotational speed [rpm]": rpm,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
            }])

            X_input = preprocessor.transform(input_df)
            X_input_df = pd.DataFrame(X_input, columns=FEATURE_NAMES_OUT)

            pred = model.predict(X_input_df)[0]
            proba = model.predict_proba(X_input_df)[0, 1]

            # -- 8. Output ----------------------------------------------------
            st.subheader("Result")
            if pred == 1:
                st.error(f"⚠️ Machine Failure = **Yes**  (probability: {proba:.1%})")
            else:
                st.success(f"✅ Machine Failure = **No**  (probability: {proba:.1%})")
            st.caption(
                "This probability is a model estimate on synthetic training data, not a "
                "guarantee — treat it as decision support, not a certainty."
            )

            # -- 9 & 10. Local SHAP explanation --------------------------------
            st.subheader("Why did the model say this?")
            with st.spinner("Computing local SHAP explanation..."):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_input_df)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)) and np.ndim(base_value) > 0:
                        base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else base_value[0]

                    contrib = pd.Series(shap_values[0], index=X_input_df.columns)
                    contrib_sorted = contrib.reindex(contrib.abs().sort_values(ascending=False).index)

                    fig, ax = plt.subplots(figsize=(7, 4))
                    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in contrib_sorted.values]
                    ax.barh(contrib_sorted.index[::-1], contrib_sorted.values[::-1], color=colors[::-1])
                    ax.set_xlabel("SHAP value (impact on failure prediction)")
                    ax.set_title("Local feature contributions for this prediction")
                    st.pyplot(fig)

                    top_pos = contrib_sorted[contrib_sorted > 0].head(3)
                    top_neg = contrib_sorted[contrib_sorted < 0].sort_values().head(3)

                    st.markdown("**Human-readable summary**")
                    if len(top_pos) > 0:
                        st.write(
                            "Pushing **toward failure**: "
                            + ", ".join(f"`{k}`" for k in top_pos.index)
                        )
                    if len(top_neg) > 0:
                        st.write(
                            "Pushing **toward normal operation**: "
                            + ", ".join(f"`{k}`" for k in top_neg.index)
                        )
                    st.caption(
                        "These are statistical associations the model learned from training "
                        "data, not proven causal relationships between a sensor reading and a "
                        "failure."
                    )
                except Exception as e:
                    st.warning(f"Could not compute a SHAP explanation for this input: {e}")

# ---------------------------------------------------------------------------
# 6. Model performance
# ---------------------------------------------------------------------------
elif page == "📊 Model Performance":
    st.header("Final model performance")
    st.write(f"Selected configuration: **{FINAL_MODEL_NAME} + {FINAL_STRATEGY}**")

    metric_cols = st.columns(len(FINAL_METRICS))
    for col, (k, v) in zip(metric_cols, FINAL_METRICS.items()):
        col.metric(k, f"{v:.3f}")

    st.markdown(
        """
This configuration was chosen using a weighted score that favors **Recall** and **PR-AUC**
(and F1 as a tie-breaker) over raw Accuracy, because `Machine failure` is a rare event —
optimizing for accuracy alone would reward a model that simply predicts "no failure" almost
every time.
"""
    )

# ---------------------------------------------------------------------------
# 7. Research comparison
# ---------------------------------------------------------------------------
elif page == "🧪 Research Comparison":
    st.header("No Handling vs. Class Weight vs. SMOTE — all 9 configurations")
    st.dataframe(
        results_df.style.format(
            {c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]}
        ),
        use_container_width=True,
    )
    st.markdown(
        """
**Research finding.** Moving from *No Handling* to *Class Weight* or *SMOTE* generally
improves Recall (fewer missed failures) for all three model families, usually at some cost
to Precision. The full notebook (`predictive_maintainance.ipynb`) goes one step further and compares SHAP
feature-importance rankings between the Class-Weight and SMOTE variants of the same model,
to check whether the choice of imbalance-handling technique changes not just performance but
*which features the model treats as most important*.
"""
    )

# ---------------------------------------------------------------------------
# 8 & 13. Footer — limitations notice
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Limitations:** AI4I 2020 is a **synthetic** dataset. Predictions here are research "
    "outputs for demonstration purposes only, not guaranteed maintenance decisions for a real "
    "machine."
)
