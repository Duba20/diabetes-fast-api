import os
import time
from typing import List

import requests
import streamlit as st


DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "https://diabetes-fast-api.onrender.com")
BASE_DEFAULT_VALUES: List[float] = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
FEATURE_NAMES_8: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

st.set_page_config(page_title="Diabetes Predictor", page_icon=":hospital:", layout="centered")
st.title("Diabetes Prediction UI")
st.caption("Streamlit frontend connected to FastAPI backend")

backend_url = st.text_input(
    "Backend URL",
    value=DEFAULT_BACKEND_URL,
    help="Example: https://diabetes-fast-api.onrender.com",
).rstrip("/")

with st.form("predict_form"):
    st.subheader("Input Features")
    features: List[float] = []
    for idx in range(8):
        features.append(
            st.number_input(
                FEATURE_NAMES_8[idx],
                value=float(BASE_DEFAULT_VALUES[idx]),
                step=0.1,
                format="%.6f",
            )
        )
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        payload = {"features": features}
        named_payload = {FEATURE_NAMES_8[i]: float(features[i]) for i in range(8)}
        response = None

        def post_with_backoff(json_payload: dict) -> requests.Response | None:
            resp = None
            for attempt in range(2):
                resp = requests.post(f"{backend_url}/predict", json=json_payload, timeout=30)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait_seconds = max(1, int(retry_after))
                    else:
                        wait_seconds = 3 * (attempt + 1)
                    time.sleep(wait_seconds)
                    continue
                return resp
            return resp

        # Prefer named payload first to match your current ML backend schema and avoid extra 422 round-trips.
        response = post_with_backoff(named_payload)

        # Fallback for backends that expect `features` instead of named fields.
        if response is not None and response.status_code == 422:
            response = post_with_backoff(payload)

        if response is None:
            st.error("No response from backend.")
        elif response.status_code == 200:
            data = response.json()
            prediction = int(data.get("prediction", data.get("label", 0)))
            probability = data.get("probability")
            risk_level = data.get("risk_level")

            st.success("Prediction successful")
            if probability is not None:
                probability = float(probability)
                probability_pct = probability * 100.0
                if prediction == 1:
                    st.markdown(f"### Result: High likelihood of diabetes ({probability_pct:.2f}%)")
                else:
                    st.markdown(f"### Result: Low likelihood of diabetes ({probability_pct:.2f}%)")
            else:
                if prediction == 1:
                    st.markdown("### Result: High likelihood of diabetes")
                else:
                    st.markdown("### Result: Low likelihood of diabetes")

            if risk_level is None:
                risk_level = "High" if prediction == 1 else "Low"
            st.write(f"Risk level: {risk_level}")
        else:
            st.error(f"Request failed: HTTP {response.status_code}")
            if response.status_code == 429:
                st.info("The backend is rate-limiting requests. Please wait a few seconds and try again.")
            try:
                st.json(response.json())
            except Exception:
                st.code(response.text[:1000] or "No response body")
    except Exception as exc:
        st.error(f"Could not connect to backend: {str(exc)}")

st.divider()
st.write("Quick links")
st.markdown(f"- API docs: {backend_url}/docs")
st.markdown(f"- Health: {backend_url}/health")
