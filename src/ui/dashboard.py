import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Project Failure Intelligence Platform", page_icon="🔮", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if st.session_state.theme == "dark":
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, #07111f 0%, #0e1f35 100%); color: #f5f7fb; }
        .block-container { padding-top: 1.2rem; }
        div[data-testid="stMetric"] { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 0.8rem; }
        .stButton > button { border-radius: 999px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp { background: #f5f7fb; color: #14213d; }
        div[data-testid="stMetric"] { background: white; border: 1px solid #e5e7eb; border-radius: 14px; padding: 0.8rem; }
        .stButton > button { border-radius: 999px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("🔮 Project Failure Intelligence Platform")
st.caption("Upload real project data or test a live scenario to get executive-ready risk forecasts and recommendations.")

with st.sidebar:
    st.header("Operations Console")
    st.write("Monitor delivery risk with uploaded project data and historical context.")
    st.session_state.theme = st.radio("Theme", ["dark", "light"], horizontal=True)
    st.divider()
    st.subheader("Upload project dataset")
    uploaded_file = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    st.divider()
    st.subheader("Manual snapshot")
    delayed_tasks = st.number_input("Delayed tasks", min_value=0, value=22)
    git_commits = st.number_input("Weekly commits", min_value=0, value=140)
    open_bugs = st.number_input("Open bugs", min_value=0, value=18)
    high_priority_bugs = st.number_input("High priority bugs", min_value=0, value=4)
    sprint_velocity = st.number_input("Sprint velocity", min_value=0.0, value=72.0, step=0.1)
    developer_workload = st.number_input("Developer workload", min_value=0.0, value=5.2, step=0.1)
    code_review_duration = st.number_input("Code review duration (hours)", min_value=0.0, value=12.0, step=0.1)
    requirement_changes = st.number_input("Requirement changes", min_value=0, value=2)
    meeting_attendance = st.number_input("Meeting attendance (%)", min_value=0.0, max_value=100.0, value=91.0, step=0.1)
    ci_cd_failures = st.number_input("CI/CD failures", min_value=0, value=1)
    testing_coverage = st.number_input("Testing coverage (%)", min_value=0.0, max_value=100.0, value=77.0, step=0.1)
    pull_request_activity = st.number_input("Pull requests", min_value=0, value=13)
    run_button = st.button("Run prediction", use_container_width=True)

if run_button:
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
        try:
            response = requests.post(f"{API_BASE}/api/v1/predict/upload", files=files, timeout=120)
        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")
            st.stop()
    else:
        payload = {
            "delayed_tasks": int(delayed_tasks),
            "git_commits": int(git_commits),
            "open_bugs": int(open_bugs),
            "high_priority_bugs": int(high_priority_bugs),
            "sprint_velocity": float(sprint_velocity),
            "developer_workload": float(developer_workload),
            "code_review_duration": float(code_review_duration),
            "requirement_changes": int(requirement_changes),
            "meeting_attendance": float(meeting_attendance),
            "ci_cd_failures": int(ci_cd_failures),
            "testing_coverage": float(testing_coverage),
            "pull_request_activity": int(pull_request_activity),
        }
        try:
            response = requests.post(f"{API_BASE}/api/v1/predict/snapshot", json=payload, timeout=120)
        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")
            st.stop()

    if response.status_code != 200:
        st.error(f"Prediction request failed: {response.text}")
        st.stop()

    response_data = response.json()
    if uploaded_file is not None:
        summary = response_data.get("summary", {})
        preview = response_data.get("preview_rows", [])
        results = response_data.get("results", [])
    else:
        summary = {"total_rows": 1, "high_risk_rows": 1 if response_data["prediction"]["is_failed"] else 0, "average_risk_pct": response_data["prediction"]["project_failure_risk_pct"], "average_confidence": response_data["prediction"]["confidence_score"]}
        preview = []
        results = [response_data["prediction"]]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows evaluated", summary.get("total_rows", 0))
    col2.metric("High risk rows", summary.get("high_risk_rows", 0))
    col3.metric("Average risk", f"{summary.get('average_risk_pct', 0):.2f}%")
    col4.metric("Average confidence", f"{summary.get('average_confidence', 0):.2f}")

    st.subheader("Prediction outcomes")
    if results:
        result_df = pd.DataFrame(results)
        st.dataframe(result_df[["row_index", "prediction_label", "confidence_score", "project_failure_risk_pct"]], use_container_width=True)

        feature_df = pd.DataFrame(
            [
                {"feature": item["feature"], "importance": item["importance"]}
                for item in results[0].get("driver_summary", [])
            ]
        )
        if not feature_df.empty:
            fig = px.bar(feature_df, x="feature", y="importance", color="feature", title="Key risk drivers")
            st.plotly_chart(fig, use_container_width=True)

    if preview:
        st.subheader("Upload preview")
        st.dataframe(pd.DataFrame(preview).head(10), use_container_width=True)

else:
    st.info("Use the sidebar to upload a dataset or run a demo scenario. The app will score the data and store history in the local database.")

try:
    history = requests.get(f"{API_BASE}/api/v1/history", timeout=30).json()
    if history:
        st.subheader("Recent predictions")
        history_df = pd.DataFrame(history)
        history_df["evaluated_at"] = pd.to_datetime(history_df["evaluated_at"])
        st.dataframe(history_df[["evaluated_at", "prediction_label", "risk_pct", "confidence", "source_type", "source_name"]], use_container_width=True)
except requests.RequestException:
    st.caption("History service unavailable; start the FastAPI backend to enable analytics history.")