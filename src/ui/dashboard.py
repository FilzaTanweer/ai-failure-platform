import streamlit as st
import requests

# Page layout configuration settings
st.set_page_config(
    page_title="Project Failure Risk Intelligence Platform",
    page_icon="🔮",
    layout="wide"
)

# Application titles and description summaries
st.title("🔮 AI Project Failure Risk Intelligence Platform")
st.markdown("---")
st.sidebar.header("📊 Real-Time Operations Telemetry")
st.sidebar.markdown("Adjust active weekly metric distributions below to evaluate project health stability in real-time.")

# 1. Sidebar Input Field Configurations mapping directly to our model blueprints
delayed_tasks = st.sidebar.slider("Delayed Tasks Count", 0, 150, 20)
git_commits = st.sidebar.slider("Weekly Git Commits Pushed", 0, 300, 45)
open_bugs = st.sidebar.slider("Active Open Bugs", 0, 200, 30)
high_priority_bugs = st.sidebar.slider("High Priority Bugs Count", 0, 50, 5)
sprint_velocity = st.sidebar.slider("Sprint Velocity Index", 0.0, 100.0, 65.0)
developer_workload = st.sidebar.slider("Developer Workload Stress Level", 0.0, 10.0, 4.2)
code_review_duration = st.sidebar.slider("Code Review Duration Window (Hours)", 0.0, 72.0, 14.5)
requirement_changes = st.sidebar.slider("Scope Requirement Changes Count", 0, 15, 2)
meeting_attendance = st.sidebar.slider("Team Meeting Attendance Percentage", 0.0, 100.0, 92.0)
ci_cd_failures = st.sidebar.slider("Weekly CI/CD Pipeline Failures", 0, 20, 1)
testing_coverage = st.sidebar.slider("Unit Testing Code Coverage (%)", 0.0, 100.0, 78.5)
pull_request_activity = st.sidebar.slider("Active Pull Request Operations Count", 0, 80, 12)

# Build unified payload schema targeting API constraints
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
    "pull_request_activity": int(pull_request_activity)
}

# 2. Layout Distribution Components for Analytics Cards
col1, col2 = st.columns(2)

API_URL = "http://127.0.0.1:8000/predict"

try:
    # Ship synchronous prediction payload to our active FastAPI microservice
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        # Extract response layers
        is_failed = result["prediction"]["is_failed"]
        label = result["prediction"]["prediction_label"]
        confidence = result["prediction"]["confidence_score"]
        risk_pct = result["risk_analysis"]["project_failure_risk_pct"]
        
        # Render visual layout metric items inside defined dashboard column arrays
        with col1:
            st.subheader("⚠️ Core Health Status")
            if is_failed:
                st.error(f"Prediction Alert: {label}")
            else:
                st.success(f"Status Normal: {label}")
                
            st.metric(
                label="Classification Engine Confidence Score", 
                value=f"{confidence * 100:.2f}%"
            )
            
        with col2:
            st.subheader("📈 Statistical Drift Analysis")
            st.metric(
                label="Calculated Project Failure Risk Margin", 
                value=f"{risk_pct}%",
                delta=f"{risk_pct - 35.0:.1f}% vs Threshold" if risk_pct > 35.0 else None,
                delta_color="inverse"
            )
            
            # Progress bars for quick visualization
            st.progress(int(risk_pct))
            
    else:
        st.warning(f"Unable to complete telemetry evaluation loops. API Status Code received: {response.status_code}")

except requests.exceptions.ConnectionError:
    st.info("💡 Service Link Status Offline: Waiting for FastAPI server execution... Make sure your Uvicorn API engine is actively running on port 8000!")