import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.core.config import settings
from src.core.logging import app_logger

class ProjectDataGenerator:
    """
    Enterprise Data Generation Engine.
    Generates ~50,000 highly realistic rows of historical, interconnected 
    software engineering metrics across various synthetic projects.
    """
    def __init__(self, num_projects: int = 500, weeks_per_project: int = 100):
        self.num_projects = num_projects
        self.weeks_per_project = weeks_per_project
        self.total_rows = num_projects * weeks_per_project
        
    def generate_dataset(self) -> pd.DataFrame:
        """
        Executes the generation loop applying domain-specific statistical 
        correlations to simulate realistic software development lifecycles.
        """
        app_logger.info(f"Initiating synthetic data generation for {self.total_rows} snapshot rows...")
        
        np.random.seed(42)  # Secure reproducibility across setups
        data = []
        
        # Define baseline profiles for projects (0 = highly stable, 1 = chaotic/high-risk)
        project_profiles = np.random.beta(a=2, b=5, size=self.num_projects)
        
        base_date = datetime(2024, 1, 1)
        
        for proj_id in range(self.num_projects):
            profile = project_profiles[proj_id]
            
            # Base variables controlled by the underlying project profile
            base_velocity = np.random.uniform(30, 60)
            base_team_size = np.random.randint(5, 25)
            
            for week in range(self.weeks_per_project):
                # 1. Simulate evolution and scope creep over time
                # High-profile risk leads to massive requirement volatility
                requirement_changes = int(np.random.poisson(lam=1.5 + (profile * 4)) + np.random.randint(0, max(1, int(week * 0.05 + 1))))
                
                # 2. Velocity degradation based on scope creep and profile friction
                velocity_drop_modifier = min(0.8, (requirement_changes * 0.03) + (profile * 0.2))
                sprint_velocity = max(10.0, float(base_velocity * (1 - velocity_drop_modifier) + np.random.normal(0, 3)))
                
                # 3. Workload calculations (higher task pressure / team size ratio)
                developer_workload = float(min(100.0, 50.0 + (profile * 30.0) + (requirement_changes * 2.0) + np.random.normal(0, 5)))
                
                # 4. Git code commit patterns
                git_commits = int(max(20, base_team_size * np.random.uniform(10, 25) * (1 - (developer_workload / 250))))
                
                # 5. Testing coverage decays under extreme velocity duress
                testing_coverage = float(max(40.0, 85.0 - (profile * 25.0) - (developer_workload * 0.1) + np.random.normal(0, 2)))
                
                # 6. Bug tracking metrics generation (correlated to workload and poor testing coverage)
                bug_multiplier = 1.0 + (profile * 1.5) + ((85.0 - testing_coverage) * 0.04)
                open_bugs = int(np.random.poisson(lam=15 * bug_multiplier) + (requirement_changes * 1.5))
                high_priority_bugs = int(np.random.binomial(n=open_bugs, p=max(0.05, min(0.4, 0.1 + (profile * 0.2)))))
                
                # 7. Operational friction and blockers
                code_review_duration = float(max(1.0, 4.0 + (profile * 12.0) + (high_priority_bugs * 0.5) + np.random.normal(0, 1)))
                ci_cd_failures = int(np.random.poisson(lam=1.0 + (profile * 3.0) + (high_priority_bugs * 0.1)))
                delayed_tasks = int(np.random.poisson(lam=5 + (profile * 15) + (high_priority_bugs * 0.8)))
                
                # 8. Team dynamic tracking
                meeting_attendance = float(max(50.0, 95.0 - (profile * 25.0) - (developer_workload * 0.15)))
                pull_request_activity = int(max(5, (git_commits // 10) + np.random.randint(-3, 4)))
                
                # 9. Class Labels & Ground-Truth Risk Threshold Calculations
                # Target: Overall Project Failure Risk
                risk_score = (
                    (delayed_tasks * 0.25) + 
                    (high_priority_bugs * 0.20) + 
                    ((60 - sprint_velocity) * 0.15) + 
                    (requirement_changes * 0.15) + 
                    (code_review_duration * 0.15) + 
                    ((100 - meeting_attendance) * 0.10)
                )
                
                # Map to target metrics requested in proposal
                project_failure_risk = min(99.0, max(5.0, risk_score + np.random.normal(0, 4)))
                failure_label = 1 if project_failure_risk > 65.0 else 0
                
                # Deadline delay in days
                estimated_deadline_delay = max(0, int((project_failure_risk * 0.4) + np.random.normal(0, 2)))
                
                # Budget Risk estimation
                budget_risk = min(99.0, max(5.0, (project_failure_risk * 0.85) + np.random.normal(0, 3)))
                
                # Developer Burnout State evaluation
                developer_burnout = "High" if developer_workload > 78.0 or code_review_duration > 14.0 else ("Medium" if developer_workload > 60.0 else "Low")
                
                snapshot_date = base_date + timedelta(weeks=week)
                
                data.append({
                    "project_id": f"PROJ_{proj_id:03d}",
                    "week_index": week,
                    "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                    "delayed_tasks": delayed_tasks,
                    "git_commits": git_commits,
                    "open_bugs": open_bugs,
                    "high_priority_bugs": high_priority_bugs,
                    "sprint_velocity": round(sprint_velocity, 2),
                    "developer_workload": round(developer_workload, 2),
                    "code_review_duration": round(code_review_duration, 2),
                    "requirement_changes": requirement_changes,
                    "meeting_attendance": round(meeting_attendance, 2),
                    "ci_cd_failures": ci_cd_failures,
                    "testing_coverage": round(testing_coverage, 2),
                    "pull_request_activity": pull_request_activity,
                    "project_failure_risk_pct": round(project_failure_risk, 2),
                    "estimated_deadline_delay_days": estimated_deadline_delay,
                    "budget_risk_pct": round(budget_risk, 2),
                    "developer_burnout_status": developer_burnout,
                    "is_failed": failure_label
                })
                
        df = pd.DataFrame(data)
        
        # Save structural file out cleanly to enterprise data directory location
        output_path = settings.DATA_DIR / "synthetic_project_data.csv"
        df.to_csv(output_path, index=False)
        app_logger.info(f"Dataset generated and persisted successfully. Shape: {df.shape} -> Path: {output_path}")
        return df

if __name__ == "__main__":
    generator = ProjectDataGenerator()
    generator.generate_dataset()