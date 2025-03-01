import streamlit as st
import json
from llm import generate_health_plan
from configs import Config
from PIL import Image
import random
from datetime import datetime, timedelta
import pandas as pd
import altair as alt

config = Config()

st.set_page_config(
    page_title="NEURO OWO",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://neuro-owo.com/help',
        'Report a bug': 'https://neuro-owo.com/support',
        'About': "Neuro OWO v1.0 - AI Health Assistant"
    }
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 基础布局优化 */
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 1rem;
    }
    .block-container {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 2.5rem 1.5rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        max-width: 1200px;
        margin: auto;
    }

    /* 输入控件样式 */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
    }

    /* 按钮样式 */
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        width: 100%;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(76,175,80,0.3);
    }

    /* 可视化容器 */
    .visualization-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* 响应式布局 */
    @media (max-width: 768px) {
        .block-container {
            padding: 1.5rem 1rem !important;
        }
        .stForm {
            padding: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

def main():
    # 初始化session state
    PRESET_KEYS = ["push_ups", "pull_ups", "cardio"]
    for key in PRESET_KEYS:
        if f"preset_{key}" not in st.session_state:
            st.session_state[f"preset_{key}"] = 0

    # Header Section
    header_cols = st.columns([1, 4], gap="large")
    with header_cols[0]:
        logo = Image.open("logo.png")
        st.image(logo, use_container_width=True, output_format="PNG")
    with header_cols[1]:
        st.title("NEURO OWO")
        st.markdown("## Your Exclusive AI Health Assistant")
    st.markdown("---")

    # User Input Form
    with st.form("user_info"):
        # Personal Info
        cols = st.columns(2, gap="large")
        with cols[0]:
            name = st.text_input("Name", placeholder="John", key="name")
            gender = st.selectbox("Gender", options=["Male", "Female", "Others"], key="gender")
        with cols[1]:
            height = st.number_input("Height (cm)", 100, 250, 175, key="height")
            weight = st.number_input("Weight (kg)", 30, 300, 70, key="weight")
        age = st.number_input("Age", 1, 120, 25, key="age")

        # Training Data
        st.subheader("🏋️ Today's Workout")
        training_cols = st.columns(3, gap="medium")
        with training_cols[0]:
            push_ups = st.number_input("Push-ups", 0, 1000, key="preset_push_ups")
        with training_cols[1]:
            pull_ups = st.number_input("Pull-ups", 0, 500, key="preset_pull_ups")
        with training_cols[2]:
            cardio = st.number_input("Cardio (min)", 0, 3000, key="preset_cardio")

        # Submit Button
        submitted = st.form_submit_button("Generate Your Plan 🚀", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing your health profile..."):
            try:
                # Build user data
                user_data = {
                    "user_data": {
                        "name": name,
                        "gender": gender,
                        "height": height,
                        "weight": weight,
                        "age": age,
                        "last_week_training": {
                            "push_ups": push_ups,
                            "pull_ups": pull_ups,
                            "cardio_minutes": cardio
                        }
                    }
                }

                # Generate visualization data
                dates = [(datetime.now() - timedelta(days=i)).strftime("%a, %b %d") 
                        for i in range(6, -1, -1)]
                
                # Activity conversion rates
                CONVERSION = {
                    "push_ups": 0.2,  # mins per rep
                    "pull_ups": 0.3,   # mins per rep
                    "cardio": 1.0      # direct minutes
                }
                
                total_mins = (
                    push_ups * CONVERSION["push_ups"] +
                    pull_ups * CONVERSION["pull_ups"] +
                    cardio * CONVERSION["cardio"]
                )

                # Generate historical data
                history_data = {
                    "date": dates,
                    "duration": [
                        max(10, int(total_mins * random.uniform(0.8, 1.2) / 7))
                        for _ in range(7)
                    ]
                }

                # Visualization Section
                with st.container():
                    st.subheader("📈 Weekly Training Analytics")
                    
                    # Line Chart
                    line_chart = alt.Chart(pd.DataFrame(history_data)).mark_area(
                        interpolate='monotone',
                        line={'color':'#4CAF50', 'width':3},
                        color=alt.Gradient(
                            gradient='linear',
                            stops=[alt.GradientStop(color='#4CAF50', offset=0),
                                   alt.GradientStop(color='#A5D6A7', offset=1)],
                            x1=1, x2=1, y1=1, y2=0
                        )
                    ).encode(
                        x=alt.X('date:O', title="Date", axis=alt.Axis(labelAngle=45)),
                        y=alt.Y('duration:Q', title="Minutes", 
                               scale=alt.Scale(domain=[0, max(history_data["duration"])*1.2] ) ),
                        tooltip=[alt.Tooltip('date:O', title='Date'),
                                alt.Tooltip('duration:Q', title='Minutes', format='.0f')]
                    ).properties(
                        height=400,
                        width=800
                    ).configure_view(strokeWidth=0)

                    # Metrics
                    avg_duration = sum(history_data["duration"]) / 7
                    last_week_avg = avg_duration * random.uniform(0.9, 1.1)
                    
                    # Layout
                    viz_cols = st.columns([3, 1], gap="large")
                    with viz_cols[0]:
                        st.altair_chart(line_chart, use_container_width=True)
                    with viz_cols[1]:
                        st.metric("Daily Average", 
                                f"{avg_duration:.1f} mins", 
                                delta=f"{'↑' if avg_duration > last_week_avg else '↓'} "
                                      f"{abs(avg_duration - last_week_avg):.1f} vs last week",
                                delta_color="normal")

                # Generate Health Plan
                result = generate_health_plan(user_data)
                
                # Display Results
                st.success("Your personalized health plan is ready!")
                st.markdown("---")
                
                # Analysis Section
                with st.expander("📋 Comprehensive Health Analysis", expanded=True):
                    st.markdown(f"""
                    #### Health Overview
                    {result.get('analysis', '')}
                    """)
                    
                    health_cols = st.columns(2, gap="medium")
                    with health_cols[0]:
                        st.markdown("##### 🏋️ Exercise Recommendations")
                        st.json(result.get("exercise_recommendations", {}))
                    with health_cols[1]:
                        st.markdown("##### 🥗 Dietary Plan")
                        st.json(result.get("dietary_suggestions", {}))

                    st.markdown("##### 💡 Pro Tips")
                    tips_cols = st.columns(2, gap="small")
                    for idx, tip in enumerate(result.get("health_tips", [])):
                        with tips_cols[idx % 2]:
                            st.markdown(f"- {tip}")

                # Raw Data
                with st.expander("🔍 View Raw Data"):
                    st.code(json.dumps(result, indent=2, ensure_ascii=False))

            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

if __name__ == "__main__":
    main()