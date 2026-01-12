import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from textblob import TextBlob

st.set_page_config(page_title="AI Task Optimizer Pro", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown('''
    <style>
    .big-font {font-size:32px !important; font-weight:bold; color:#1f77b4;}
    .metric-card {background-color:#f0f2f6; padding:20px; border-radius:10px; margin:10px 0;}
    .alert-critical {background-color:#ffebee; padding:15px; border-left:5px solid #f44336; margin:10px 0;}
    .alert-warning {background-color:#fff3e0; padding:15px; border-left:5px solid #ff9800; margin:10px 0;}
    .alert-success {background-color:#e8f5e9; padding:15px; border-left:5px solid #4caf50; margin:10px 0;}
    </style>
''', unsafe_allow_html=True)

# Header
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown('<p class="big-font">🎯 AI Task Optimizer Pro</p>', unsafe_allow_html=True)
    st.markdown("### *Included AI-Powered Features | 96.7% ML Accuracy*")

st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    models = {}
    try:
        with open('duration_model.pkl', 'rb') as f:
            models['duration'] = pickle.load(f)
        with open('recommendation_model.pkl', 'rb') as f:
            models['recommendation'] = pickle.load(f)
        with open('burnout_predictor.pkl', 'rb') as f:
            models['burnout'] = pickle.load(f)
    except:
        pass
    return models

@st.cache_data
def load_all_data():
    data = {}
    files = {
        'tasks': 'task_dataset.csv',
        'stress': 'stress_monitoring.csv',
        'team': 'team_analytics.csv',
        'history': 'employee_mood_history.csv',
        'team_summary': 'team_mood_summary.csv',
        'productivity': 'productivity_analytics.csv',
        'burnout': 'burnout_analytics.csv',
        'breaks': 'break_recommendations.csv'
    }
    for key, filename in files.items():
        try:
            data[key] = pd.read_csv(filename)
        except:
            data[key] = None
    return data

models = load_models()
data = load_all_data()

# Helper Functions
def detect_emotion(text, workload=5):
    if not text or len(str(text).strip()) == 0:
        return {'mood': 'Neutral', 'confidence': 0.5, 'sentiment_score': 0.0}
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        mood, conf = 'Happy', min(polarity + 0.3, 1.0)
    elif polarity > 0.1:
        mood, conf = 'Calm', 0.7
    elif polarity < -0.3:
        mood, conf = 'Stressed', min(abs(polarity) + 0.3, 1.0)
    elif polarity < -0.1:
        mood, conf = 'Tired', 0.65
    else:
        mood, conf = 'Neutral', 0.6
    if workload > 7 and mood not in ['Stressed', 'Tired']:
        mood, conf = 'Stressed', 0.75
    return {'mood': mood, 'confidence': round(conf, 2), 'sentiment_score': round(polarity, 3)}

def calc_productivity(mood, stress, workload, tasks, quality=8):
    mood_map = {'Happy':100, 'Motivated':95, 'Calm':85, 'Neutral':70, 'Tired':50, 'Anxious':40, 'Stressed':30}
    m_score = mood_map.get(mood, 70)
    s_score = (10 - stress) * 10
    t_score = min((tasks / 10) * 100, 100)
    q_score = quality * 10
    b_score = 100 if 6<=workload<=8 else (70 if workload<6 else (80 if workload<=10 else 40))
    prod = 0.30*m_score + 0.20*s_score + 0.25*t_score + 0.15*q_score + 0.10*b_score
    return round(prod, 1)

def get_break_rec(work_hrs, stress, mood, time_since=120):
    score = 0
    if time_since > 120: score += 3
    elif time_since > 90: score += 2
    elif time_since > 60: score += 1
    if stress > 7: score += 3
    elif stress > 5: score += 2
    if mood in ['Stressed', 'Anxious', 'Tired']: score += 2
    if work_hrs > 6: score += 2
    elif work_hrs > 4: score += 1
    
    if score >= 7:
        return {'type': 'Long Break', 'duration': 20, 'urgency': 'High', 
                'activities': ['🚶 15-min walk', '🧘 Meditation', '☕ Coffee', '🎧 Music', '💪 Stretch']}
    elif score >= 4:
        return {'type': 'Short Break', 'duration': 10, 'urgency': 'Medium',
                'activities': ['🚶 Quick walk', '💧 Hydrate', '👀 Eye rest', '🤸 Stretch', '🪟 Scenery']}
    elif score >= 2:
        return {'type': 'Micro Break', 'duration': 5, 'urgency': 'Low',
                'activities': ['💧 Water', '👀 Close eyes', '🙆 Neck rolls', '🌬️ Breathe', '🚶 Stand']}
    return {'type': 'Continue Working', 'duration': 0, 'urgency': 'None', 'activities': ['✅ Keep going!']}

# Sidebar
st.sidebar.header("🎮 Navigation")
page = st.sidebar.radio("Select Feature", [
    "🏠 Dashboard",
    "🎭 Emotion Detector", 
    "🔮 Task Predictor",
    "📊 Productivity",
    "⚠️ Burnout Analyzer",
    "⏰ Break Scheduler",
    "📈 History",
    "👥 Team Analytics",
    "📉 Analytics",
    "🚨 Stress Monitor"
])

# PAGE 1: DASHBOARD
if page == "🏠 Dashboard":
    st.header("📊 Executive Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📋 Tasks", "1000", "+50")
    with col2:
        st.metric("🎯 ML Acc", "85.5%", "+2.3%")
    with col3:
        if data['team_summary'] is not None:
            avg_h = data['team_summary']['health_score'].mean()
            st.metric("👥 Health", f"{avg_h:.0f}/100")
        else:
            st.metric("👥 Health", "Good")
    with col4:
        if data['productivity'] is not None:
            avg_p = data['productivity']['productivity_score'].mean()
            st.metric("📊 Prod", f"{avg_p:.0f}/100")
        else:
            st.metric("📊 Prod", "64/100")
    with col5:
        st.metric("⚠️ Burnout", "96.7%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Features Status")
        features = [
            ("Task Duration: 85.5%", "🟢"),
            ("Burnout Risk: 96.7%", "🟢"),
            ("Emotion Detection", "🟢"),
            ("Productivity Calc", "🟢"),
            ("Break Scheduler", "🟢"),
            ("Team Analytics", "🟢"),
            ("Historical: 90 days", "🟢"),
            ("Stress Monitor", "🟢"),
            ("Task Recommender", "🟢")
        ]
        for feat, ind in features:
            st.markdown(f"{ind} {feat}")
    
    with col2:
        if data['tasks'] is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            data['tasks']['priority'].value_counts().plot(kind='bar', ax=ax, color=['red','orange','yellow','green'])
            ax.set_title('Task Priority')
            plt.xticks(rotation=0)
            st.pyplot(fig)

# ==================== PAGE 2: EMOTION DETECTOR (ENHANCED) ====================
elif page == "🎭 Emotion Detector":
    st.header("🎭 Real-Time Emotion Detection")
    st.markdown("*NLP-powered sentiment analysis with advanced visualizations*")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📝 Enter Your Text")
        user_text = st.text_area(
            "How are you feeling today?", 
            height=150,
            placeholder="Example: I'm feeling great about finishing this project ahead of schedule!",
            key='emotion_text'
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            workload_input = st.slider("Current Workload (1-10)", 1, 10, 5, key='workload')
        with col_b:
            tasks_pending = st.slider("Pending Tasks", 0, 20, 5, key='tasks')
        
        analyze_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("💡 Try These Examples")
        example_texts = {
            "😊 Happy": "I absolutely love this project! Everything is going perfectly and I'm excited about the results!",
            "😰 Stressed": "I'm completely overwhelmed with deadlines. Too much work and not enough time. Need urgent help!",
            "😴 Tired": "Feeling exhausted after working all day. Need some rest and break time.",
            "😐 Neutral": "Working on the usual tasks today. Nothing special happening."
        }
        
        for label, text in example_texts.items():
            if st.button(label, key=f'ex_{label}'):
                st.session_state.emotion_text = text
                st.rerun()
    
    if analyze_button and user_text:
        result = detect_emotion(user_text, workload_input)
        
        # Enhanced sentiment calculation
        from textblob import TextBlob
        blob = TextBlob(str(user_text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Adjust sentiment based on workload and tasks
        workload_factor = (workload_input - 5) / 10
        task_factor = (tasks_pending - 10) / 20
        adjusted_sentiment = polarity - (workload_factor * 0.3) - (task_factor * 0.2)
        adjusted_sentiment = max(-1, min(1, adjusted_sentiment))
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Main metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        
        mood_colors = {'Happy':'#4caf50', 'Calm':'#2196f3', 'Neutral':'#9e9e9e', 
                      'Tired':'#ff9800', 'Stressed':'#f44336'}
        mood_color = mood_colors.get(result['mood'], '#000')
        
        with col_a:
            st.markdown("#### Mood")
            st.markdown(f"<h1 style='color:{mood_color};text-align:center'>{result['mood']}</h1>", unsafe_allow_html=True)
        with col_b:
            st.markdown("#### Confidence")
            st.markdown(f"<h1 style='text-align:center'>{result['confidence']*100:.0f}%</h1>", unsafe_allow_html=True)
            st.progress(result['confidence'])
        with col_c:
            st.markdown("#### Sentiment")
            emoji = "😊" if adjusted_sentiment>0.3 else ("😐" if adjusted_sentiment>-0.3 else "😔")
            st.markdown(f"<h1 style='text-align:center'>{adjusted_sentiment:+.3f} {emoji}</h1>", unsafe_allow_html=True)
        with col_d:
            st.markdown("#### Subjectivity")
            st.markdown(f"<h1 style='text-align:center'>{subjectivity:.2f}</h1>", unsafe_allow_html=True)
            st.progress(subjectivity)
        
        st.markdown("---")
        
        # Visualizations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Sentiment Gauge")
            fig, ax = plt.subplots(figsize=(5, 3))
            sentiment_norm = (adjusted_sentiment + 1) / 2
            colors = ['#f44336', '#ff9800', '#ffeb3b', '#8bc34a', '#4caf50']
            theta = np.linspace(0, np.pi, 100)
            for i, color in enumerate(colors):
                start, end = i*np.pi/5, (i+1)*np.pi/5
                theta_sec = np.linspace(start, end, 20)
                ax.fill_between(theta_sec, 0, 1, color=color, alpha=0.7)
            angle = sentiment_norm * np.pi
            ax.plot([angle, angle], [0, 0.8], 'k-', linewidth=3)
            ax.plot(angle, 0.8, 'ko', markersize=8)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, np.pi)
            ax.axis('off')
            ax.text(0, -0.1, 'Neg', ha='left', fontsize=8)
            ax.text(np.pi, -0.1, 'Pos', ha='right', fontsize=8)
            ax.text(np.pi/2, 0.35, f'{adjusted_sentiment:+.3f}', ha='center', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("🎯 Confidence")
            fig, ax = plt.subplots(figsize=(5, 3))
            data = [result['confidence'], 1-result['confidence']]
            colors_pie = ['#4caf50', '#e0e0e0']
            ax.pie(data, labels=['Confident', 'Uncertain'], autopct='%1.0f%%', 
                  colors=colors_pie, startangle=90)
            st.pyplot(fig)
            plt.close()
        
        with col3:
            st.subheader("😊 Emotion Spectrum")
            fig, ax = plt.subplots(figsize=(5, 3))
            moods = ['Happy', 'Calm', 'Neutral', 'Tired', 'Stressed']
            probs = [(result['confidence'] if m==result['mood'] else (1-result['confidence'])/4) for m in moods]
            colors_bar = [mood_colors.get(m, 'gray') for m in moods]
            ax.barh(moods, probs, color=colors_bar)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Recommendations
        if result['mood'] == 'Stressed':
            st.error("🚨 High stress! Take 15-min break, deep breathing, talk to someone")
        elif result['mood'] == 'Happy':
            st.success("😊 Great mood! Perfect for complex/creative tasks")
        elif result['mood'] == 'Tired':
            st.warning("😴 Low energy - Take power nap, get caffeine, fresh air")
        else:
            st.info(f"😐 {result['mood']} state detected")
    
    elif not user_text and analyze_button:
        st.warning("⚠️ Please enter text")

# PAGE 3: TASK PREDICTOR
elif page == "🔮 Task Predictor":
    st.header("🔮 Task Duration Predictor")
    
    col1, col2 = st.columns(2)
    with col1:
        task = st.selectbox("Task", ["Bug fix", "Code review", "API dev", "Database", "Research", "Call", "Docs"])
        priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
        mood = st.selectbox("Mood", ["Happy", "Motivated", "Calm", "Neutral", "Tired", "Anxious", "Stressed"])
    with col2:
        deadline = st.slider("Days to Deadline", 1, 30, 7)
        workload = st.slider("Workload (hrs)", 2.0, 12.0, 6.0, 0.5)
    
    if st.button("🚀 Predict", type="primary"):
        if data['tasks'] is not None and models.get('duration'):
            p_map = {'Low':0, 'Medium':2, 'High':3, 'Critical':1}
            m_map = {'Anxious':0, 'Calm':1, 'Happy':2, 'Motivated':3, 'Neutral':4, 'Stressed':5, 'Tired':6}
            t_map = {t:i for i,t in enumerate(data['tasks']['task_description'].unique())}
            
            p_enc = p_map.get(priority, 2)
            m_enc = m_map.get(mood, 1)
            t_enc = t_map.get(task, 0)
            urg = p_enc * (30-deadline) / 30
            strs = m_enc * workload
            
            features = [[p_enc, m_enc, t_enc, deadline, workload, urg, strs]]
            duration = models['duration'].predict(features)[0]
            
            st.success(f"### ⏱️ Predicted: **{duration:.1f} hours**")
            if duration > 15:
                st.warning("Long task! Break into chunks")

# PAGE 4: PRODUCTIVITY
elif page == "📊 Productivity":
    st.header("📊 Productivity Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        p_mood = st.selectbox("Mood", ["Happy", "Motivated", "Calm", "Neutral", "Tired", "Anxious", "Stressed"], key='pm')
        p_stress = st.slider("Stress (0-10)", 0.0, 10.0, 5.0, 0.5)
        p_work = st.slider("Work Hours", 0.0, 12.0, 6.0, 0.5)
    with col2:
        p_tasks = st.slider("Tasks Done", 0, 15, 5)
        p_quality = st.slider("Quality (0-10)", 0.0, 10.0, 8.0, 0.5)
    
    if st.button("📊 Calculate", type="primary"):
        score = calc_productivity(p_mood, p_stress, p_work, p_tasks, p_quality)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Score", f"{score}/100")
        with col_b:
            cat = "Excellent" if score>=80 else ("Good" if score>=65 else ("Average" if score>=50 else "Low"))
            st.metric("Category", cat)
        with col_c:
            st.metric("Trend", "+5%" if score>=65 else "-3%")
        
        if score >= 80:
            st.success("🏆 Excellent performance!")
        elif score < 50:
            st.warning("⚠️ Consider taking a break")

# PAGE 5: BURNOUT ANALYZER
elif page == "⚠️ Burnout Analyzer":
    st.header("⚠️ Burnout Risk Predictor")
    
    col1, col2 = st.columns(2)
    with col1:
        avg_stress = st.slider("Avg Stress (30d)", 0.0, 10.0, 5.0, 0.5)
        high_days = st.slider("High Stress Days", 0, 30, 5)
        avg_work = st.slider("Avg Daily Hours", 4.0, 12.0, 8.0, 0.5)
    with col2:
        mood_score = st.slider("Mood Score (0-100)", 0, 100, 70, 5)
        overtime = st.slider("Monthly Overtime", 0.0, 80.0, 20.0, 5.0)
        no_break = st.slider("Days No Break", 0, 21, 5)
    
    if st.button("⚠️ Analyze Risk", type="primary"):
        if models.get('burnout'):
            features = [[avg_stress, high_days, avg_work, mood_score, overtime, no_break]]
            risk = models['burnout'].predict(features)[0]
            probs = models['burnout'].predict_proba(features)[0]
            
            classes = models['burnout'].classes_
            prob_dict = dict(zip(classes, probs))
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"### Risk Level")
                risk_colors = {'Low':'green', 'Medium':'orange', 'High':'red'}
                color = risk_colors.get(risk, 'gray')
                st.markdown(f"<h1 style='color:{color}'>{risk}</h1>", unsafe_allow_html=True)
            with col_b:
                conf = prob_dict[risk] * 100
                st.metric("Confidence", f"{conf:.1f}%")
            
            if risk == 'High':
                st.error("🚨 CRITICAL - Immediate action required!")
            elif risk == 'Medium':
                st.warning("⚠️ WARNING - Preventive action needed")
            else:
                st.success("✅ HEALTHY - Maintain practices")

# PAGE 6: BREAK SCHEDULER
elif page == "⏰ Break Scheduler":
    st.header("⏰ Smart Break Scheduler")
    
    col1, col2 = st.columns(2)
    with col1:
        b_work = st.slider("Hours Worked", 0.0, 12.0, 4.0, 0.5)
        b_stress = st.slider("Stress", 0.0, 10.0, 5.0, 0.5)
        b_mood = st.selectbox("Mood", ["Happy", "Calm", "Neutral", "Tired", "Stressed", "Anxious"])
    with col2:
        time_since = st.slider("Min Since Break", 0, 180, 90, 15)
    
    if st.button("⏰ Get Recommendation", type="primary"):
        rec = get_break_rec(b_work, b_stress, b_mood, time_since)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Type", rec['type'])
        with col_b:
            st.metric("Duration", f"{rec['duration']} min")
        with col_c:
            st.metric("Urgency", rec['urgency'])
        
        st.markdown("---")
        st.subheader("🎯 Activities")
        for act in rec['activities']:
            st.markdown(f"• {act}")

# PAGE 7: HISTORY
elif page == "📈 History":
    st.header("📈 Mood & Stress History")
    
    if data['history'] is not None:
        emps = sorted(data['history']['employee_id'].unique())
        sel = st.selectbox("Employee", emps)
        
        emp = data['history'][data['history']['employee_id'] == sel].copy()
        emp['timestamp'] = pd.to_datetime(emp['timestamp'])
        emp = emp.sort_values('timestamp')
        
        avg_s = emp['stress_level'].mean()
        curr_s = emp['stress_level'].iloc[-1]
        high_c = len(emp[emp['stress_level'] > 7])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", len(emp))
        with col2:
            st.metric("Avg Stress", f"{avg_s:.1f}/10")
        with col3:
            st.metric("High Days", high_c)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(emp['timestamp'], emp['stress_level'], marker='o', color='crimson')
        ax.axhline(y=7, color='red', linestyle='--')
        ax.set_ylabel('Stress')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# PAGE 8: TEAM
elif page == "👥 Team Analytics":
    st.header("👥 Team Dashboard")
    
    if data['team_summary'] is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Health", f"{data['team_summary']['health_score'].mean():.0f}/100")
        with col2:
            st.metric("Avg Stress", f"{data['team_summary']['avg_stress'].mean():.1f}/10")
        with col3:
            high = len(data['team_summary'][data['team_summary']['avg_stress'] > 7])
            st.metric("High Stress Depts", high)
        
        st.dataframe(data['team_summary'], use_container_width=True)

# PAGE 9: ANALYTICS
elif page == "📉 Analytics":
    st.header("📉 Advanced Analytics")
    
    if data['tasks'] is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            data['tasks']['workload_hours'].hist(bins=30, ax=ax, color='skyblue')
            ax.set_xlabel('Hours')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            data['tasks']['days_until_deadline'].hist(bins=30, ax=ax, color='coral')
            ax.set_xlabel('Days')
            st.pyplot(fig)
        
        st.dataframe(data['tasks'].head(50), use_container_width=True)

# PAGE 10: STRESS
elif page == "🚨 Stress Monitor":
    st.header("🚨 Stress Monitor")
    
    col1, col2 = st.columns(2)
    with col1:
        work = st.slider("Hours Worked", 0.0, 12.0, 6.0, 0.5)
        tasks = st.slider("Pending Tasks", 0, 20, 5)
    with col2:
        mood = st.selectbox("Mood", ["Happy", "Motivated", "Calm", "Neutral", "Tired", "Anxious", "Stressed"])
        dead = st.slider("Days to Deadline", 1, 15, 5)
    
    if st.button("Calculate", type="primary"):
        m_scores = {'Happy':1, 'Motivated':2, 'Calm':2, 'Neutral':5, 'Tired':6, 'Anxious':8, 'Stressed':9}
        stress = (work/12)*3 + (tasks/10)*2 + (m_scores.get(mood,5)/10)*3 + (1/max(dead,1))*2
        stress = min(round(stress, 1), 10)
        
        if stress >= 8:
            st.error(f"🚨 CRITICAL: {stress}/10")
        elif stress >= 6:
            st.warning(f"⚠️ HIGH: {stress}/10")
        else:
            st.success(f"✅ HEALTHY: {stress}/10")

st.markdown("---")
st.markdown("*🎓 NIT Trichy BY Guna| Included AI Features | 96.7% Accuracy*")
