
import streamlit as st
import pandas as pd
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
st.set_page_config(page_title="ScoreScope", layout="wide")
st.markdown("""
    <style>
    /* ===== GLOBAL FONT & BACKGROUND ===== */
    * {
        font-family: 'Poppins', sans-serif !important;
    }

    /* ===== INPUT FIELDS & DROPDOWNS ===== */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #000000 !important;  /* solid black box */
        color: #FFFFFF !important;              /* white text */
        border: 1.5px solid #555 !important;    /* dark grey outline */
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 16px !important;
    }

    /* ===== DROPDOWN MENU OPTIONS ===== */
    ul[role="listbox"] {
        background-color: #000000 !important;  /* solid black dropdown menu */
        color: #FFFFFF !important;
        border: 1px solid #666 !important;
        border-radius: 8px !important;
    }

    li[role="option"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        font-weight: 500 !important;
        padding: 6px 10px !important;
    }

    li[role="option"]:hover {
        background-color: #1a1a1a !important;  /* subtle grey hover effect */
    }

    /* ===== LABELS ===== */
    label, .stTextInput label, .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ===== BUTTON ===== */
    div.stButton > button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1.5px solid #444 !important;
        border-radius: 12px !important;
        padding: 10px 25px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
    }

    div.stButton > button:hover {
        background-color: #1a1a1a !important;
        transform: scale(1.04);
        box-shadow: 0px 6px 12px rgba(0,0,0,0.5);
    }

    /* ===== SLIDERS ===== */
    .stSlider label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ===== CONTAINER PANEL ===== */
    section.main > div {
        background: rgba(0,0,0,0.55);
        border-radius: 15px;
        padding: 25px;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top:60px;'></div>", unsafe_allow_html=True)

st.markdown("Enter Student Name")

name = st.text_input("Student Name", placeholder="Enter your name")

if name.strip() != "":
    st.markdown(
        f"<h4 style='color:#00e6e6;'>Hi, <b>{name}</b>! Please fill the form below to generate your personalised score !</h4>",
        unsafe_allow_html=True
    )
    st.markdown("---")   # separator line


def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}
        /* Header and prediction styles */
        .center-title {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            margin-bottom: 6px;
        }}
        .center-title img {{ width: 70px; height: auto; border-radius: 8px; }}
        .center-title h1 {{ font-size: 2.4rem; margin: 0; color: #ffffff; padding-top: 8px; }}
        .prediction-text {{
            font-size: 1.3rem;
            font-weight: 800;
            color: #00ffff;
            font-family: 'Trebuchet MS', sans-serif;
            text-transform: uppercase;
            margin-left: 8px;
            display: inline;
        }}
        .stButton>button {{
            background-color: #0072ff;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }}
        .block-container {{ padding-top: 1rem; padding-bottom: 1rem; max-width: 95%; }}
        h1,h2,h3,label,p,span,div {{ color: #ffffff !important; }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background("Background.png")

@st.cache_data
def load_data():
    return pd.read_csv("cstperformance01.csv")

df = load_data()

required_columns = [
    'Gender','Age','Department','Attendance (%)',
    'Midterm_Score','Final_Score','Assignments_Avg',
    'Projects_Score','Study_Hours_per_Week',
    'Extracurricular_Activities','Quizzes_Avg',
    'Internet_Access_at_Home','Parent_Education_Level',
    'Family_Income_Level','Stress_Level (1-10)',
    'Sleep_Hours_per_Night','Total_Score'
]
if not all(col in df.columns for col in required_columns):
    st.error("Dataset missing required columns; check cstperformance01.csv")
    st.stop()

df = df[required_columns]
cat_cols = [
    'Gender','Department','Extracurricular_Activities',
    'Internet_Access_at_Home','Parent_Education_Level','Family_Income_Level'
]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    encoders[c] = le
    
X = df.drop(columns=['Total_Score'])
y = df['Total_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
st.markdown(
    """
    <div style='display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 10px; margin-bottom: 5px;'>
        <img src='data:image/png;base64,{}' width='40' style='border-radius: 10px;'/>
        <h1 style='color: white; margin: 0; font-size: 2.2rem;'>ScoreScope</h1>
    </div>
    """.format(base64.b64encode(open("sslogo.png", "rb").read()).decode()),
    unsafe_allow_html=True
)
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        attendance = st.slider("Attendance (%)", 0, 100, 0)
        study_hours = st.slider("Study Hours/Week", 0, 60, 0)
        stress = st.slider("Stress Level (1-10)", 1, 10, 0)
        sleep = st.slider("Sleep Hours/Night", 0, 12, 0)
        midterm = st.number_input("Midterm Score", min_value=0, max_value=100, value=0)

    with c2:
        final = st.number_input("Final Score", min_value=0, max_value=100, value=0)
        assignments = st.number_input("Assignments Avg", min_value=0, max_value=100, value=0)
        projects = st.number_input("Projects Score", min_value=0, max_value=100, value=0)
        quizzes = st.number_input("Quizzes Avg", min_value=0, max_value=100, value=0)
        gender = st.selectbox("Gender", encoders['Gender'].classes_)

    with c3:
        age = st.number_input("Age", min_value=10, max_value=30, value=10)
        dept = st.selectbox("Department", encoders['Department'].classes_)
        activities = st.selectbox("Extracurricular Activities", encoders['Extracurricular_Activities'].classes_)
        internet = st.selectbox("Internet Access at Home", encoders['Internet_Access_at_Home'].classes_)
        parent_edu = st.selectbox("Parent Education Level", encoders['Parent_Education_Level'].classes_)
        income = st.selectbox("Family Income Level", encoders['Family_Income_Level'].classes_)

    col_btn, col_text = st.columns([1, 4])
    with col_btn:
        submitted = st.form_submit_button("Predict")
    with col_text:
        result_placeholder = st.empty()
if submitted:
    # Build DataFrame exactly like training format
    input_df = pd.DataFrame([{
        'Gender': encoders['Gender'].transform([gender])[0],
        'Age': age,
        'Department': encoders['Department'].transform([dept])[0],
        'Attendance (%)': attendance,
        'Midterm_Score': midterm,
        'Final_Score': final,
        'Assignments_Avg': assignments,
        'Projects_Score': projects,
        'Study_Hours_per_Week': study_hours,
        'Extracurricular_Activities': encoders['Extracurricular_Activities'].transform([activities])[0],
        'Quizzes_Avg': quizzes,
        'Internet_Access_at_Home': encoders['Internet_Access_at_Home'].transform([internet])[0],
        'Parent_Education_Level': encoders['Parent_Education_Level'].transform([parent_edu])[0],
        'Family_Income_Level': encoders['Family_Income_Level'].transform([income])[0],
        'Stress_Level (1-10)': stress,
        'Sleep_Hours_per_Night': sleep
    }])

    predicted_score = model.predict(input_df)[0]

    # Grade
    if predicted_score >= 90:
        grade = "A"
    elif predicted_score >= 75:
        grade = "B"
    elif predicted_score >= 60:
        grade = "C"
    elif predicted_score >= 40:
        grade = "D"
    else:
        grade = "E"

    st.success(
        f"Hey {name}, your predicted score is {predicted_score:.2f} "
        f"and this gives a grade of {grade}."
    )


