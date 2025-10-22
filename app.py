# app.py - Clean final version (single submit block, no NameError)
import streamlit as st
import pandas as pd
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="ScoreScope", layout="wide")

# ------------------------
# Background styling (no blur)
# ------------------------
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

# call background (ensure bckgrnd.png exists)
set_background("bckgrnd.png")

# ------------------------
# Load dataset
# ------------------------
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

# ------------------------
# Encode categorical columns
# ------------------------
cat_cols = [
    'Gender','Department','Extracurricular_Activities',
    'Internet_Access_at_Home','Parent_Education_Level','Family_Income_Level'
]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])
    encoders[c] = le

# ------------------------
# Train model (simple linear)
# ------------------------
X = df.drop(columns=['Total_Score'])
y = df['Total_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------
# Title and layout (center logo + title)
# ------------------------
st.markdown(
    """
    <div style='display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 10px; margin-bottom: 5px;'>
        <img src='data:image/png;base64,{}' width='40' style='border-radius: 10px;'/>
        <h1 style='color: white; margin: 0; font-size: 2.4rem;'>ScoreScope</h1>
    </div>
    """.format(base64.b64encode(open("sslogo.png", "rb").read()).decode()),
    unsafe_allow_html=True
)

# Input form (ALL controls inside the form)
# ------------------------
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        study_hours = st.slider("Study Hours/Week", 0, 60, 15)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        sleep = st.slider("Sleep Hours/Night", 0, 12, 7)
        midterm = st.number_input("Midterm Score", min_value=0, max_value=100, value=50)

    with c2:
        final = st.number_input("Final Score", min_value=0, max_value=100, value=60)
        assignments = st.number_input("Assignments Avg", min_value=0, max_value=100, value=70)
        projects = st.number_input("Projects Score", min_value=0, max_value=100, value=65)
        quizzes = st.number_input("Quizzes Avg", min_value=0, max_value=100, value=55)
        gender = st.selectbox("Gender", encoders['Gender'].classes_)

    with c3:
        age = st.number_input("Age", min_value=10, max_value=30, value=18)
        dept = st.selectbox("Department", encoders['Department'].classes_)
        activities = st.selectbox("Extracurricular Activities", encoders['Extracurricular_Activities'].classes_)
        internet = st.selectbox("Internet Access at Home", encoders['Internet_Access_at_Home'].classes_)
        parent_edu = st.selectbox("Parent Education Level", encoders['Parent_Education_Level'].classes_)
        income = st.selectbox("Family Income Level", encoders['Family_Income_Level'].classes_)

    # button and placeholder (INSIDE the form)
    col_btn, col_text = st.columns([1, 4])
    with col_btn:
        submitted = st.form_submit_button("Predict")
    with col_text:
        result_placeholder = st.empty()

## ------------------------
# Single prediction block (after form). input_data created here.
# ------------------------
if submitted:
    input_data = {
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
    }

    # create input dataframe and predict
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]

    # display result next to button via placeholder
    result_placeholder.markdown(
        f"<p class='prediction-text'>YOUR PREDICTED SCORE IS: {pred:.2f}</p>",
        unsafe_allow_html=True
    )
