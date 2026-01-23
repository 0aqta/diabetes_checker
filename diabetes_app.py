import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai

# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Checker – Canada",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load Gemini API key (from secrets.toml or environment) ────────────────
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("Gemini API key not found in .streamlit/secrets.toml\n\n"
             "Please create file .streamlit/secrets.toml with:\n"
             'GEMINI_API_KEY = "your-key-here"')
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ── Load model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_diabetes_model.pkl')
    except FileNotFoundError:
        st.error("Model file 'xgboost_diabetes_model.pkl' not found.\nPlease place it in the same folder.")
        st.stop()

model = load_model()

# Exact column order your model expects
FEATURE_ORDER = [
    'SEX', 'GENHLTH', 'PHYSHLTH', 'CVDSTRK3', 'RFHLTH', 'TOTINDA', '_MICHD',
    'RACE', 'AGEGRP', 'BMI', 'EDUCATION', 'SMOKER', 'Year', 'INCOME'
]

# ── TITLE & DISCLAIMER ────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Checker – Canada")

st.info("""
**Important – Please read**  
This is an **educational tool only** — it is **not** a medical diagnosis.  
Always consult your family doctor, nurse practitioner or certified diabetes educator.  
Results are estimates based on survey data patterns.
""", icon="⚠️")

# ── INPUT SECTION ─────────────────────────────────────────────────────────
st.subheader("Tell us about yourself")

col1, col2, col3 = st.columns(3)

with col1:
    # SEXVAR
    sex_label = st.selectbox(
        "Sex",
        options=["Male", "Female"],
        index=1
    )
    sex = 1.0 if "Male" in sex_label else 2.0

    # GENHLTH
    genhlth_label = st.selectbox(
        "How would you rate your general health?",
        options=["Excellent ", "Very good ", "Good ", "Fair ", "Poor "],
        index=2
    )
    genhlth_map = {"Excellent ":1, "Very good ":2, "Good ":3, "Fair ":4, "Poor ":5}
    genhlth = genhlth_map[genhlth_label]

    # PHYSHLTH
    physhlth = st.slider(
        "During the past 30 days, how many days was your physical health not good?",
        0, 30, 5, help="0-30 = Number of days, 88 = None"
    )
    if physhlth == 0:
        physhlth = 88.0  # none

with col2:
    # CVDSTRK3
    cvdstrk3_label = st.selectbox(
        "Ever told by a doctor that you had a stroke?",
        options=["Yes ", "No "],
        index=1
    )
    cvdstrk3 = 1.0 if "Yes" in cvdstrk3_label else 2.0

    # _RFHLTH
    rfhlth_label = st.selectbox(
        "Do you get enough fruits and vegetables?",
        options=["Yes", "No"],
        index=0
    )
    rfhlth = 1.0 if "Yes" in rfhlth_label else 2.0

    # _TOTINDA
    totinda_label = st.selectbox(
        "Did you do any physical activity or exercise in the past month?",
        options=["Yes ", "No "],
        index=0
    )
    totinda = 1.0 if "Yes" in totinda_label else 2.0

    # _MICHD
    michd_label = st.selectbox(
        "Ever told you had coronary heart disease or heart attack?",
        options=["Yes", "No"],
        index=1
    )
    michd = 1.0 if "Yes" in michd_label else 2.0

with col3:
    # _RACE
    race_options = [
        "White only, non-Hispanic ", "Black only, non-Hispanic ",
        "American Indian/Alaskan Native only ", "Asian only ",
        "Native Hawaiian/Other Pacific Islander only ", "Other race only ",
        "Multiracial, non-Hispanic ", "Hispanic "
    ]
    race_label = st.selectbox("Race / Ethnicity", race_options, index=0)
    race = race_options.index(race_label) + 1

    # _AGEG5YR
    age_options = [
        "18–24 ", "25–29 ", "30–34 ", "35–39 ", "40–44 ", "45–49 ",
        "50–54 ", "55–59 ", "60–64 ", "65–69 ", "70–74 ", "75–79 ", "80+ "
    ]
    age_label = st.selectbox("Your age group", age_options, index=7)
    agegrp = age_options.index(age_label) + 1

    # BMI
    bmi_real = st.number_input("Your Body Mass Index (BMI)", 10.0, 60.0, 25.5, 0.1)
    bmi = bmi_real * 100  # scale like training data

    # _EDUCAG
    educ_options = [
        "Did not graduate high school", "Graduated high school",
        "Attended college/technical school", "Graduated college/technical school"
    ]
    educ_label = st.selectbox("Highest level of education", educ_options, index=2)
    education = educ_options.index(educ_label) + 1

    # _SMOKER3
    smoker_options = ["Current – every day", "Current – some days", "Former smoker", "Never smoked"]
    smoker_label = st.selectbox("Smoking status", smoker_options, index=3)
    smoker = smoker_options.index(smoker_label) + 1

    # INCOME
    income_options = [
        "<$15,000", "$15,000–<$25,000", "$25,000–<$35,000 ", "$35,000–<$50,000 ", "$50,000+ "
    ]
    income_label = st.selectbox("Income category", income_options, index=4)
    income = income_options.index(income_label) + 1

# Year (fixed)
year = 2023.0

# ── PREDICTION ────────────────────────────────────────────────────────────
if st.button("Calculate My Diabetes Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing your answers..."):
        input_dict = {
            'SEX': sex, 'GENHLTH': genhlth, 'PHYSHLTH': physhlth, 'CVDSTRK3': cvdstrk3,
            'RFHLTH': rfhlth, 'TOTINDA': totinda, '_MICHD': michd, 'RACE': race,
            'AGEGRP': agegrp, 'BMI': bmi, 'EDUCATION': education,
            'SMOKER': smoker, 'Year': year, 'INCOME': income
        }

        df = pd.DataFrame([input_dict])[FEATURE_ORDER]

        try:
            prob = model.predict_proba(df)[0][1]
        except Exception as e:
            st.error(f"Model prediction failed – possible input mismatch.\n{str(e)}")
            st.stop()

        if prob >= 0.52:
            level, color = "High", "error"
        elif prob >= 0.25:
            level, color = "Medium", "warning"
        else:
            level, color = "Low", "success"

        getattr(st, color)(f"**Estimated diabetes risk probability: {prob*100:.1f}%**")
        st.subheader(f"Your estimated risk level: **{level}**")

        # ── GEMINI ADVICE ─────────────────────────────────────────────────
        st.subheader("Personalized Prevention Advice (Powered by Gemini)")

        prompt = f"""You are a friendly, supportive diabetes prevention coach from Diabetes Canada, based in Ontario.

User profile:
- Sex: {sex_label}
- Age group: {age_label}
- BMI: {bmi_real:.1f}
- General health: {genhlth_label}
- Days of poor physical health last month: {physhlth}
- Stroke history: {cvdstrk3_label}
- Fruit/veg intake: {rfhlth_label}
- Physical activity last month: {totinda_label}
- Heart disease/heart attack history: {michd_label}
- Race/ethnicity: {race_label}
- Education: {educ_label}
- Smoking status: {smoker_label}

Estimated risk level: {level} ({prob*100:.1f}% probability)

Write warm, encouraging, practical advice (250–400 words) focusing on:
- Healthy eating
- Regular physical activity
- Weight management
- Stress reduction
- Regular screening/check-ups

Include 3–5 specific, achievable tips.

Mention Toronto/GTA resources:
- Diabetes Education Program at Toronto General Hospital or Sunnybrook Health Sciences Centre
- Community health programs in GTA
- Diabetes Canada free virtual classes & support groups
- Diabetes Canada helpline: 1-800-226-8464 or www.diabetes.ca

End with strong disclaimer:
"This is NOT medical advice. Please consult your family doctor, nurse practitioner, or certified diabetes educator for personalized care."
"""

        try:
            gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            response = gemini_model.generate_content(prompt)
            st.markdown(response.text)
        except Exception as e:
            st.error(f"Could not generate advice.\n\n{str(e)}")

# ── FOOTER ────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Educational awareness tool • Built in GTA • Model: XGBoost • AI: Google Gemini")