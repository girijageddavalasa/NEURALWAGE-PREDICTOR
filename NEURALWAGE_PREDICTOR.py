import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import time
from lightgbm import LGBMClassifier

# --- GLOBAL STYLING ---
custom_css = """
<style>
body {
    background-color: #0f0f0f;
    color: #ffffff;
}
section.main > div {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 30px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}
h1, h2, h3 {
    color: #00f7ff;
    text-shadow: 0 0 15px #00f7ff;
}
.stButton>button {
    background: linear-gradient(to right, #00f7ff, #007cf0);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
}
.stSelectbox>div>div {
    color: white;
}
.metric-label {
    font-size: 18px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv("adult.csv")

    data.replace('?', pd.NA, inplace=True)
    data['workclass'].fillna('NotListed', inplace=True)
    data['occupation'].fillna('others', inplace=True)

    data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
    data = data[~data['education'].isin(['5th-6th', '1st-4th', 'Preschool'])]
    data.drop(columns=['fnlwgt', 'education', 'relationship', 'race', 'native-country'], inplace=True)
    data = data[(data['age'] <= 75) & (data['age'] >= 17)]

    X = data.drop(columns=['income'])
    y = data['income']

    numerical_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_cols = ['workclass', 'marital-status', 'occupation', 'gender']

    preprocessor = ColumnTransformer(transformers=[
        ('num', MinMaxScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    all_feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

    return X_processed_df, y, preprocessor, data

# Load Data
X, y, preprocessor, original_data = load_and_preprocess_data()

# --- Model ---
@st.cache_resource
def train_model(X_train, y_train):
    clf = LGBMClassifier(random_state=2)
    clf.fit(X_train, y_train)
    return clf

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y)
model = train_model(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)

# --- Page Layout ---
st.set_page_config(layout="wide", page_title="💸 LightGBM Wage Predictor")
st.title("💼 LightGBM Wage Predictor 💸")

with st.container():
    st.markdown("Welcome to the most futuristic salary prediction platform! Predict salaries with flair. 🔮")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🤖 Predict Salary", "📉 Model Insights"])

# --- Tab 1: Data Explorer ---
with tab1:
    st.subheader("Explore the Dataset 👀")
    st.write("Dive into employee data used to build the predictor!")

    with st.expander("📋 View Sample Records"):
        st.dataframe(original_data.head(15), use_container_width=True)

    colA, colB = st.columns(2)

    with colA:
        selected_category = st.selectbox("Categorical column:", ['workclass', 'marital-status', 'occupation', 'gender'])
        fig_cat = plt.figure(figsize=(10, 5))
        sns.countplot(data=original_data, x=selected_category, hue='income', palette='cool')
        plt.xticks(rotation=45)
        st.pyplot(fig_cat)

    with colB:
        selected_numerical = st.selectbox("Numerical column:", ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week'])
        fig_num = plt.figure(figsize=(10, 5))
        sns.histplot(data=original_data, x=selected_numerical, hue='income', kde=True, palette='rocket', bins=30)
        st.pyplot(fig_num)

# --- Tab 2: Prediction UI ---
with tab2:
    st.subheader("🧠 AI Salary Prediction")
    st.markdown("Customize employee attributes below and hit **Predict**!")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 17, 75, 30)
        workclass = st.selectbox("Workclass", sorted(original_data['workclass'].unique()))
        edu_num = st.slider("Education (Years)", 1, 16, 10)
    with col2:
        marital = st.selectbox("Marital Status", sorted(original_data['marital-status'].unique()))
        occupation = st.selectbox("Occupation", sorted(original_data['occupation'].unique()))
        gender = st.selectbox("Gender", sorted(original_data['gender'].unique()))
    with col3:
        gain = st.number_input("Capital Gain", 0, 99999, 0)
        loss = st.number_input("Capital Loss", 0, 99999, 0)
        hours = st.slider("Weekly Hours", 1, 99, 40)

    input_df = pd.DataFrame([[age, workclass, edu_num, marital, occupation, gender, gain, loss, hours]],
        columns=['age', 'workclass', 'educational-num', 'marital-status', 'occupation', 'gender',
                 'capital-gain', 'capital-loss', 'hours-per-week'])

    if st.button("🎯 Predict Now"):
        with st.spinner("Crunching LightGBM model..."):
            time.sleep(1.5)
            input_processed = preprocessor.transform(input_df)
            if hasattr(input_processed, "toarray"):
                input_processed = input_processed.toarray()

            result = model.predict(input_processed)[0]
            if result == '>50K':
                st.success(f"🎉 This employee is likely to earn **>50K**. High earner detected!")
                st.balloons()
            else:
                st.warning(f"🧾 This employee is predicted to earn **<=50K**. Consider reskilling opportunities.")

# --- Tab 3: Model Performance ---
with tab3:
    st.subheader("📉 Performance Overview")
    st.metric("✅ Accuracy", f"{accuracy * 100:.2f}%")

    st.markdown("### 🧮 Confusion Matrix")
    cm = confusion_matrix(ytest, ypred, labels=model.classes_)
    fig_cm, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='cool', ax=ax, values_format='d')
    st.pyplot(fig_cm)

    st.markdown("""---  
    🔍 **Legend:**  
    - **TP**: True Positive  
    - **TN**: True Negative  
    - **FP**: False Positive  
    - **FN**: False Negative  
    """)

# --- Footer ---
st.markdown("---")
st.markdown("💡 Built with neural love using Streamlit • Customized uniquely for YOU 🧠⚡")
