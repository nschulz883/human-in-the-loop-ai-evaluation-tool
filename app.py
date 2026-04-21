import streamlit as st 
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# ML FUNCTIONS
# =========================

def train_model(df):
    X = df["prompt"] + " " + df["response"]
    y = df["helpfulness"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    return model, vectorizer


def predict_helpfulness(model, vectorizer, prompt, response):
    text = prompt + " " + response
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction


# =========================
# UI
# =========================

st.title(" AI Response Quality Rater")

prompt = st.text_area("Enter Prompt")
response = st.text_area("Enter AI Response")

st.write("### Rate the Response")

helpfulness = st.slider("Helpfulness (1-5)", 1, 5)
correctness = st.selectbox("Correctness", ["Correct", "Incorrect"])
clarity = st.selectbox("Clarity", ["Clear", "Confusing"])

# =========================
# LOAD DATA + TRAIN MODEL
# =========================

model = None
vectorizer = None
predicted_score = None

if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv")

    # Only train if enough data
    if len(df) > 5:
        model, vectorizer = train_model(df)

        if prompt and response:
            predicted_score = predict_helpfulness(model, vectorizer, prompt, response)
            st.write("### 🤖 AI Predicted Helpfulness:", predicted_score)

# =========================
# SAVE DATA
# =========================

if st.button("Submit Evaluation"):
    new_data = pd.DataFrame([{
        "prompt": prompt,
        "response": response,
        "helpfulness": helpfulness,
        "correctness": correctness,
        "clarity": clarity
    }])

    file = "data.csv"

    if os.path.exists(file):
        df = pd.read_csv(file)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data

    df.to_csv(file, index=False)
    st.success("Evaluation saved!")

    # Compare AI vs Human
    if predicted_score is not None:
        diff = abs(predicted_score - helpfulness)
        st.write("### 🔍 AI vs Human Difference:", diff)

# =========================
# STATS + CHARTS
# =========================

st.write("##  Evaluation Stats")

if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv")

    st.write("Total evaluations:", len(df))

    avg_helpfulness = df["helpfulness"].mean()
    st.write("Average Helpfulness:", round(avg_helpfulness, 2))

    correct_pct = (df["correctness"] == "Correct").mean() * 100
    st.write("Correctness %:", round(correct_pct, 2), "%")

    if st.checkbox("Show Dataset"):
        st.write(df)

    # -------------------------
    # CHARTS
    # -------------------------

    # Helpfulness chart
    st.write("###  Helpfulness Distribution")
    help_counts = df["helpfulness"].value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.bar(help_counts.index.astype(str), help_counts.values)
    ax.set_xlabel("Helpfulness Score")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    # Correctness pie chart
    st.write("### 🥧 Correctness Breakdown")
    correct_counts = df["correctness"].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.pie(correct_counts.values, labels=correct_counts.index, autopct='%1.1f%%')

    st.pyplot(fig2)

    # Clarity chart
    st.write("###  Clarity Distribution")
    clarity_counts = df["clarity"].value_counts()

    fig3, ax3 = plt.subplots()
    ax3.bar(clarity_counts.index, clarity_counts.values)
    ax3.set_xlabel("Clarity")
    ax3.set_ylabel("Count")

    st.pyplot(fig3)

else:
    st.info("No data yet. Submit evaluations to train the model.")

# =========================
# FOOTER
# =========================

st.divider()
st.caption("Built as an AI Trainer System for evaluating model responses")
