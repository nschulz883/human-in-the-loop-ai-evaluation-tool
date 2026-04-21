
# 🧠 AI Response Quality Evaluation System

A human-in-the-loop machine learning system built with Streamlit that evaluates AI-generated responses and collects labeled feedback to improve model performance over time.

---

## 🚀 Features

- Evaluate AI responses based on:
  - Helpfulness (1–5)
  - Correctness (Correct / Incorrect)
  - Clarity (Clear / Confusing)

- Stores labeled data for future training
- Visual analytics dashboard with charts
- Basic ML model predicts helpfulness scores
- Human vs AI comparison system

---

## 🧠 Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib

---

## 📊 What it does

1. User enters prompt and AI response  
2. System allows human evaluation  
3. Data is stored in dataset  
4. ML model predicts response quality  
5. Charts show dataset insights



---

## 🧠 Purpose

This project simulates real-world AI training workflows used in:
- AI model evaluation
- Data annotation pipelines
- Human-in-the-loop machine learning systems

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

