import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_models

st.set_page_config(page_title="Placement Dashboard", layout="wide")

st.title("🎓 Placement Prediction Dashboard")
st.write("Analyze, Predict & Improve Placement Chances")

# Sidebar
st.sidebar.header("📥 Student Input")

cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)
aptitude = st.sidebar.slider("Aptitude Score", 0, 100, 60)
communication = st.sidebar.slider("Communication", 0, 10, 6)
technical = st.sidebar.slider("Technical Skills", 0, 10, 7)
internship = st.sidebar.selectbox("Internship", ["No", "Yes"])

internship_val = 1 if internship == "Yes" else 0

# Train models
results, model_outputs, best_model, scaler = train_models()

# Metrics
st.subheader("📊 Model Performance")

cols = st.columns(len(results))
for i, (model, acc) in enumerate(results.items()):
    cols[i].metric(model, f"{acc*100:.2f}%")

st.success(f"🏆 Best Model: {best_model}")

# Graph
st.subheader("📈 Model Comparison")
df_results = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
st.bar_chart(df_results.set_index("Model"))

# Confusion Matrix
st.subheader("🧩 Confusion Matrix")
cm = model_outputs[best_model]["cm"]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# ROC Curve
st.subheader("📉 ROC Curve")
roc_data = model_outputs[best_model]

if roc_data["fpr"] is not None:
    fig2, ax2 = plt.subplots()
    ax2.plot(roc_data["fpr"], roc_data["tpr"], label=f"AUC = {roc_data['auc']:.2f}")
    ax2.plot([0,1], [0,1], linestyle="--")
    ax2.legend()
    st.pyplot(fig2)

# Feature Importance
st.subheader("⭐ Feature Importance")
rf_model = model_outputs["Random Forest"]["model"]
importance = rf_model.feature_importances_

features = ["CGPA", "Aptitude", "Communication", "Technical", "Internship"]
df_imp = pd.DataFrame({"Feature": features, "Importance": importance})
st.bar_chart(df_imp.set_index("Feature"))

# Prediction
st.subheader("🤖 Prediction")

if st.button("Predict Placement"):
    input_data = np.array([[cgpa, aptitude, communication, technical, internship_val]])
    
    input_scaled = scaler.transform(input_data)

    model = model_outputs[best_model]["model"]

    pred = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]

    st.info(f"📊 Placement Probability: {prob*100:.2f}%")

    if pred[0] == 1:
        st.success("🎉 High chance of placement")
    else:
        st.error("⚠️ Low chance of placement")

# Suggestions
st.subheader("💡 Suggestions")

if cgpa < 7:
    st.warning("Improve CGPA")

if aptitude < 70:
    st.warning("Work on Aptitude")

if technical < 7:
    st.warning("Improve Technical Skills")

if internship_val == 0:
    st.warning("Try to get Internship Experience")