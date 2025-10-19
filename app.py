import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from train import preprocess, Model, compute_accuracy, device

st.title("Titanic Survival Classifier – Model Evaluation & Inference")

MODEL_PATH = "model_weights.pth"
model = Model()
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

st.sidebar.header("Model & Data Options")
uploaded_file = st.sidebar.file_uploader("Upload your test CSV file", type=["csv"])

if uploaded_file is not None:
    st.success("Test data loaded successfully!")
    test_df = pd.read_csv(uploaded_file)

    if 'Survived' not in test_df.columns:
        st.error("The test CSV must include a 'Survived' column for evaluation.")
    else:
        y_true = test_df['Survived'].values
        X_test = preprocess(test_df.drop(columns=['Survived']))
        X_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy()
        y_pred = (preds >= 0.5).astype(int).flatten()

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        st.subheader("Evaluation Results")
        st.write(f"**Accuracy:** {acc:.3f}")
        st.json(report)

        cm = confusion_matrix(y_true, y_pred, normalize='all')
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)

        st.subheader("Prediction Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(preds, bins=20, color="skyblue", edgecolor="black")
        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
else:
    st.info("Please upload a CSV file to evaluate the model.")
