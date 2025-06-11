import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
from shap.plots import _waterfall

# ========== Model Definition ==========
class DNNModel1(nn.Module):
    def __init__(self, input_dim):
        super(DNNModel1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16), nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(16, 32), nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(32, 16), nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(16, 8), nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

class DNNModel2(nn.Module):
    def __init__(self, input_dim):
        super(DNNModel2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32), nn.Dropout(0.2), nn.ReLU(),
            nn.Linear(32, 64), nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(64, 128), nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(128, 128), nn.Dropout(0.2), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# ========== Scaler Preparation ==========
def prepare_scaler(df, skip_list, target_col='RF'):
    X = df.drop(columns=[target_col])
    X_skip = X.iloc[:, skip_list]
    X_scale = X.drop(X.columns[skip_list], axis=1)
    scaler = StandardScaler()
    scaler.fit(X_scale)
    return scaler, X.columns.tolist()

# ========== SHAP Visualization ==========
def explain_shap_waterfall(model, input_df, background_df, skip_list, scaler, st_placeholder):
    Xb_skip = background_df.iloc[:, skip_list]
    Xb_scale = background_df.drop(background_df.columns[skip_list], axis=1)
    Xb_scaled = scaler.transform(Xb_scale)
    background_final = np.concatenate([Xb_scaled, Xb_skip], axis=1)

    Xi_skip = input_df.iloc[:, skip_list]
    Xi_scale = input_df.drop(input_df.columns[skip_list], axis=1)
    Xi_scaled = scaler.transform(Xi_scale)
    input_final = np.concatenate([Xi_scaled, Xi_skip], axis=1)

    explainer = shap.KernelExplainer(
        model=lambda x: model(torch.tensor(x, dtype=torch.float32).to(device)).cpu().detach().numpy(),
        data=background_final
    )
    shap_values = explainer.shap_values(input_final)

    plt.clf()
    _waterfall.waterfall_legacy(
        explainer.expected_value[0],
        shap_values[0][0],
        feature_names=input_df.columns
    )
    fig = plt.gcf()
    st_placeholder.pyplot(fig)

# ========== Prediction ==========
def predict_patient(input_df, model, skip_list, scaler, threshold):
    input_skip = input_df.iloc[:, skip_list]
    input_scale = input_df.drop(input_df.columns[skip_list], axis=1)
    input_scaled = scaler.transform(input_scale)
    input_final = np.concatenate([input_scaled, input_skip], axis=1)
    input_tensor = torch.tensor(input_final, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        prob = model(input_tensor).item()
    risk = "High risk" if prob >= threshold else "Low risk"
    return prob, risk

# ========== Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Load Data ==========
df1 = pd.read_csv("traindata1.CSV")
skip1 = [2, 3, 4, 5, 6, 7]
scaler1, columns1 = prepare_scaler(df1, skip1)

df2 = pd.read_csv("traindata2.CSV")
skip2 = [6, 7, 8, 9, 10]
scaler2, columns2 = prepare_scaler(df2, skip2)

model1 = DNNModel1(input_dim=len(columns1)).to(device)
model1.load_state_dict(torch.load("dnn_model1.pth", map_location=device))

model2 = DNNModel2(input_dim=len(columns2)).to(device)
model2.load_state_dict(torch.load("dnn_model2.pth", map_location=device))

# ========== Streamlit UI ==========
st.title("Risk Prediction for Postoperative Respiratory Failure (PRF)")

model_type = st.sidebar.radio("Choose Model", ["Preoperative Model", "Pre + Intraoperative Model"])

if model_type == "Preoperative Model":
    a = [
        st.sidebar.number_input("Age", 18, 120),
        st.sidebar.number_input("Preoperative LVEF (%)", 1, 100),
        0 if st.sidebar.selectbox('Pre-op WBC', ['<10*10⁹', '≥10*10⁹']) == '<10*10⁹' else 1,
        0 if st.sidebar.selectbox('Pre-op Cr', ['≤110 μmol/L', '>110 μmol/L']) == '≤110 μmol/L' else 1,
        0 if st.sidebar.selectbox('ASA status', ['I/II', 'III/IV/V']) == 'I/II' else 1,
        {'<25': 0, '25~40': 1, '40~70': 2, '>70': 3}[st.sidebar.selectbox('PAP (mmHg)', ['<25', '25~40', '40~70', '>70'])],
        0 if st.sidebar.selectbox('Emergency', ['No', 'Yes']) == 'No' else 1,
        0 if st.sidebar.selectbox('COPD', ['No', 'Yes']) == 'No' else 1
    ]
    input_df = pd.DataFrame([a], columns=columns1)
    prob, risk = predict_patient(input_df, model1, skip1, scaler1, threshold=0.2158)
else:
    a = [
        st.sidebar.number_input("Age", 18, 120),
        st.sidebar.number_input("Pre-op LVEF (%)", 1, 100),
        st.sidebar.number_input("CPB duration (min)", 1),
        st.sidebar.number_input("Crystalloid infusion (ml/kg)", 0.0),
        st.sidebar.number_input("Colloid infusion (ml/kg)", 0.0),
        st.sidebar.number_input("Auto blood (ml/kg)", 0.0),
        0 if st.sidebar.selectbox('Pre-op WBC', ['<10*10⁹', '≥10*10⁹']) == '<10*10⁹' else 1,
        0 if st.sidebar.selectbox('ASA status', ['I/II', 'III/IV/V']) == 'I/II' else 1,
        {'<25': 0, '25~40': 1, '40~70': 2, '>70': 3}[st.sidebar.selectbox('PAP (mmHg)', ['<25', '25~40', '40~70', '>70'])],
        0 if st.sidebar.selectbox('Emergency', ['No', 'Yes']) == 'No' else 1,
        0 if st.sidebar.selectbox('COPD', ['No', 'Yes']) == 'No' else 1
    ]
    input_df = pd.DataFrame([a], columns=columns2)
    prob, risk = predict_patient(input_df, model2, skip2, scaler2, threshold=0.2633)

if st.button("Predict PRF Risk"):
    st.success(f"Predicted Probability: {prob*100:.1f}%")
    st.success(f"Risk Category: {risk}")

    st.write("\n**🔍 SHAP Feature Contribution Analysis**")
    with st.spinner("Calculating SHAP values..."):
        if model_type == "Preoperative Model":
            explain_shap_waterfall(model1, input_df, df1.drop(columns=['RF']), skip1, scaler1, st)
        else:
            explain_shap_waterfall(model2, input_df, df2.drop(columns=['RF']), skip2, scaler2, st)

st.markdown("---")
st.markdown("*This application does not store any of your input data.*")
