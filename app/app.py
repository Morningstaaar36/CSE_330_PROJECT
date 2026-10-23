"""Streamlit demo: predict drug-protein binding affinity (pKd) with the trained MDGTDTInet checkpoints.

Run from the repo root:  streamlit run app/app.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy import stats
from torch_geometric.data import Batch, Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dta import config as C
from dta.models.mdgtdti import MDGTDTInet

DRUGS_DIR, PROTEINS_DIR, ESM_DIR = C.DRUGS_DIR, C.PROTEINS_DIR, C.ESM_DIR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCENARIOS_CONFIG = {
    s: {'checkpoint': os.path.join(C.CKPT_DIR, f'best_{s}_fold4.pth'),
        'predictions_csv': os.path.join(C.RESULTS_DIR, s, 'predictions.csv'),
        'description': d}
    for s, d in [('cold_protein', 'Unseen target proteins'),
                 ('cold_drug', 'Unseen compounds'),
                 ('cold_both', 'Unseen compounds & proteins')]
}


@st.cache_resource
def load_model(checkpoint_path):
    model = MDGTDTInet(C.HIDDEN_DIM, C.HIDDEN_DIM, C.NUM_LAYERS, C.NUM_HEADS, C.ESM_DIM).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.eval()


@st.cache_data(show_spinner=False)
def load_predictions_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def get_available_drugs():
    return sorted(f[:-3] for f in os.listdir(DRUGS_DIR) if f.endswith('.pt')) if os.path.isdir(DRUGS_DIR) else []


@st.cache_data
def get_available_proteins():
    return sorted(f[:-3] for f in os.listdir(PROTEINS_DIR) if f.endswith('.pt')) if os.path.isdir(PROTEINS_DIR) else []


def predict_binding_affinity(model, drug_id, protein_id):
    """Run inference on a single drug-protein pair."""
    try:
        paths = {'Drug': os.path.join(DRUGS_DIR, f'{drug_id}.pt'),
                 'Protein': os.path.join(PROTEINS_DIR, f'{protein_id}.pt'),
                 'ESM embedding': os.path.join(ESM_DIR, f'{protein_id}.npy')}
        for name, path in paths.items():
            if not os.path.exists(path):
                return None, f"❌ {name} file not found: {path}"

        drug_graph = torch.load(paths['Drug'], weights_only=False)
        p = torch.load(paths['Protein'], weights_only=False)
        protein_graph = Data(x=p['x'], edge_index=p['edge_index'], edge_attr=p['edge_attr'],
                             laplacian_pe=p['laplacian_pe'])
        esm = torch.FloatTensor(np.load(paths['ESM embedding'], allow_pickle=True)).unsqueeze(0)

        with torch.no_grad():
            pred = model(Batch.from_data_list([drug_graph]).to(device),
                         Batch.from_data_list([protein_graph]).to(device), esm.to(device))
        return float(pred.item()), "✅ Success"
    except Exception as e:
        return None, f"❌ Error: {e}"


# ──────────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="MDGTDTInet · DTA", layout="wide")

st.title("🧬 Drug-Protein Binding Affinity Predictor")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
scenario = st.sidebar.radio(
    "Select Scenario:",
    list(SCENARIOS_CONFIG.keys()),
    format_func=lambda x: f"{x.upper()} - {SCENARIOS_CONFIG[x]['description']}"
)

config = SCENARIOS_CONFIG[scenario]
model = load_model(config['checkpoint'])
df_predictions = load_predictions_csv(config['predictions_csv'])

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Select Drug-Protein Pair")
    
    # Load available compounds and proteins
    available_drugs = get_available_drugs()
    available_proteins = get_available_proteins()
    
    # Dropdowns
    drug_id = st.selectbox(
        "🧪 Compound ID:",
        options=available_drugs,
        index=available_drugs.index("126565") if "126565" in available_drugs else 0,
        help="Select a compound from the database"
    )
    
    protein_id = st.selectbox(
        "🧬 Protein ID:",
        options=available_proteins,
        index=available_proteins.index("P12345") if "P12345" in available_proteins else 0,
        help="Select a protein target from the database"
    )
    
    col_pred, col_clear = st.columns(2)
    with col_pred:
        predict_btn = st.button("🔮 Predict Affinity", use_container_width=True)
    
    with col_clear:
        if st.button("🔄 Clear Selection", use_container_width=True):
            st.rerun()

with col2:
    st.subheader("📊 Scenario Information")
    st.info(f"""
    **Scenario:** {scenario.replace('_', ' ').title()}
    
    {config['description']}
    """)

# Prediction results
if predict_btn:
    with st.spinner("🔄 Running inference..."):
        prediction, message = predict_binding_affinity(model, drug_id, protein_id)
    
    if prediction is not None:
        st.success(message)
        
        # Display prediction prominently
        col_result_a, col_result_b, col_result_c = st.columns(3)
        
        with col_result_a:
            st.metric(
                label="Predicted Binding Affinity",
                value=f"{prediction:.4f}",
                help="pKd = -log10(Kd)"
            )
        
        with col_result_b:
            # Affinity strength interpretation
            if prediction > 7:
                strength = "🟢 Very Strong"
                color = "green"
            elif prediction > 5:
                strength = "🟡 Strong"
                color = "yellow"
            elif prediction > 3:
                strength = "🟠 Moderate"
                color = "orange"
            else:
                strength = "🔴 Weak"
                color = "red"
            
            st.metric(
                label="Binding Strength",
                value=strength,
                help="Relative binding strength classification"
            )
        
        with col_result_c:
            st.metric(
                label="Scenario",
                value=scenario.replace('_', ' ').title(),
            )
    else:
        st.error(message)

# Test Set Examples
st.divider()
st.subheader("📋 Test Set Examples")

if df_predictions is not None:
    # Show statistics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        mse = ((df_predictions['actual'] - df_predictions['predicted']) ** 2).mean()
        st.metric("MSE", f"{mse:.4f}")
    
    with metrics_col2:
        rmse = np.sqrt(mse)
        st.metric("RMSE", f"{rmse:.4f}")
    
    with metrics_col3:
        pearson_r = np.corrcoef(df_predictions['actual'], df_predictions['predicted'])[0, 1]
        st.metric("Pearson r", f"{pearson_r:.4f}")
    
    with metrics_col4:
        spearman_r = stats.spearmanr(df_predictions['actual'], df_predictions['predicted'])[0]
        st.metric("Spearman ρ", f"{spearman_r:.4f}")
    
    # Sample predictions
    st.write("**Sample Test Predictions:**")
    sample_df = df_predictions.sample(n=min(10, len(df_predictions)))[
        ['COMPOUND_ID', 'PROTEIN_ID', 'actual', 'predicted', 'error']
    ].copy()
    sample_df['actual'] = sample_df['actual'].apply(lambda x: f"{x:.4f}")
    sample_df['predicted'] = sample_df['predicted'].apply(lambda x: f"{x:.4f}")
    sample_df['error'] = sample_df['error'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(sample_df, use_container_width=True)
    
    # Visualization
    st.write("**Prediction Accuracy Plot:**")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_predictions['actual'], df_predictions['predicted'],
              alpha=0.5, s=20, color='#5C6BC0')
    
    lo = min(df_predictions['actual'].min(), df_predictions['predicted'].min()) - 0.5
    hi = max(df_predictions['actual'].max(), df_predictions['predicted'].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], '--', color='red', lw=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual pKd', fontsize=11)
    ax.set_ylabel('Predicted pKd', fontsize=11)
    ax.set_title(f'Predicted vs Actual - {scenario.upper()}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
else:
    st.warning("⚠️ Predictions CSV not found")


