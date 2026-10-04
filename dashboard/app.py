"""Interactive Decision Support System & Web Dashboard.

AI-Driven Intelligent Campaign Optimization for Digital Marketing.
Built with Streamlit & Plotly.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup system path to import parent modules
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.data_loader import get_preprocessed_data
from data.channel_touchpoint_generator import get_channel_touchpoint_data, CHANNEL_CONFIG
from models.conversion_model import train_and_save_best_model, SAVED_MODEL_PATH
from models.clv_revenue_model import train_and_save_clv_model, SAVED_CLV_MODEL_PATH
from models.model_explainability import get_global_feature_importance, compute_customer_prediction_factors
from optimization.problem_formulation import CampaignConstraints
from optimization.optimization_benchmark import run_full_optimization_benchmark
from analytics.marketing_metrics import compute_channel_performance_breakdown
from analytics.attribution_models import compute_multitouch_attribution


# Page Configuration
st.set_page_config(
    page_title="AI Campaign Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Executive UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-title { font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.6rem; color: #0f172a; font-weight: 700; }
    .metric-sub { font-size: 0.8rem; color: #10b981; font-weight: 500; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 48px; white-space: pre-wrap; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_campaign_data():
    """Loads and caches customer data, touchpoints, and analytics."""
    cust_df = get_preprocessed_data()
    tp_df = get_channel_touchpoint_data(cust_df)
    channel_perf = compute_channel_performance_breakdown(tp_df)
    attr_df = compute_multitouch_attribution(tp_df)
    return cust_df, tp_df, channel_perf, attr_df


@st.cache_data
def load_or_train_models(force_retrain: bool = False):
    cust_df = get_preprocessed_data()
    if force_retrain or not SAVED_MODEL_PATH.exists():
        conv_leaderboard, _ = train_and_save_best_model(cust_df)
    else:
        import pickle
        with open(SAVED_MODEL_PATH, "rb") as f:
            art = pickle.load(f)
            conv_leaderboard = pd.DataFrame(art["results_table"])

    if force_retrain or not SAVED_CLV_MODEL_PATH.exists():
        clv_leaderboard, _ = train_and_save_clv_model(cust_df)
    else:
        import pickle
        with open(SAVED_CLV_MODEL_PATH, "rb") as f:
            art = pickle.load(f)
            clv_leaderboard = pd.DataFrame(art["results_table"])

    return conv_leaderboard, clv_leaderboard


@st.cache_data
def run_cached_benchmark(budget: float):
    cust_df = get_preprocessed_data()
    tp_df = get_channel_touchpoint_data(cust_df)
    return run_full_optimization_benchmark(tp_df, total_budget=budget)


# App Header
st.title("🎯 AI-Driven Intelligent Campaign Optimization Platform")
st.markdown(
    "**Multi-Objective Decision Support System** integrating Machine Learning predictions with Pareto Evolutionary Algorithms (NSGA-II) & Operations Research."
)

# Load Data
with st.spinner("Initializing AI Engines & Loading Marketing Data..."):
    cust_df, tp_df, channel_perf, attr_df = load_all_campaign_data()
    conv_leaderboard, clv_leaderboard = load_or_train_models()

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/bullish.png", width=64)
st.sidebar.header("Campaign Controls")
campaign_budget = st.sidebar.slider("Total Campaign Budget ($)", min_value=500.0, max_value=10000.0, value=2500.0, step=250.0)
st.sidebar.markdown(f"**Customer Base:** {len(cust_df):,} profiles")
st.sidebar.markdown(f"**Touchpoints Analyzed:** {len(tp_df):,}")

if st.sidebar.button("🔄 Retrain AI Models"):
    with st.spinner("Retraining Predictive Models..."):
        conv_leaderboard, clv_leaderboard = load_or_train_models(force_retrain=True)
        st.cache_data.clear()
        st.sidebar.success("Models retrained successfully!")

st.sidebar.markdown("---")
st.sidebar.info("""
**System Engine Capabilities:**
- 🎯 Audience RFM & Behavioral Targeting
- 📊 Multi-Touch Attribution & ROAS Analysis
- ⚡ Pareto Multi-Objective Optimization (NSGA-II)
- 🧠 Predictive GBDT Models & SHAP Explainability
""")

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive KPIs",
    "👥 Customer RFM Segments",
    "🧠 AI Predictive Studio",
    "⚡ Multi-Objective Pareto Studio",
    "🎛️ What-If Simulator",
    "🚀 Campaign Action Plan"
])

# ==============================================================================
# TAB 1: EXECUTIVE KPIS & PERFORMANCE ANALYTICS
# ==============================================================================
with tab1:
    st.subheader("Campaign Performance & Multi-Channel Economics")

    # High-level Metrics Row
    tot_spend = channel_perf["Spend ($)"].sum()
    tot_rev = channel_perf["Revenue ($)"].sum()
    tot_conv = channel_perf["Conversions"].sum()
    tot_contacts = channel_perf["Total Contacts"].sum()
    overall_roas = tot_rev / max(tot_spend, 1e-4)
    overall_cpa = tot_spend / max(tot_conv, 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Spend", f"${tot_spend:,.2f}")
    c2.metric("Total Revenue", f"${tot_rev:,.2f}", f"+{(tot_rev - tot_spend):,.2f} Net")
    c3.metric("Total Conversions", f"{tot_conv:,}", f"{(tot_conv/tot_contacts*100):.2f}% CVR")
    c4.metric("Blended ROAS", f"{overall_roas:.2f}x", "Target: >1.5x")
    c5.metric("Avg CPA", f"${overall_cpa:.2f}")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("#### Channel Revenue vs. Spend Breakdown")
        fig_bar = go.Figure(data=[
            go.Bar(name="Spend ($)", x=channel_perf["Channel"], y=channel_perf["Spend ($)"], marker_color="#ef4444"),
            go.Bar(name="Revenue ($)", x=channel_perf["Channel"], y=channel_perf["Revenue ($)"], marker_color="#10b981")
        ])
        fig_bar.update_layout(barmode="group", height=360, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("#### Multi-Touch Attribution Credit Share")
        fig_attr = px.bar(
            attr_df,
            x="Channel",
            y=["First_Touch_%", "Last_Touch_%", "Shapley_Algorithmic_%"],
            barmode="group",
            height=360,
            labels={"value": "Attribution Share (%)", "variable": "Attribution Model"}
        )
        fig_attr.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_attr, use_container_width=True)

    st.markdown("#### Channel Efficiency & Economics Table")
    st.dataframe(channel_perf.style.format({
        "Spend ($)": "${:,.2f}",
        "Revenue ($)": "${:,.2f}",
        "CTR (%)": "{:.2f}%",
        "CPC ($)": "${:.2f}",
        "CPA ($)": "${:.2f}",
        "ROAS": "{:.2f}x",
        "ROI (%)": "{:.1f}%",
        "Avg Fatigue Risk": "{:.3f}"
    }), use_container_width=True)


# ==============================================================================
# TAB 2: CUSTOMER RFM & BEHAVIORAL SEGMENTS
# ==============================================================================
with tab2:
    st.subheader("Customer Behavioral Profiling & Segmentation")

    col_rfm1, col_rfm2 = st.columns([3, 2])
    with col_rfm1:
        st.markdown("#### 3D RFM Behavioral Space (Recency vs. Spend vs. Frequency)")
        fig_rfm = px.scatter_3d(
            cust_df.sample(min(800, len(cust_df)), random_state=42),
            x="Recency",
            y="TotalSpend",
            z="TotalPurchases",
            color="Response",
            color_discrete_map={0: "#94a3b8", 1: "#2563eb"},
            labels={"Response": "Campaign Response", "TotalSpend": "Monetary ($)", "TotalPurchases": "Frequency"},
            opacity=0.8,
            height=450
        )
        fig_rfm.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_rfm, use_container_width=True)

    with col_rfm2:
        st.markdown("#### Customer Persona Distribution")
        cust_df["Persona"] = pd.cut(
            cust_df["Estimated_CLV"],
            bins=[-np.inf, 200, 600, np.inf],
            labels=["Low Value / Casual", "Core Steady Shoppers", "VIP / High CLV Champions"]
        )
        fig_pie = px.pie(cust_df, names="Persona", hole=0.45, color_discrete_sequence=px.colors.sequential.RdBu)
        fig_pie.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### Product Spend Mix across Categories")
    spend_summary = cust_df[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].mean().reset_index()
    spend_summary.columns = ["Category", "Average Spend ($)"]
    spend_summary["Category"] = spend_summary["Category"].str.replace("Mnt", "")
    fig_spend = px.bar(spend_summary, x="Category", y="Average Spend ($)", color="Category", height=300)
    st.plotly_chart(fig_spend, use_container_width=True)


# ==============================================================================
# TAB 3: AI PREDICTIVE STUDIO
# ==============================================================================
with tab3:
    st.subheader("AI Predictive Modeling Leaderboard & Explainability")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### Binary Conversion Prediction Leaderboard")
        st.dataframe(conv_leaderboard.style.highlight_max(axis=0, subset=["ROC-AUC", "F1-Score", "Accuracy"]), use_container_width=True)

    with col_m2:
        st.markdown("#### CLV & Monetary Revenue Regression Leaderboard")
        st.dataframe(clv_leaderboard.style.highlight_min(axis=0, subset=["RMSE ($)", "MAE ($)"]), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Global Feature Importance (Top Drivers of Response)")
    df_imp = get_global_feature_importance()
    fig_imp = px.bar(
        df_imp.head(10),
        x="Importance_Pct",
        y="Feature",
        orientation="h",
        color="Importance_Pct",
        color_continuous_scale="Viridis",
        labels={"Importance_Pct": "Relative Predictive Importance (%)"},
        height=350
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=10, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Individual Customer Explainability Probe")
    st.markdown("Select an individual customer profile to see how their demographic and behavioral traits drive predicted conversion:")
    sample_ids = cust_df["ID"].head(20).tolist()
    selected_id = st.selectbox("Select Customer ID:", sample_ids)
    cust_row = cust_df[cust_df["ID"] == selected_id].iloc[0]

    factors_df = compute_customer_prediction_factors(cust_row)
    c_p1, c_p2 = st.columns([1, 2])
    with c_p1:
        st.markdown(f"""
        **Customer #{selected_id} Profile:**
        - **Age:** {int(cust_row['Age'])} yrs
        - **Income:** ${cust_row['Income']:,.0f}
        - **Recency:** {int(cust_row['Recency'])} days
        - **Total Spend:** ${cust_row['TotalSpend']:,.2f}
        - **Past Campaigns:** {int(cust_row['PastCampaignsAccepted'])} accepted
        - **Predicted CLV:** ${cust_row['Estimated_CLV']:,.2f}
        """)
    with c_p2:
        fig_waterfall = px.bar(
            factors_df.head(6),
            x="Impact_on_Probability",
            y="Feature",
            color="Direction",
            orientation="h",
            color_discrete_map={"Positive (+)": "#10b981", "Negative (-)": "#ef4444"},
            height=280,
            title="Local Feature Contribution (Shapley Proxy)"
        )
        fig_waterfall.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_waterfall, use_container_width=True)


# ==============================================================================
# TAB 4: MULTI-OBJECTIVE PARETO STUDIO
# ==============================================================================
with tab4:
    st.subheader("Multi-Objective Optimization Studio: Pareto Frontier & Strategy Archetypes")
    st.markdown(
        "Simultaneously optimizing **Revenue**, **Conversions**, **Campaign Cost**, and **Ad Fatigue**."
    )

    comp_df, extra_opt = run_cached_benchmark(campaign_budget)
    pareto_df = extra_opt["pareto_front"]

    st.markdown("#### Head-to-Head Optimization Benchmark")
    st.dataframe(comp_df.style.format({
        "Campaign_Cost": "${:,.2f}",
        "Expected_Revenue": "${:,.2f}",
        "Expected_Conversions": "{:,.1f}",
        "ROI_Pct": "{:,.1f}%",
        "CPA": "${:,.2f}",
        "Customer_Fatigue": "{:.2f}"
    }).highlight_max(axis=0, subset=["ROI_Pct", "Expected_Revenue"]), use_container_width=True)

    st.markdown("---")
    col_p1, col_p2 = st.columns([3, 2])
    with col_p1:
        st.markdown("#### Pareto Optimal Frontier (Campaign Cost vs. Expected Revenue)")
        fig_pareto = px.scatter(
            pareto_df,
            x="Campaign_Cost",
            y="Expected_Revenue",
            color="Strategy_Archetype",
            size="ROI_Pct",
            hover_data=["Targeted_Customers", "Customer_Fatigue", "ROI_Pct"],
            labels={"Campaign_Cost": "Total Campaign Cost ($)", "Expected_Revenue": "Expected Revenue ($)"},
            height=450
        )
        fig_pareto.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pareto, use_container_width=True)

    with col_p2:
        st.markdown("#### Pareto Strategy Archetype Insights")
        st.info("""
        **1. High-ROI (Cost-Efficient) Cluster:**
        - Targets high-intent, high-CLV segments on low-cost channels (Email, high-affinity SMS).
        - Minimizes waste and avoids ad fatigue wear-out.

        **2. Balanced Growth Cluster:**
        - Optimal trade-off between volume and margin.
        - Blends Paid Search with Social Media and Email.

        **3. Aggressive Reach Cluster:**
        - Maximizes market share and conversion volume.
        - Trades off higher marginal CPA and increased ad fatigue.
        """)

    st.markdown("#### 3D Multi-Objective Trade-Off Space (Cost vs. Revenue vs. Customer Fatigue)")
    fig_pareto_3d = px.scatter_3d(
        pareto_df,
        x="Campaign_Cost",
        y="Customer_Fatigue",
        z="Expected_Revenue",
        color="Strategy_Archetype",
        labels={"Customer_Fatigue": "Ad Fatigue / Churn Risk", "Campaign_Cost": "Cost ($)", "Expected_Revenue": "Revenue ($)"},
        height=500
    )
    fig_pareto_3d.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_pareto_3d, use_container_width=True)


# ==============================================================================
# TAB 5: WHAT-IF SCENARIO SIMULATOR
# ==============================================================================
with tab5:
    st.subheader("🎛️ Interactive What-If Campaign Budget Simulator")
    st.markdown("Interactively adjust channel budget allocations and observe real-time projected returns, conversions, and fatigue wear-out.")

    col_sim_ctrl, col_sim_view = st.columns([1, 2])
    with col_sim_ctrl:
        st.markdown("#### Allocate Budget Across Channels")
        b_email = st.slider("Email Budget ($)", 0.0, 1000.0, 200.0, 25.0)
        b_search = st.slider("Paid Search Budget ($)", 0.0, 3000.0, 1000.0, 50.0)
        b_social = st.slider("Social Media Budget ($)", 0.0, 2000.0, 600.0, 50.0)
        b_display = st.slider("Display Ads Budget ($)", 0.0, 1000.0, 300.0, 25.0)
        b_sms = st.slider("Direct SMS Budget ($)", 0.0, 1000.0, 400.0, 25.0)

        sim_total_spend = b_email + b_search + b_social + b_display + b_sms
        st.markdown(f"**Total Allocated Spend:** `${sim_total_spend:,.2f}`")

    with col_sim_view:
        # Compute simulation dynamics
        channels = ["Email", "Paid_Search", "Social_Media", "Display_Ads", "SMS"]
        budgets = [b_email, b_search, b_social, b_display, b_sms]
        sim_records = []
        tot_sim_rev = 0.0
        tot_sim_conv = 0.0
        tot_sim_fatigue = 0.0

        for ch, b in zip(channels, budgets):
            cfg = CHANNEL_CONFIG[ch]
            n_contacts = int(b / cfg["cost_per_contact"])
            cvr = cfg["base_cvr"]
            conversions = n_contacts * cvr
            rev = conversions * 85.0  # Average order value baseline
            fatigue = (n_contacts * cfg["complaint_risk_base"]) * (1.5 if ch == "SMS" else 1.0)

            tot_sim_rev += rev
            tot_sim_conv += conversions
            tot_sim_fatigue += fatigue

            sim_records.append({
                "Channel": ch,
                "Allocated Budget ($)": b,
                "Estimated Contacts": n_contacts,
                "Projected Conversions": round(conversions, 1),
                "Projected Revenue ($)": round(rev, 2)
            })

        sim_roi = ((tot_sim_rev - sim_total_spend) / max(sim_total_spend, 1e-4)) * 100.0
        sim_cpa = sim_total_spend / max(tot_sim_conv, 1e-4)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Projected Revenue", f"${tot_sim_rev:,.2f}")
        m2.metric("Projected ROI", f"{sim_roi:.1f}%")
        m3.metric("Projected Conversions", f"{tot_sim_conv:,.1f}")
        m4.metric("Projected CPA", f"${sim_cpa:.2f}")

        if tot_sim_fatigue > 45.0:
            st.warning(f"⚠️ **High Ad Fatigue Warning:** Contact intensity on high-intrusion channels (SMS/Display) is elevated ({tot_sim_fatigue:.1f} fatigue score). Recommend reallocating to Email or Search.")
        else:
            st.success(f"✅ **Safe Fatigue Range:** Customer satisfaction penalty is well-controlled ({tot_sim_fatigue:.1f} score).")

        df_sim = pd.DataFrame(sim_records)
        fig_sim = px.pie(df_sim, names="Channel", values="Allocated Budget ($)", hole=0.4, title="Simulated Budget Distribution")
        fig_sim.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_sim, use_container_width=True)


# ==============================================================================
# TAB 6: CAMPAIGN ACTION PLAN & RECOMMENDATIONS
# ==============================================================================
with tab6:
    st.subheader("🎯 Actionable Campaign Decision Recommendations")
    st.markdown("Exportable individual customer targeting directives generated by the Multi-Objective Optimization Engine.")

    top_candidates = tp_df.sort_values(by="Effective_CVR", ascending=False).head(500).copy()
    top_candidates["Recommended_Offer"] = np.where(
        top_candidates["Customer_TotalSpend"] > 800,
        "Premium VIP Bundle (20% Off Wine/Meat)",
        np.where(top_candidates["Customer_Age"] < 40, "Digital Mobile First Deal", "Loyalty Rewards Credit")
    )
    top_candidates["Priority_Rank"] = range(1, len(top_candidates) + 1)

    export_df = top_candidates[[
        "Priority_Rank", "Customer_ID", "Channel", "Contact_Frequency",
        "Effective_CVR", "Channel_Cost", "Customer_CLV", "Recommended_Offer"
    ]].rename(columns={
        "Contact_Frequency": "Optimal_Frequency",
        "Effective_CVR": "Predicted_Conv_Prob",
        "Channel_Cost": "Allocated_Cost_$"
    })

    st.dataframe(export_df.head(50), use_container_width=True)

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Full Campaign Targeting Recommendations (CSV)",
        data=csv_data,
        file_name="optimized_campaign_targeting_plan.csv",
        mime="text/csv"
    )
