import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Executive Analytics Dashboard", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    sales = pd.read_csv('sales_ledger.csv')
    overheads = pd.read_csv('overheads.csv')
    sales['Date'] = pd.to_datetime(sales['Date'])
    return sales, overheads

try:
    sales, overheads = load_data()
except FileNotFoundError:
    st.error("Data files not found. Please run the Data Generator script first!")
    st.stop()

# --- GLOBAL DATA PROCESSING ---
y1_data = sales[sales['Date'].dt.year == 2024]
y2_data = sales[sales['Date'].dt.year == 2025]
product_list = sorted(sales['Product'].unique().tolist())

# --- SIDEBAR: STRATEGIC CONTROLS ---
st.sidebar.header("🕹️ Strategy Simulator (2026)")
st.sidebar.markdown("Adjust levers to simulate Year 3 performance.")

target_product = st.sidebar.selectbox("Select Target Product", ["All Products"] + product_list)

price_change = st.sidebar.slider(f"Price Adjustment (%)", -20, 20, 5)
cost_reduction = st.sidebar.slider(f"Cost Reduction (%)", 0, 20, 10)
volume_growth = st.sidebar.slider("Global Target Volume Growth (%)", -10, 30, 5)

# --- NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Executive Summary", "Deep Dive Analytics"])

# --- HEADER SECTION ---
st.title("🧭 Executive Analytics Dashboard")
with st.expander("ℹ️ Project Objective & User Guide", expanded=True):
    st.write("""
    **Objective:** This dashboard bridges the gap between raw transaction data and C-Suite strategy. 
    It identifies 'Profit Leakage' and simulates how market shifts impact the bottom line.
     
    **How to use:** Toggle between the **Executive Summary** for a financial bridge view, and **Deep Dive Analytics** for product-level performance. Use the sidebar to simulate Year (2026) growth scenarios. Use the **Product Selector** in the sidebar to simulate targeted price and cost strategies.
    """)

# --- KPI CALCULATIONS ---
total_rev_y2 = (y2_data['Units_Sold'] * y2_data['Price']).sum()
total_cogs_y2 = (y2_data['Units_Sold'] * y2_data['Unit_Cost']).sum()
total_fixed_y2 = overheads['Monthly_Cost_2025'].sum() * 12
margin_y2 = (total_rev_y2 - total_cogs_y2) / total_rev_y2
net_profit_y2 = (total_rev_y2 - total_cogs_y2) - total_fixed_y2

total_rev_y1 = (y1_data['Units_Sold'] * y1_data['Price']).sum()
total_cogs_y1 = (y1_data['Units_Sold'] * y1_data['Unit_Cost']).sum()
margin_y1 = (total_rev_y1 - total_cogs_y1) / total_rev_y1
net_profit_y1 = (total_rev_y1 - total_cogs_y1) - (overheads['Monthly_Cost_2024'].sum() * 12)

rev_growth = ((total_rev_y2 - total_rev_y1) / total_rev_y1) * 100
margin_delta_bps = (margin_y2 - margin_y1) * 10000 
profit_growth = ((net_profit_y2 - net_profit_y1) / abs(net_profit_y1)) * 100
top_item = y2_data.groupby('Product')['Units_Sold'].sum().idxmax()

# --- PAGE 1: EXECUTIVE SUMMARY ---
if page == "Executive Summary":
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("2025 Total Revenue", f"${total_rev_y2:,.0f}", f"{rev_growth:.1f}% vs LY")
    m2.metric("Gross Margin %", f"{margin_y2:.1%}", f"{margin_delta_bps:,.0f} bps vs LY")
    m3.metric("Net Operating Profit", f"${net_profit_y2:,.0f}", f"{profit_growth:.1f}% vs LY")
    
    with m4:
        st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #f0f2f6; height: 100px; display: flex; flex-direction: column; justify-content: center;">
                <p style="margin: 0; font-size: 14px; color: #6c757d;">Top Performer</p>
                <p style="margin: 0; font-size: 18px; font-weight: 600; color: #31333F; line-height: 1.2; word-wrap: break-word;">{top_item}</p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 2. Profit Bridge
    st.subheader("📊 The Annual Profit Bridge (2024 to 2025)")
    vol_imp = (y2_data['Units_Sold'].sum() - y1_data['Units_Sold'].sum()) * (y1_data['Price'].mean() - y1_data['Unit_Cost'].mean())
    pri_imp = y2_data['Units_Sold'].sum() * (y2_data['Price'].mean() - y1_data['Price'].mean())
    cst_imp = y2_data['Units_Sold'].sum() * (y1_data['Unit_Cost'].mean() - y2_data['Unit_Cost'].mean())
    
    fig_waterfall = go.Figure(go.Waterfall(
        orientation = "v", measure = ["relative", "relative", "relative", "relative", "total"],
        x = ["2024 Base Profit", "Volume Mix", "Price Strategy", "Cost Inflation", "2025 Actual Profit"],
        y = [net_profit_y1, vol_imp, pri_imp, cst_imp, net_profit_y2],
        text = [f"${net_profit_y1:,.0f}", f"{vol_imp:,.0f}", f"{pri_imp:,.0f}", f"{cst_imp:,.0f}", f"${net_profit_y2:,.0f}"],
        increasing = {"marker":{"color":"#2ECC71"}}, decreasing = {"marker":{"color":"#E74C3C"}}, totals = {"marker":{"color":"#3498DB"}}
    ))
    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.divider()

    # 3. Projection Section
    st.subheader("🚀 2026 Strategic Projection")
    with st.expander("📝 View Projection Methodology", expanded=False):
        st.latex(r"Revenue_{sim} = Units_{y2} \times (1 + \Delta Vol) \times (Price_{y2} \times (1 + \Delta Price))")
        st.write("""
        The model uses a **Linear Sensitivity Analysis**:
        - **Revenue:** Projected volume multiplied by simulated price.
        - **COGS:** Simulated units multiplied by adjusted unit costs (reflecting supplier negotiations).
        - **Fixed Costs:** Assumed constant at 2025 levels to isolate operational impact.
        """)

    # PRODUCT-SPECIFIC SIMULATION LOGIC
    sim_data = y2_data.copy()
    if target_product == "All Products":
        sim_data['Price'] = sim_data['Price'] * (1 + price_change/100)
        sim_data['Unit_Cost'] = sim_data['Unit_Cost'] * (1 - cost_reduction/100)
    else:
        sim_data.loc[sim_data['Product'] == target_product, 'Price'] *= (1 + price_change/100)
        sim_data.loc[sim_data['Product'] == target_product, 'Unit_Cost'] *= (1 - cost_reduction/100)
    
    sim_data['Units_Sold'] *= (1 + volume_growth/100)
    sim_rev = (sim_data['Units_Sold'] * sim_data['Price']).sum()
    sim_cogs = (sim_data['Units_Sold'] * sim_data['Unit_Cost']).sum()
    sim_profit = (sim_rev - sim_cogs) - total_fixed_y2

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write(f"### Strategy Impact")
        st.write(f"Projected 2026 Operating Profit: **${sim_profit:,.0f}**")
        delta_sim = sim_profit - net_profit_y2
        st.metric("Projected Delta vs 2025", f"${delta_sim:,.0f}", delta_color="normal")
        
        # DYNAMIC INSIGHTS ENGINE
        st.markdown("---")
        st.markdown("#### 💡 Strategic Insights")
        
        # Insight 1: Price Impact
        price_impact_val = (sim_data['Price'].mean() - y2_data['Price'].mean()) * sim_data['Units_Sold'].sum()
        if price_change != 0:
            direction = "increase" if price_change > 0 else "reduction"
            st.info(f"**Price Strategy:** A {abs(price_change)}% {direction} in price affects the top-line by **${abs(price_impact_val):,.0f}**. Watch for churn in the '{target_product}' segment.")
        
        # Insight 2: Cost Reduction Impact
        cost_impact_val = (y2_data['Unit_Cost'].mean() - sim_data['Unit_Cost'].mean()) * sim_data['Units_Sold'].sum()
        if cost_reduction > 0:
            st.success(f"**Operational Efficiency:** Reducing supply costs by {cost_reduction}% recovers **${cost_impact_val:,.0f}** in margin leakage.")

        # Insight 3: Volume vs Margin trade-off
        if sim_profit < net_profit_y2:
            st.error(f"**Margin Alert:** Current price/cost levers are eroding the bottom line. Re-evaluate '{target_product}' positioning.")

    with c2:
        fig_sim = go.Figure(data=[
            go.Bar(name='2025 Actual', x=['Profit'], y=[net_profit_y2], marker_color='#3498DB'),
            go.Bar(name='2026 Projected', x=['Profit'], y=[sim_profit], marker_color='#2ECC71')
        ])
        fig_sim.update_layout(barmode='group', height=400, showlegend=True)
        st.plotly_chart(fig_sim, use_container_width=True)

# --- PAGE 2: DEEP DIVE ANALYTICS ---
else:
    st.header("📊 Deep Dive Analytics")
    
    # 1. Dual Filters for Year and Product
    col_a, col_b = st.columns(2)
    with col_a:
        selected_year = st.selectbox("Select Year to Analyze", [2024, 2025])
    with col_b:
        # Added Product Filter for the Chart
        selected_prod = st.selectbox("Select Product for Trend Analysis", ["All Products"] + product_list)

    # 2. Filter Logic
    year_data = sales[sales['Date'].dt.year == selected_year].copy()
    
    if selected_prod != "All Products":
        # Filter the data for the specific product before grouping
        chart_data = year_data[year_data['Product'] == selected_prod].copy()
    else:
        chart_data = year_data.copy()

    chart_data['Month'] = chart_data['Date'].dt.strftime('%m - %b')
    
    # 3. Monthly Aggregation
    monthly = chart_data.groupby('Month').agg({
        'Units_Sold': 'sum', 
        'Price': 'mean', 
        'Unit_Cost': 'mean'
    }).reset_index()
    
    monthly['Revenue'] = monthly['Units_Sold'] * monthly['Price']
    monthly['Profit'] = monthly['Revenue'] - (monthly['Units_Sold'] * monthly['Unit_Cost'])

    # 4. Monthly Trend Chart
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(x=monthly['Month'], y=monthly['Revenue'], name="Revenue", marker_color='#3498DB'))
    fig_trend.add_trace(go.Scatter(x=monthly['Month'], y=monthly['Profit'], name="Profit", line=dict(color='#E74C3C', width=3)))
    
    chart_title = f"Monthly Performance: {selected_prod} ({selected_year})"
    fig_trend.update_layout(title=chart_title, xaxis_title="Month", yaxis_title="USD ($)")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("📦 Product Performance Analysis")
    p_2024 = y1_data.groupby('Product')['Units_Sold'].sum()
    p_2025 = y2_data.groupby('Product')['Units_Sold'].sum()
    prod_table = pd.DataFrame({'2024 Units': p_2024, '2025 Units': p_2025})
    prod_table['Growth (%)'] = ((prod_table['2025 Units'] - prod_table['2024 Units']) / prod_table['2024 Units']) * 100
    st.dataframe(prod_table.style.format("{:.1f}").background_gradient(subset=['Growth (%)'], cmap='RdYlGn'), use_container_width=True)

    with st.expander("📂 Audit Raw Data"):
        st.dataframe(sales.sort_values(by='Date', ascending=False), height=400)

st.divider()
st.caption(f"Strategy Directional Compass | Compiled on {datetime.now().strftime('%Y-%m-%d')} | Executive Decision Support Tool")