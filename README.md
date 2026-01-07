# 🧭 The Strategic Profit Compass
### *Bridging Transactional Data to C-Suite Strategy*

https://executive-analytics.streamlit.app/

## 🎯 Executive Summary
In most organizations, a "Profit & Loss" statement tells you *what* happened, but rarely *why*. The **Executive analytics** is a managerial decision-support tool designed for C-Suite executives to bridge the gap between raw sales ledgers and strategic positioning.

This tool isolates the drivers of profit—**Volume, Price, and Cost**—and provides an interactive environment to simulate future market entries and pricing strategies.

## 🛠️ The Financial Architecture
The project is built on three pillars of "Managerial Intelligence":

1. **The Profit Bridge (Waterfall):** A Year-over-Year (YoY) analysis that decomposes profit changes into Price Effect, Volume Mix, and Cost Inflation.
2. **Strategic Simulation Engine:** A "What-If" environment allowing users to adjust price elasticity and supply-side cost reductions for specific product lines (e.g., Aura-Grid, Flux-Chiller).
3. **Deep-Dive Analytics:** Monthly trend analysis with granular product filters to identify seasonality and margin leakage.

## 🧮 Logic & Methodology
- **Financial Precision:** Growth metrics are calculated in **Basis Points (bps)** to mirror executive-level reporting.
- **Sensitivity Modeling:** The 2026 Projection uses a Linear Sensitivity Model:
  $$Profit_{sim} = \sum [Units_{y2} \cdot (1 + \Delta Vol) \cdot (Price_{sim} - Cost_{sim})] - Fixed\,Overheads$$
- **Technology Stack:** Python, Pandas, Plotly, and Streamlit.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the data generator: `python data_generator.py`.
4. Launch the app: `streamlit run app.py`.
