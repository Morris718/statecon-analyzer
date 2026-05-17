# my_analysis.py
import pandas as pd
from statecon_analyzer import (
    DescriptiveAnalyzer,
    EconometricsEngine,
    EconomicIndicators,
    HypothesisTestingSuite,
    CostBenefitAnalysis
)

# ============================================================
# LOAD YOUR OWN DATA
# ============================================================

# Option A: Load from CSV
# data = pd.read_csv("your_data.csv")

# Option B: Load from Excel
# data = pd.read_excel("your_data.xlsx")

# Option C: Create data manually
data = pd.DataFrame({
    'year': [2018, 2019, 2020, 2021, 2022, 2023],
    'gdp': [95.5, 98.2, 92.1, 99.8, 105.2, 108.7],
    'inflation': [2.1, 2.3, 1.8, 3.2, 5.1, 4.2],
    'unemployment': [4.5, 4.3, 6.8, 5.9, 4.8, 4.2]
})

print("Your Data:")
print(data)

# ============================================================
# 1. DESCRIPTIVE ANALYSIS
# ============================================================
print("\n--- Descriptive Statistics ---")
analyzer = DescriptiveAnalyzer(data)
print(analyzer.full_summary().round(2))

# ============================================================
# 2. RUN REGRESSION
# ============================================================
print("\n--- Regression Analysis ---")
econ = EconometricsEngine(data)
results = econ.ols_regression(
    y_col='gdp',
    x_cols=['inflation', 'unemployment']
)
print("R-squared:", round(results['model_fit']['r_squared'], 4))
print(results['coefficients'].round(4))

# ============================================================
# 3. CALCULATE ECONOMIC INDICATORS
# ============================================================
print("\n--- Economic Indicators ---")

# Inflation rate
inflation = EconomicIndicators.inflation_rate(
    cpi_current=120, cpi_previous=115
)
print(f"Inflation Rate: {inflation:.2f}%")

# GDP growth
growth = EconomicIndicators.gdp_growth_rate(
    gdp_current=108.7, gdp_previous=105.2
)
print(f"GDP Growth Rate: {growth:.2f}%")

# Keynesian multiplier
mult = EconomicIndicators.multiplier(mpc=0.75)
print(f"Multiplier (MPC=0.75): {mult:.2f}")

# ============================================================
# 4. COST-BENEFIT ANALYSIS (e.g., for a project)
# ============================================================
print("\n--- Project Evaluation ---")
cba = CostBenefitAnalysis()

# Investment of $50,000, returns over 5 years
project_cash_flows = [-50000, 15000, 18000, 20000, 22000, 25000]
discount_rate = 0.08  # 8%

npv = cba.npv(project_cash_flows, discount_rate)
irr = cba.irr(project_cash_flows)

print(f"NPV: ${npv:,.2f}")
print(f"IRR: {irr*100:.2f}%")
print("Decision:", "ACCEPT" if npv > 0 else "REJECT")
