"""
Sector Performance Analysis
Using StatEcon Analyzer Toolkit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statecon_analyzer import (
    DescriptiveAnalyzer,
    EconometricsEngine,
    EconomicIndicators,
    HypothesisTestingSuite,
    CostBenefitAnalysis
)

# ============================================================
# STEP 1: LOAD YOUR DATA
# ============================================================
FILE_PATH = r"C:\Users\user\OneDrive\Documents\sector_performance_summary.csv"

print("=" * 70)
print("SECTOR PERFORMANCE ANALYSIS")
print("=" * 70)

try:
    data = pd.read_csv(FILE_PATH)
    print(f"\n[OK] File loaded successfully!")
    print(f"   Rows: {data.shape[0]}, Columns: {data.shape[1]}")
except FileNotFoundError:
    print(f"[ERROR] File not found at: {FILE_PATH}")
    exit()
except Exception as e:
    print(f"[ERROR] {e}")
    exit()

# ============================================================
# STEP 2: EXPLORE DATA STRUCTURE
# ============================================================
print("\n" + "=" * 70)
print("[1] DATA OVERVIEW")
print("=" * 70)

print("\n--- Column Names ---")
for i, col in enumerate(data.columns, 1):
    print(f"   {i}. {col} ({data[col].dtype})")

print("\n--- First 5 Rows ---")
print(data.head())

print("\n--- Data Types & Missing Values ---")
info_df = pd.DataFrame({
    'Data Type': data.dtypes,
    'Missing Values': data.isnull().sum(),
    'Missing %': (data.isnull().sum() / len(data) * 100).round(2),
    'Unique Values': data.nunique()
})
print(info_df)

# ============================================================
# STEP 3: DESCRIPTIVE STATISTICS
# ============================================================
print("\n" + "=" * 70)
print("[2] DESCRIPTIVE STATISTICS")
print("=" * 70)

analyzer = DescriptiveAnalyzer(data)
numeric_cols = analyzer.numeric_cols

if len(numeric_cols) == 0:
    print("[WARNING] No numeric columns found for statistical analysis.")
else:
    print(f"\nNumeric columns detected: {numeric_cols}")
    print("\n--- Full Statistical Summary ---")
    summary = analyzer.full_summary()
    print(summary.round(3))
    
    # Save summary
    summary.to_csv("descriptive_summary.csv")
    print("\n[SAVED] descriptive_summary.csv")

# ============================================================
# STEP 4: CATEGORICAL ANALYSIS (For sector names, categories)
# ============================================================
print("\n" + "=" * 70)
print("[3] CATEGORICAL ANALYSIS")
print("=" * 70)

categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    for col in categorical_cols:
        print(f"\n--- {col} ---")
        value_counts = data[col].value_counts()
        print(value_counts.head(10))
        print(f"   Total unique: {data[col].nunique()}")
else:
    print("No categorical columns found.")

# ============================================================
# STEP 5: CORRELATION ANALYSIS
# ============================================================
if len(numeric_cols) >= 2:
    print("\n" + "=" * 70)
    print("[4] CORRELATION ANALYSIS")
    print("=" * 70)
    
    corr_results = analyzer.correlation_analysis()
    print("\n--- Correlation Matrix ---")
    print(corr_results['matrix'].round(3))
    
    if not corr_results['strong_correlations'].empty:
        print("\n--- Strong Correlations (|r| > 0.5) ---")
        print(corr_results['strong_correlations'])
    else:
        print("\nNo strong correlations found.")

# ============================================================
# STEP 6: TOP & BOTTOM PERFORMERS (by sector)
# ============================================================
print("\n" + "=" * 70)
print("[5] TOP & BOTTOM PERFORMERS")
print("=" * 70)

# Try to identify the sector column
sector_col = None
for col in categorical_cols:
    if any(keyword in col.lower() for keyword in ['sector', 'industry', 'category', 'name']):
        sector_col = col
        break

if sector_col and numeric_cols:
    print(f"\nGrouping by: {sector_col}")
    
    for metric in numeric_cols[:3]:  # Top 3 metrics
        print(f"\n--- Top 5 by {metric} ---")
        top5 = data.nlargest(5, metric)[[sector_col, metric]]
        print(top5.to_string(index=False))
        
        print(f"\n--- Bottom 5 by {metric} ---")
        bottom5 = data.nsmallest(5, metric)[[sector_col, metric]]
        print(bottom5.to_string(index=False))

# ============================================================
# STEP 7: OUTLIER DETECTION
# ============================================================
if numeric_cols:
    print("\n" + "=" * 70)
    print("[6] OUTLIER DETECTION (IQR Method)")
    print("=" * 70)
    
    outliers = analyzer.outlier_detection()
    for col, info in outliers.items():
        if info['n_outliers'] > 0:
            print(f"\n{col}:")
            print(f"   Outliers: {info['n_outliers']} ({info['pct_outliers']:.1f}%)")
            print(f"   Normal range: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")

# ============================================================
# STEP 8: VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("[7] GENERATING VISUALIZATIONS")
print("=" * 70)

sns.set_style("whitegrid")

# Visualization 1: Distribution of numeric columns
if numeric_cols:
    n_cols = min(len(numeric_cols), 4)
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    
    for i, col in enumerate(numeric_cols):
        axes[i].hist(data[col].dropna(), bins=20, color='steelblue', edgecolor='black')
        axes[i].set_title(f'Distribution: {col}', fontsize=10)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Hide extra subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('distributions.png', dpi=100, bbox_inches='tight')
    print("[SAVED] distributions.png")
    plt.show()

# Visualization 2: Correlation heatmap
if len(numeric_cols) >= 2:
    plt.figure(figsize=(10, 8))
    corr = data[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=1)
    plt.title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=100, bbox_inches='tight')
    print("[SAVED] correlation_heatmap.png")
    plt.show()

# Visualization 3: Bar chart of sectors (if sector column exists)
if sector_col and numeric_cols:
    metric = numeric_cols[0]
    plt.figure(figsize=(12, 6))
    
    sector_avg = data.groupby(sector_col)[metric].mean().sort_values(ascending=False).head(15)
    sector_avg.plot(kind='bar', color='coral', edgecolor='black')
    plt.title(f'Average {metric} by {sector_col} (Top 15)', fontsize=14)
    plt.xlabel(sector_col)
    plt.ylabel(f'Average {metric}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('sector_performance.png', dpi=100, bbox_inches='tight')
    print("[SAVED] sector_performance.png")
    plt.show()

# Visualization 4: Box plot for outlier visualization
if numeric_cols:
    plt.figure(figsize=(12, 6))
    data[numeric_cols].boxplot()
    plt.title('Box Plot: Distribution & Outliers', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('boxplots.png', dpi=100, bbox_inches='tight')
    print("[SAVED] boxplots.png")
    plt.show()

# ============================================================
# STEP 9: KEY INSIGHTS SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("[8] KEY INSIGHTS")
print("=" * 70)

print(f"\n* Total records analyzed: {len(data)}")
print(f"* Numeric variables: {len(numeric_cols)}")
print(f"* Categorical variables: {len(categorical_cols)}")

if numeric_cols:
    print(f"\n* Best performing metric (highest mean):")
    means = data[numeric_cols].mean().sort_values(ascending=False)
    print(f"   {means.index[0]}: {means.iloc[0]:.2f}")
    
    print(f"\n* Most volatile metric (highest CV%):")
    cv = (data[numeric_cols].std() / data[numeric_cols].mean() * 100).abs().sort_values(ascending=False)
    print(f"   {cv.index[0]}: {cv.iloc[0]:.2f}%")

if sector_col and numeric_cols:
    metric = numeric_cols[0]
    best_sector = data.loc[data[metric].idxmax(), sector_col]
    worst_sector = data.loc[data[metric].idxmin(), sector_col]
    print(f"\n* Best performing {sector_col} (by {metric}): {best_sector}")
    print(f"* Worst performing {sector_col} (by {metric}): {worst_sector}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print("\nGenerated files in current folder:")
print("   - descriptive_summary.csv")
print("   - distributions.png")
print("   - correlation_heatmap.png")
print("   - sector_performance.png")
print("   - boxplots.png")

input("\nPress Enter to exit...")
