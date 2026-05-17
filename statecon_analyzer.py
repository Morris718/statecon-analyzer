"""
StatEcon Analyzer v1.0
A toolkit for Applied Statistics & Economics professionals
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# MODULE 1: DESCRIPTIVE STATISTICS ENGINE
# ============================================================

class DescriptiveAnalyzer:
    """Comprehensive descriptive statistics for economic data."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    def full_summary(self) -> pd.DataFrame:
        """Generate a comprehensive statistical summary."""
        summary = {}
        for col in self.numeric_cols:
            series = self.data[col].dropna()
            summary[col] = {
                'count': len(series),
                'mean': series.mean(),
                'median': series.median(),
                'mode': series.mode().iloc[0] if not series.mode().empty else None,
                'std_dev': series.std(),
                'variance': series.var(),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'min': series.min(),
                'max': series.max(),
                'range': series.max() - series.min(),
                'iqr': series.quantile(0.75) - series.quantile(0.25),
                'cv': (series.std() / series.mean()) * 100 if series.mean() != 0 else None,
                'missing_pct': (self.data[col].isna().sum() / len(self.data)) * 100
            }
        return pd.DataFrame(summary).T
    
    def normality_tests(self) -> pd.DataFrame:
        """Run multiple normality tests on all numeric columns."""
        results = {}
        for col in self.numeric_cols:
            series = self.data[col].dropna()
            if len(series) >= 8:
                shapiro_stat, shapiro_p = stats.shapiro(series[:5000])
                ks_stat, ks_p = stats.kstest(series, 'norm', 
                                              args=(series.mean(), series.std()))
                dagostino_stat, dagostino_p = stats.normaltest(series)
                results[col] = {
                    'shapiro_stat': shapiro_stat,
                    'shapiro_p': shapiro_p,
                    'ks_stat': ks_stat,
                    'ks_p': ks_p,
                    'dagostino_stat': dagostino_stat,
                    'dagostino_p': dagostino_p,
                    'is_normal_5pct': shapiro_p > 0.05
                }
        return pd.DataFrame(results).T
    
    def correlation_analysis(self, method='pearson') -> Dict:
        """Detailed correlation analysis."""
        corr_matrix = self.data[self.numeric_cols].corr(method=method)
        
        # Find strongest correlations
        strong_corrs = []
        for i in range(len(self.numeric_cols)):
            for j in range(i+1, len(self.numeric_cols)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.5:
                    strong_corrs.append({
                        'var1': self.numeric_cols[i],
                        'var2': self.numeric_cols[j],
                        'correlation': r,
                        'strength': 'Strong' if abs(r) > 0.7 else 'Moderate'
                    })
        
        return {
            'matrix': corr_matrix,
            'strong_correlations': pd.DataFrame(strong_corrs)
        }
    
    def outlier_detection(self, method='iqr') -> Dict:
        """Detect outliers using IQR or Z-score method."""
        outliers = {}
        for col in self.numeric_cols:
            series = self.data[col].dropna()
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                mask = (series < lower) | (series > upper)
            else:  # z-score
                z_scores = np.abs(stats.zscore(series))
                mask = z_scores > 3
                lower, upper = series.mean() - 3*series.std(), series.mean() + 3*series.std()
            
            outliers[col] = {
                'n_outliers': mask.sum(),
                'pct_outliers': (mask.sum() / len(series)) * 100,
                'lower_bound': lower,
                'upper_bound': upper,
                'outlier_indices': series[mask].index.tolist()
            }
        return outliers


# ============================================================
# MODULE 2: REGRESSION & ECONOMETRICS ENGINE
# ============================================================

class EconometricsEngine:
    """Advanced regression and econometric analysis."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = None
    
    def ols_regression(self, y_col: str, x_cols: List[str], 
                        add_constant: bool = True) -> Dict:
        """Perform OLS regression with comprehensive diagnostics."""
        y = self.data[y_col].dropna()
        X = self.data[x_cols].loc[y.index]
        
        # Drop rows with any NaN
        mask = X.notna().all(axis=1)
        y, X = y[mask], X[mask]
        
        if add_constant:
            X = sm.add_constant(X)
        
        model = OLS(y, X).fit()
        self.results = model
        
        # Diagnostic tests
        diagnostics = self._regression_diagnostics(model, y, X)
        
        return {
            'summary': model.summary(),
            'coefficients': pd.DataFrame({
                'coef': model.params,
                'std_err': model.bse,
                't_stat': model.tvalues,
                'p_value': model.pvalues,
                'ci_lower': model.conf_int()[0],
                'ci_upper': model.conf_int()[1],
                'significant_5pct': model.pvalues < 0.05
            }),
            'model_fit': {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_p_value': model.f_pvalue,
                'aic': model.aic,
                'bic': model.bic,
                'log_likelihood': model.llf,
                'n_observations': int(model.nobs)
            },
            'diagnostics': diagnostics,
            'model': model
        }
    
    def _regression_diagnostics(self, model, y, X) -> Dict:
        """Run diagnostic tests for regression assumptions."""
        from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
        from statsmodels.stats.stattools import durbin_watson
        
        residuals = model.resid
        
        # Durbin-Watson (autocorrelation)
        dw = durbin_watson(residuals)
        
        # Breusch-Pagan (heteroscedasticity)
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
        except:
            bp_stat, bp_p = None, None
        
        # Jarque-Bera (normality of residuals)
        jb_stat, jb_p = stats.jarque_bera(residuals)
        
        return {
            'durbin_watson': dw,
            'autocorrelation': 'Possible' if dw < 1.5 or dw > 2.5 else 'Unlikely',
            'breusch_pagan_stat': bp_stat,
            'breusch_pagan_p': bp_p,
            'heteroscedasticity': 'Detected' if bp_p and bp_p < 0.05 else 'Not detected',
            'jarque_bera_stat': jb_stat,
            'jarque_bera_p': jb_p,
            'residuals_normal': jb_p > 0.05,
            'mean_vif': self._calculate_vif(X) if X.shape[1] > 1 else None
        }
    
    def _calculate_vif(self, X) -> pd.DataFrame:
        """Calculate Variance Inflation Factors."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = pd.DataFrame()
        cols = [c for c in X.columns if c != 'const']
        X_vif = X[cols] if 'const' not in cols else X
        vif_data['variable'] = cols
        vif_data['vif'] = [variance_inflation_factor(X_vif.values, i) 
                           for i in range(len(cols))]
        vif_data['multicollinearity'] = vif_data['vif'].apply(
            lambda x: 'Severe' if x > 10 else ('Moderate' if x > 5 else 'Low'))
        return vif_data
    
    def logistic_regression(self, y_col: str, x_cols: List[str]) -> Dict:
        """Logistic regression for binary economic outcomes."""
        from statsmodels.discrete.discrete_model import Logit
        
        y = self.data[y_col].dropna()
        X = sm.add_constant(self.data[x_cols].loc[y.index].dropna())
        
        mask = X.notna().all(axis=1)
        y, X = y[mask], X[mask]
        
        model = Logit(y, X).fit(disp=0)
        
        return {
            'summary': model.summary(),
            'odds_ratios': np.exp(model.params),
            'marginal_effects': model.get_margeff().summary(),
            'pseudo_r_squared': model.prsquared,
            'aic': model.aic,
            'bic': model.bic
        }


# ============================================================
# MODULE 3: TIME SERIES & FORECASTING ENGINE
# ============================================================

class TimeSeriesEngine:
    """Time series analysis for economic data."""
    
    def __init__(self, series: pd.Series, freq: str = None):
        self.series = series.dropna()
        self.freq = freq
    
    def decompose(self, model='additive', period=None):
        """Decompose time series into trend, seasonal, and residual."""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        result = seasonal_decompose(self.series, model=model, period=period)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        result.observed.plot(ax=axes[0], title='Observed')
        result.trend.plot(ax=axes[1], title='Trend')
        result.seasonal.plot(ax=axes[2], title='Seasonal')
        result.resid.plot(ax=axes[3], title='Residual')
        plt.tight_layout()
        
        return result, fig
    
    def stationarity_test(self) -> Dict:
        """Augmented Dickey-Fuller and KPSS tests."""
        from statsmodels.tsa.stattools import adfuller, kpss
        
        # ADF Test
        adf_result = adfuller(self.series)
        
        # KPSS Test
        kpss_result = kpss(self.series, regression='c', nlags='auto')
        
        return {
            'adf_statistic': adf_result[0],
            'adf_p_value': adf_result[1],
            'adf_critical_values': adf_result[4],
            'adf_stationary': adf_result[1] < 0.05,
            'kpss_statistic': kpss_result[0],
            'kpss_p_value': kpss_result[1],
            'kpss_critical_values': kpss_result[3],
            'kpss_stationary': kpss_result[1] > 0.05,
            'recommendation': self._stationarity_recommendation(adf_result[1], kpss_result[1])
        }
    
    def _stationarity_recommendation(self, adf_p, kpss_p):
        if adf_p < 0.05 and kpss_p > 0.05:
            return "Series is stationary"
        elif adf_p >= 0.05 and kpss_p <= 0.05:
            return "Series is non-stationary - differencing recommended"
        elif adf_p < 0.05 and kpss_p <= 0.05:
            return "Series is trend-stationary - detrending recommended"
        else:
            return "Results inconclusive - further investigation needed"
    
    def auto_arima_forecast(self, steps: int = 12, max_order: int = 5) -> Dict:
        """Find best ARIMA model and forecast."""
        best_aic = np.inf
        best_order = None
        best_model = None
        
        for p in range(max_order):
            for d in range(3):
                for q in range(max_order):
                    try:
                        model = ARIMA(self.series, order=(p, d, q)).fit()
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_order = (p, d, q)
                            best_model = model
                    except:
                        continue
        
        forecast = best_model.forecast(steps=steps)
        conf_int = best_model.get_forecast(steps=steps).conf_int()
        
        return {
            'best_order': best_order,
            'aic': best_aic,
            'bic': best_model.bic,
            'forecast': forecast,
            'confidence_interval': conf_int,
            'model': best_model,
            'residual_diagnostics': {
                'ljung_box_p': sm.stats.acorr_ljungbox(
                    best_model.resid, lags=[10])['lb_pvalue'].values[0],
                'residuals_normal': stats.jarque_bera(best_model.resid)[1] > 0.05
            }
        }
    
    def growth_rate_analysis(self) -> pd.DataFrame:
        """Calculate various growth rates for economic data."""
        df = pd.DataFrame({'value': self.series})
        df['pct_change'] = df['value'].pct_change() * 100
        df['log_return'] = np.log(df['value'] / df['value'].shift(1)) * 100
        df['cumulative_growth'] = ((df['value'] / df['value'].iloc[0]) - 1) * 100
        df['moving_avg_4'] = df['value'].rolling(window=4).mean()
        df['moving_avg_12'] = df['value'].rolling(window=12).mean()
        return df


# ============================================================
# MODULE 4: ECONOMIC INDICATORS TOOLKIT
# ============================================================

class EconomicIndicators:
    """Tools for calculating and analyzing economic indicators."""
    
    @staticmethod
    def calculate_cpi(basket_current: Dict[str, float], 
                      basket_base: Dict[str, float]) -> float:
        """Calculate Consumer Price Index."""
        current_cost = sum(basket_current.values())
        base_cost = sum(basket_base.values())
        return (current_cost / base_cost) * 100
    
    @staticmethod
    def inflation_rate(cpi_current: float, cpi_previous: float) -> float:
        """Calculate inflation rate from CPI values."""
        return ((cpi_current - cpi_previous) / cpi_previous) * 100
    
    @staticmethod
    def real_gdp(nominal_gdp: float, gdp_deflator: float) -> float:
        """Calculate Real GDP."""
        return (nominal_gdp / gdp_deflator) * 100
    
    @staticmethod
    def gdp_growth_rate(gdp_current: float, gdp_previous: float) -> float:
        """Calculate GDP growth rate."""
        return ((gdp_current - gdp_previous) / gdp_previous) * 100
    
    @staticmethod
    def unemployment_rate(unemployed: float, labor_force: float) -> float:
        """Calculate unemployment rate."""
        return (unemployed / labor_force) * 100
    
    @staticmethod
    def labor_force_participation(labor_force: float, 
                                   working_age_pop: float) -> float:
        """Calculate labor force participation rate."""
        return (labor_force / working_age_pop) * 100
    
    @staticmethod
    def gini_coefficient(incomes: np.array) -> float:
        """Calculate Gini coefficient for income inequality."""
        sorted_incomes = np.sort(incomes)
        n = len(sorted_incomes)
        cumulative = np.cumsum(sorted_incomes)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_incomes))) / (n * np.sum(sorted_incomes)) - (n + 1) / n
        return gini
    
    @staticmethod
    def lorenz_curve(incomes: np.array) -> Tuple[np.array, np.array]:
        """Calculate Lorenz curve coordinates."""
        sorted_incomes = np.sort(incomes)
        cumulative_share = np.cumsum(sorted_incomes) / np.sum(sorted_incomes)
        population_share = np.arange(1, len(incomes) + 1) / len(incomes)
        return population_share, cumulative_share
    
    @staticmethod
    def terms_of_trade(export_price_index: float, 
                       import_price_index: float) -> float:
        """Calculate Terms of Trade."""
        return (export_price_index / import_price_index) * 100
    
    @staticmethod
    def multiplier(mpc: float) -> float:
        """Calculate Keynesian multiplier."""
        return 1 / (1 - mpc)
    
    @staticmethod
    def elasticity(pct_change_quantity: float, 
                   pct_change_price: float) -> Dict:
        """Calculate and interpret price elasticity of demand."""
        elasticity = pct_change_quantity / pct_change_price
        if abs(elasticity) > 1:
            interpretation = "Elastic"
        elif abs(elasticity) < 1:
            interpretation = "Inelastic"
        else:
            interpretation = "Unit Elastic"
        
        return {
            'elasticity': elasticity,
            'absolute_value': abs(elasticity),
            'interpretation': interpretation
        }


# ============================================================
# MODULE 5: HYPOTHESIS TESTING SUITE
# ============================================================

class HypothesisTestingSuite:
    """Comprehensive hypothesis testing for economic research."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def t_test(self, col1: str, col2: str = None, 
               mu: float = 0, test_type: str = 'independent') -> Dict:
        """Perform t-tests (one-sample, independent, paired)."""
        if test_type == 'one_sample':
            stat, p = stats.ttest_1samp(self.data[col1].dropna(), mu)
            test_name = f"One-sample t-test (H0: μ = {mu})"
        elif test_type == 'independent':
            stat, p = stats.ttest_ind(self.data[col1].dropna(), 
                                       self.data[col2].dropna())
            test_name = f"Independent t-test ({col1} vs {col2})"
        else:
            stat, p = stats.ttest_rel(self.data[col1].dropna(), 
                                       self.data[col2].dropna())
            test_name = f"Paired t-test ({col1} vs {col2})"
        
        return {
            'test': test_name,
            'statistic': stat,
            'p_value': p,
            'significant_1pct': p < 0.01,
            'significant_5pct': p < 0.05,
            'significant_10pct': p < 0.10,
            'decision': 'Reject H0' if p < 0.05 else 'Fail to reject H0'
        }
    
    def anova(self, groups_col: str, value_col: str) -> Dict:
        """One-way ANOVA test."""
        groups = [group[value_col].values for name, group 
                  in self.data.groupby(groups_col)]
        stat, p = stats.f_oneway(*groups)
        
        return {
            'test': 'One-way ANOVA',
            'f_statistic': stat,
            'p_value': p,
            'significant': p < 0.05,
            'n_groups': len(groups),
            'decision': 'Reject H0 (means differ)' if p < 0.05 
                        else 'Fail to reject H0 (means equal)'
        }
    
    def chi_square_test(self, col1: str, col2: str) -> Dict:
        """Chi-square test of independence."""
        contingency = pd.crosstab(self.data[col1], self.data[col2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        
        # Cramér's V
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        return {
            'test': 'Chi-square test of independence',
            'chi2_statistic': chi2,
            'p_value': p,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'effect_size': 'Small' if cramers_v < 0.3 
                          else ('Medium' if cramers_v < 0.5 else 'Large'),
            'significant': p < 0.05,
            'contingency_table': contingency,
            'expected_frequencies': pd.DataFrame(
                expected, index=contingency.index, columns=contingency.columns)
        }
    
    def mann_whitney(self, col1: str, col2: str) -> Dict:
        """Non-parametric alternative to independent t-test."""
        stat, p = stats.mannwhitneyu(self.data[col1].dropna(), 
                                      self.data[col2].dropna())
        return {
            'test': 'Mann-Whitney U test',
            'statistic': stat,
            'p_value': p,
            'significant': p < 0.05,
            'decision': 'Reject H0' if p < 0.05 else 'Fail to reject H0'
        }
    
    def granger_causality(self, col1: str, col2: str, 
                          max_lag: int = 4) -> Dict:
        """Granger causality test for economic variables."""
        from statsmodels.tsa.stattools import grangercausalitytests
        
        test_data = self.data[[col1, col2]].dropna()
        results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        
        summary = {}
        for lag in range(1, max_lag + 1):
            p_value = results[lag][0]['ssr_ftest'][1]
            summary[f'lag_{lag}'] = {
                'f_stat': results[lag][0]['ssr_ftest'][0],
                'p_value': p_value,
                'causal': p_value < 0.05
            }
        
        return {
            'test': f'Granger Causality: {col2} → {col1}',
            'results_by_lag': summary,
            'any_causality': any(v['causal'] for v in summary.values())
        }


# ============================================================
# MODULE 6: VISUALIZATION ENGINE
# ============================================================

class StatEconVisualizer:
    """Publication-quality visualizations for statistical & economic data."""
    
    def __init__(self, style='seaborn-v0_8-whitegrid'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')
    
    def distribution_plot(self, data: pd.Series, title: str = None):
        """Plot distribution with normal curve overlay."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Histogram with KDE
        axes[0].hist(data, bins=30, density=True, alpha=0.7, color='steelblue')
        x = np.linspace(data.min(), data.max(), 100)
        axes[0].plot(x, stats.norm.pdf(x, data.mean(), data.std()), 
                     'r-', lw=2, label='Normal')
        axes[0].set_title('Distribution')
        axes[0].legend()
        
        # Q-Q Plot
        stats.probplot(data, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot')
        
        # Box Plot
        axes[2].boxplot(data, vert=True)
        axes[2].set_title('Box Plot')
        
        plt.suptitle(title or data.name or 'Distribution Analysis', fontsize=14)
        plt.tight_layout()
        return fig
    
    def correlation_heatmap(self, data: pd.DataFrame, title: str = None):
        """Create an annotated correlation heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = data.select_dtypes(include=[np.number]).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                    cmap='RdBu_r', center=0, square=True, ax=ax,
                    vmin=-1, vmax=1)
        ax.set_title(title or 'Correlation Matrix')
        plt.tight_layout()
        return fig
    
    def regression_plot(self, data: pd.DataFrame, x_col: str, y_col: str):
        """Scatter plot with regression line and confidence band."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter + regression line
        sns.regplot(data=data, x=x_col, y=y_col, ax=axes[0],
                    scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        axes[0].set_title(f'{y_col} vs {x_col}')
        
        # Residual plot
        X = sm.add_constant(data[x_col])
        model = OLS(data[y_col], X).fit()
        axes[1].scatter(model.fittedvalues, model.resid, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Fitted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        
        plt.tight_layout()
        return fig
    
    def lorenz_curve_plot(self, incomes: np.array):
        """Plot Lorenz curve with Gini coefficient."""
        pop_share, income_share = EconomicIndicators.lorenz_curve(incomes)
        gini = EconomicIndicators.gini_coefficient(incomes)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Equality')
        ax.plot(np.insert(pop_share, 0, 0), 
                np.insert(income_share, 0, 0), 
                'b-', lw=2, label=f'Lorenz Curve (Gini = {gini:.4f})')
        ax.fill_between(np.insert(pop_share, 0, 0),
                        np.insert(income_share, 0, 0),
                        np.insert(pop_share, 0, 0), alpha=0.2)
        ax.set_xlabel('Cumulative Share of Population')
        ax.set_ylabel('Cumulative Share of Income')
        ax.set_title('Lorenz Curve & Income Inequality')
        ax.legend()
        plt.tight_layout()
        return fig
    
    def time_series_plot(self, data: pd.DataFrame, cols: List[str],
                         title: str = None):
        """Multi-panel time series visualization."""
        n = len(cols)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4*n), sharex=True)
        if n == 1:
            axes = [axes]
        
        for i, col in enumerate(cols):
            axes[i].plot(data.index, data[col], lw=1.5)
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3)
            
            # Add trend line
            x_numeric = np.arange(len(data[col].dropna()))
            z = np.polyfit(x_numeric, data[col].dropna().values, 1)
            p = np.poly1d(z)
            axes[i].plot(data[col].dropna().index, p(x_numeric), 
                        'r--', alpha=0.7, label='Trend')
            axes[i].legend()
        
        plt.suptitle(title or 'Time Series Analysis', fontsize=14)
        plt.tight_layout()
        return fig


# ============================================================
# MODULE 7: COST-BENEFIT ANALYSIS
# ============================================================

class CostBenefitAnalysis:
    """Economic cost-benefit analysis toolkit."""
    
    @staticmethod
    def npv(cash_flows: List[float], discount_rate: float) -> float:
        """Calculate Net Present Value."""
        return sum(cf / (1 + discount_rate)**t 
                   for t, cf in enumerate(cash_flows))
    
    @staticmethod
    def irr(cash_flows: List[float], guess: float = 0.1) -> float:
        """Calculate Internal Rate of Return."""
        return np.irr(cash_flows) if hasattr(np, 'irr') else \
               CostBenefitAnalysis._irr_newton(cash_flows, guess)
    
    @staticmethod
    def _irr_newton(cash_flows, guess=0.1, tol=1e-6, max_iter=1000):
        rate = guess
        for _ in range(max_iter):
            npv = sum(cf / (1 + rate)**t for t, cf in enumerate(cash_flows))
            dnpv = sum(-t * cf / (1 + rate)**(t+1) for t, cf in enumerate(cash_flows))
            if abs(dnpv) < 1e-12:
                break
            rate -= npv / dnpv
            if abs(npv) < tol:
                break
        return rate
    
    @staticmethod
    def benefit_cost_ratio(benefits: List[float], costs: List[float],
                           discount_rate: float) -> Dict:
        """Calculate Benefit-Cost Ratio."""
        pv_benefits = sum(b / (1 + discount_rate)**t 
                         for t, b in enumerate(benefits))
        pv_costs = sum(c / (1 + discount_rate)**t 
                       for t, c in enumerate(costs))
        
        bcr = pv_benefits / pv_costs if pv_costs > 0 else float('inf')
        
        return {
            'pv_benefits': pv_benefits,
            'pv_costs': pv_costs,
            'bcr': bcr,
            'net_benefit': pv_benefits - pv_costs,
            'recommendation': 'Accept' if bcr > 1 else 'Reject'
        }
    
    @staticmethod
    def sensitivity_analysis(cash_flows: List[float], 
                            base_rate: float,
                            rate_range: Tuple[float, float] = (0.01, 0.20),
                            steps: int = 20) -> pd.DataFrame:
        """Sensitivity analysis across discount rates."""
        rates = np.linspace(rate_range[0], rate_range[1], steps)
        results = []
        for r in rates:
            npv = CostBenefitAnalysis.npv(cash_flows, r)
            results.append({
                'discount_rate': r,
                'npv': npv,
                'viable': npv > 0
            })
        return pd.DataFrame(results)


# ============================================================
# MODULE 8: REPORT GENERATOR
# ============================================================

class ReportGenerator:
    """Generate automated statistical reports."""
    
    def __init__(self, data: pd.DataFrame, title: str = "Statistical Report"):
        self.data = data
        self.title = title
        self.sections = []
    
    def add_section(self, title: str, content: str):
        self.sections.append({'title': title, 'content': content})
    
    def auto_report(self) -> str:
        """Generate a comprehensive automatic report."""
        analyzer = DescriptiveAnalyzer(self.data)
        
        report = f"""
{'='*60}
{self.title.center(60)}
{'='*60}

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {self.data.shape[0]} observations, {self.data.shape[1]} variables

{'─'*60}
1. DATASET OVERVIEW
{'─'*60}
Variables: {', '.join(self.data.columns.tolist())}
Numeric Variables: {len(analyzer.numeric_cols)}
Missing Values:
{self.data.isnull().sum().to_string()}

{'─'*60}
2. DESCRIPTIVE STATISTICS
{'─'*60}
{analyzer.full_summary().to_string()}

{'─'*60}
3. NORMALITY TESTS
{'─'*60}
{analyzer.normality_tests().to_string()}

{'─'*60}
4. CORRELATION ANALYSIS
{'─'*60}
{analyzer.correlation_analysis()['matrix'].to_string()}

Strong Correlations:
{analyzer.correlation_analysis()['strong_correlations'].to_string()
 if not analyzer.correlation_analysis()['strong_correlations'].empty 
 else 'No strong correlations found (|r| > 0.5)'}

{'─'*60}
5. OUTLIER DETECTION (IQR Method)
{'─'*60}
"""
        outliers = analyzer.outlier_detection()
        for col, info in outliers.items():
            report += f"\n{col}: {info['n_outliers']} outliers ({info['pct_outliers']:.1f}%)"
        
        report += f"""

{'='*60}
END OF REPORT
{'='*60}
"""
        return report
    
    def save_report(self, filename: str = 'report.txt'):
        """Save report to file."""
        report = self.auto_report()
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Report saved to {filename}")


# ============================================================
# MAIN: DEMO USAGE
# ============================================================

def demo():
    """Demonstrate the StatEcon Analyzer toolkit."""
    
    print("🎯 StatEcon Analyzer v1.0")
    print("=" * 50)
    
    # Create sample economic dataset
    np.random.seed(42)
    n = 200
    
    data = pd.DataFrame({
        'gdp_growth': np.random.normal(3.2, 1.5, n),
        'inflation': np.random.normal(2.5, 0.8, n),
        'unemployment': np.random.normal(5.5, 1.2, n),
        'interest_rate': np.random.normal(3.0, 1.0, n),
        'investment': np.random.normal(20, 5, n),
        'trade_balance': np.random.normal(-2, 3, n),
        'consumer_confidence': np.random.normal(100, 15, n),
    })
    
    # Add some realistic relationships
    data['gdp_growth'] = data['gdp_growth'] + 0.5 * data['investment']/20 - 0.3 * data['unemployment']/5
    data['inflation'] = data['inflation'] + 0.2 * data['gdp_growth'] + np.random.normal(0, 0.3, n)
    
    # 1. Descriptive Analysis
    print("\n📊 Descriptive Analysis")
    analyzer = DescriptiveAnalyzer(data)
    print(analyzer.full_summary().round(3))
    
    # 2. Econometrics
    print("\n📈 OLS Regression: GDP Growth")
    econ = EconometricsEngine(data)
    results = econ.ols_regression('gdp_growth', 
                                  ['unemployment', 'investment', 'interest_rate'])
    print(f"R² = {results['model_fit']['r_squared']:.4f}")
    print(f"F-stat p-value = {results['model_fit']['f_p_value']:.6f}")
    print(results['coefficients'][['coef', 'p_value', 'significant_5pct']].round(4))
    
    # 3. Economic Indicators
    print("\n💹 Economic Indicators")
    econ_ind = EconomicIndicators()
    print(f"Multiplier (MPC=0.8): {econ_ind.multiplier(0.8):.2f}")
    print(f"Elasticity: {econ_ind.elasticity(-10, 5)}")
    
    incomes = np.random.lognormal(10, 1, 1000)
    print(f"Gini Coefficient: {econ_ind.gini_coefficient(incomes):.4f}")
    
    # 4. Hypothesis Testing
    print("\n🔬 Hypothesis Testing")
    tester = HypothesisTestingSuite(data)
    result = tester.t_test('gdp_growth', mu=3.0, test_type='one_sample')
    print(f"H0: μ_gdp = 3.0 → {result['decision']} (p={result['p_value']:.4f})")
    
    # 5. Cost-Benefit Analysis
    print("\n💰 Cost-Benefit Analysis")
    cba = CostBenefitAnalysis()
    cash_flows = [-100000, 25000, 30000, 35000, 40000, 45000]
    print(f"NPV (10%): ${cba.npv(cash_flows, 0.10):,.2f}")
    print(f"IRR: {cba._irr_newton(cash_flows)*100:.2f}%")
    
    # 6. Generate Report
    print("\n📝 Generating Report...")
    reporter = ReportGenerator(data, "Economic Analysis Report")
    report = reporter.auto_report()
    print(report[:500] + "...")
    
    print("\n✅ Demo Complete!")
    print("Available modules:")
    print("  1. DescriptiveAnalyzer - Comprehensive descriptive statistics")
    print("  2. EconometricsEngine - OLS, Logistic regression, diagnostics")
    print("  3. TimeSeriesEngine - ARIMA, decomposition, stationarity tests")
    print("  4. EconomicIndicators - CPI, GDP, Gini, elasticity, multipliers")
    print("  5. HypothesisTestingSuite - t-tests, ANOVA, chi-square, Granger")
    print("  6. StatEconVisualizer - Publication-quality charts")
    print("  7. CostBenefitAnalysis - NPV, IRR, sensitivity analysis")
    print("  8. ReportGenerator - Automated statistical reports")


if __name__ == "__main__":
    demo()
