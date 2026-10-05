"""
OADRP Evaluation Notebook — companion to experiments/evaluation.py.

This script is designed to be run as a Jupyter-style notebook using
`jupyter notebook` or converted via `jupytext`. It produces all the
visualizations and analysis tables for the experiment deliverable.

To run: python experiments/evaluation_notebook.py
     or convert to .ipynb: jupytext --to notebook evaluation_notebook.py

Produces:
  - Peak demand comparison charts (baseline vs cost-first vs occupant-first)
  - Event severity distribution
  - Cost-first vs occupant-first divergence analysis
  - Per-day forecast error analysis
  - Equipment utilization heatmap
  - Thermal model comparison (1R1C vs 2R2C)
  - Summary statistics tables
"""
# %% [markdown]
# # OADRP Evaluation Notebook
# ## Occupant-Aware Demand Response Planner — Experiment Results

# %%
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not available. Using matplotlib fallback for charts.")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_write_image(fig, filename: str, width: int = 1200, height: int = 500):
    """Safely attempt to write an image with Plotly, ignoring missing kaleido."""
    try:
        fig.write_image(str(CHARTS_DIR / filename), width=width, height=height)
    except Exception:
        pass


# %% [markdown]
# ## 1. Load Generated Data and Results

# %%
def load_results():
    """Load all generated data and evaluation results."""
    load_df = pd.read_csv(DATA_DIR / "load_timeseries.csv", parse_dates=["timestamp"])
    occ_df = pd.read_csv(DATA_DIR / "occupancy.csv", parse_dates=["timestamp"])

    daily_df = pd.read_csv(RESULTS_DIR / "daily_comparison.csv")
    with open(RESULTS_DIR / "evaluation_summary.json") as f:
        summary = json.load(f)

    forecast_df = None
    if (RESULTS_DIR / "per_day_forecast_errors.csv").exists():
        forecast_df = pd.read_csv(RESULTS_DIR / "per_day_forecast_errors.csv")

    moderate_df = None
    if (RESULTS_DIR / "moderate_event_divergence.csv").exists():
        moderate_df = pd.read_csv(RESULTS_DIR / "moderate_event_divergence.csv")

    print(f"Loaded {len(load_df)} load observations across {len(daily_df)} days")
    print(f"Peak threshold: {summary['peak_threshold_kw']} kW")
    print(f"Peak event days: {summary['n_peak_event_days']}")
    if moderate_df is not None:
        print(f"Moderate event cases: {len(moderate_df)}")
    return load_df, occ_df, daily_df, summary, forecast_df, moderate_df


# %% [markdown]
# ## 2. Peak Demand Comparison

# %%
def plot_peak_comparison(daily_df, summary):
    """Bar chart comparing baseline, cost-first, and occupant-first peak demand."""
    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Baseline", x=daily_df["date"], y=daily_df["baseline_peak_kw"],
            marker_color="#EF4444", opacity=0.7
        ))
        fig.add_trace(go.Bar(
            name="Cost-First", x=daily_df["date"], y=daily_df["cost_first_peak_kw"],
            marker_color="#3B82F6", opacity=0.8
        ))
        fig.add_trace(go.Bar(
            name="Occupant-First", x=daily_df["date"], y=daily_df["occupant_first_peak_kw"],
            marker_color="#10B981", opacity=0.8
        ))
        fig.add_hline(y=summary["peak_threshold_kw"], line_dash="dash", line_color="orange",
                      annotation_text=f"Threshold: {summary['peak_threshold_kw']} kW")
        fig.update_layout(
            title="Daily Peak Demand: Baseline vs Optimized Plans",
            xaxis_title="Date", yaxis_title="Peak Demand (kW)",
            barmode="group", template="plotly_dark",
            height=500, font=dict(size=12)
        )
        fig.write_html(str(CHARTS_DIR / "peak_comparison.html"))
        safe_write_image(fig, "peak_comparison.png", width=1200, height=500)
        print("Saved peak comparison chart (Plotly HTML)")
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(daily_df))
        w = 0.25
        ax.bar(x - w, daily_df["baseline_peak_kw"], w, label="Baseline", color="#EF4444", alpha=0.7)
        ax.bar(x, daily_df["cost_first_peak_kw"], w, label="Cost-First", color="#3B82F6", alpha=0.8)
        ax.bar(x + w, daily_df["occupant_first_peak_kw"], w, label="Occupant-First", color="#10B981", alpha=0.8)
        ax.axhline(y=summary["peak_threshold_kw"], color="orange", linestyle="--", label=f"Threshold ({summary['peak_threshold_kw']} kW)")
        ax.set_xlabel("Day")
        ax.set_ylabel("Peak Demand (kW)")
        ax.set_title("Daily Peak Demand: Baseline vs Optimized Plans")
        ax.legend()
        ax.set_xticks(x[::5])
        ax.set_xticklabels(daily_df["date"].values[::5], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(str(CHARTS_DIR / "peak_comparison.png"), dpi=150)
        plt.close()
        print("Saved peak comparison chart (matplotlib)")


# %% [markdown]
# ## 3. Event Severity Distribution

# %%
def plot_severity_distribution(daily_df):
    """Pie/donut chart of event severity distribution."""
    if "event_severity" not in daily_df.columns:
        print("No event_severity column — run evaluation.py first")
        return

    severity_counts = daily_df["event_severity"].value_counts()
    colors = {"NONE": "#9CA3AF", "MILD": "#FCD34D", "MODERATE": "#FB923C",
              "SEVERE": "#EF4444", "EXTREME": "#991B1B"}

    if HAS_PLOTLY:
        fig = go.Figure(go.Pie(
            labels=severity_counts.index.tolist(),
            values=severity_counts.values.tolist(),
            hole=0.4,
            marker_colors=[colors.get(s, "#6B7280") for s in severity_counts.index],
            textinfo="label+percent+value",
        ))
        fig.update_layout(
            title="DR Event Severity Distribution (30 Days)",
            template="plotly_dark", height=400,
        )
        fig.write_html(str(CHARTS_DIR / "severity_distribution.html"))
        safe_write_image(fig, "severity_distribution.png", width=600, height=400)
        print("Saved severity distribution chart (Plotly HTML)")
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(severity_counts.values, labels=severity_counts.index,
               colors=[colors.get(s, "#6B7280") for s in severity_counts.index],
               autopct="%1.1f%%", startangle=90)
        ax.set_title("DR Event Severity Distribution (30 Days)")
        plt.tight_layout()
        plt.savefig(str(CHARTS_DIR / "severity_distribution.png"), dpi=150)
        plt.close()
        print("Saved severity distribution chart (matplotlib)")


# %% [markdown]
# ## 4. Cost-First vs Occupant-First Divergence Analysis

# %%
def plot_divergence_analysis(daily_df):
    """Analyze and visualize when cost-first and occupant-first plans diverge."""
    if "comfort_divergence" not in daily_df.columns:
        print("No divergence columns — run evaluation.py first")
        return

    event_df = daily_df[daily_df["event_severity"] != "NONE"]
    if len(event_df) == 0:
        print("No peak events to analyze divergence")
        return

    if HAS_PLOTLY:
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=["Comfort Risk: Cost-First vs Occupant-First",
                                            "Plan Divergence by Day"],
                            vertical_spacing=0.15)

        fig.add_trace(go.Scatter(
            x=event_df["date"], y=event_df["cost_first_comfort_risk"],
            name="Cost-First Comfort Risk", mode="lines+markers",
            line=dict(color="#3B82F6"), marker=dict(size=8)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=event_df["date"], y=event_df["occupant_first_comfort_risk"],
            name="Occupant-First Comfort Risk", mode="lines+markers",
            line=dict(color="#10B981"), marker=dict(size=8)
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=event_df["date"], y=event_df["comfort_divergence"],
            name="Comfort Divergence", marker_color="#F59E0B"
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=event_df["date"], y=event_df["cost_divergence"],
            name="Cost Divergence (₹)", marker_color="#8B5CF6", opacity=0.7
        ), row=2, col=1)

        fig.update_layout(
            title="Cost-First vs Occupant-First Plan Divergence",
            height=700, template="plotly_dark", showlegend=True
        )
        fig.write_html(str(CHARTS_DIR / "divergence_analysis.html"))
        safe_write_image(fig, "divergence_analysis.png", width=1200, height=700)
        print("Saved divergence analysis chart (Plotly HTML)")
    if HAS_MATPLOTLIB:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        ax1.plot(event_df["date"], event_df["cost_first_comfort_risk"], "b-o", label="Cost-First")
        ax1.plot(event_df["date"], event_df["occupant_first_comfort_risk"], "g-s", label="Occupant-First")
        ax1.set_title("Comfort Risk: Cost-First vs Occupant-First")
        ax1.set_ylabel("Comfort Risk Score")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)

        x = np.arange(len(event_df))
        ax2.bar(x - 0.2, event_df["comfort_divergence"].values, 0.4, label="Comfort Divergence", color="#F59E0B")
        ax2.bar(x + 0.2, event_df["cost_divergence"].values, 0.4, label="Cost Divergence (₹)", color="#8B5CF6")
        ax2.set_title("Plan Divergence by Day")
        ax2.set_ylabel("Divergence")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(str(CHARTS_DIR / "divergence_analysis.png"), dpi=150)
        plt.close()
        print("Saved divergence analysis chart (matplotlib)")


# %% [markdown]
# ## 4b. Moderate Event Divergence Breakdown
# Shows divergence rates across different moderate curtailment targets (20kW to 65kW).

# %%
def plot_moderate_event_divergence(moderate_df):
    """Visualize plan divergence rate and comfort tradeoff on moderate events."""
    if moderate_df is None or len(moderate_df) == 0:
        print("No moderate event divergence data")
        return

    # Compute divergence rate by target reduction
    target_stats = moderate_df.groupby("target_reduction_kw").agg(
        total_cases=("plans_identical", "count"),
        divergent_cases=("plans_identical", lambda s: int((~s).sum())),
        divergence_rate=("plans_identical", lambda s: round(float((~s).mean()) * 100, 1)),
        avg_office_hvac_cost_first=("cost_first_office_hvac_kw", "mean"),
        avg_office_hvac_occ_first=("occupant_first_office_hvac_kw", "mean"),
    ).reset_index()

    targets = [f"{int(t)} kW" for t in target_stats["target_reduction_kw"]]
    div_rates = target_stats["divergence_rate"].tolist()

    if HAS_PLOTLY:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Divergence Rate by Target Reduction",
                                            "Office HVAC Curtailment Divergence"],
                            horizontal_spacing=0.15)
        fig.add_trace(go.Bar(
            x=targets, y=div_rates,
            text=[f"{r:.1f}%" for r in div_rates], textposition="auto",
            marker_color="#8B5CF6", name="Divergence %"
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=targets, y=target_stats["avg_office_hvac_cost_first"],
            name="Cost-First Office HVAC", marker_color="#EF4444"
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=targets, y=target_stats["avg_office_hvac_occ_first"],
            name="Occupant-First Office HVAC", marker_color="#10B981"
        ), row=1, col=2)
        fig.update_layout(
            title="Moderate Event Divergence Analysis (Cost-First vs Occupant-First)",
            template="plotly_dark", height=450, showlegend=True, barmode="group"
        )
        fig.write_html(str(CHARTS_DIR / "moderate_divergence_breakdown.html"))
        safe_write_image(fig, "moderate_divergence_breakdown.png", width=1100, height=450)
        print("Saved moderate divergence breakdown chart (Plotly HTML)")
    if HAS_MATPLOTLIB:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.bar(targets, div_rates, color="#8B5CF6")
        ax1.set_title("Divergence Rate by Target Reduction")
        ax1.set_ylabel("Divergence %")
        ax1.set_ylim(0, 105)
        for i, v in enumerate(div_rates):
            ax1.text(i, v + 2, f"{v:.1f}%", ha="center")

        x = np.arange(len(targets))
        w = 0.35
        ax2.bar(x - w/2, target_stats["avg_office_hvac_cost_first"], w, label="Cost-First", color="#EF4444")
        ax2.bar(x + w/2, target_stats["avg_office_hvac_occ_first"], w, label="Occupant-First", color="#10B981")
        ax2.set_xticks(x)
        ax2.set_xticklabels(targets)
        ax2.set_title("Office HVAC Curtailment")
        ax2.set_ylabel("Average kW Curtailed")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(str(CHARTS_DIR / "moderate_divergence_breakdown.png"), dpi=150)
        plt.close()
        print("Saved moderate divergence breakdown chart (matplotlib)")


# %% [markdown]
# ## 5. Per-Day Forecast Peak Error Analysis

# %%
def plot_forecast_errors(forecast_df):
    """Visualize per-day forecast peak errors for both models."""
    if forecast_df is None or len(forecast_df) == 0:
        print("No per-day forecast data available")
        return

    if HAS_PLOTLY:
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=["Daily Peak: Actual vs Forecasted",
                                            "Per-Day Peak Error (kW)"],
                            vertical_spacing=0.12)

        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["actual_peak_kw"],
            name="Actual Peak", mode="lines+markers",
            line=dict(color="#EF4444", width=2), marker=dict(size=6)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["hist_predicted_peak_kw"],
            name="Historical Avg Forecast", mode="lines+markers",
            line=dict(color="#3B82F6", dash="dash"), marker=dict(size=4)
        ), row=1, col=1)
        ml_valid = forecast_df.dropna(subset=["ml_predicted_peak_kw"])
        if len(ml_valid):
            fig.add_trace(go.Scatter(
                x=ml_valid["date"], y=ml_valid["ml_predicted_peak_kw"],
                name="RandomForest Forecast", mode="lines+markers",
                line=dict(color="#10B981", dash="dot"), marker=dict(size=4)
            ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=forecast_df["date"], y=forecast_df["hist_peak_error_kw"],
            name="Hist Avg Peak Error", marker_color="#3B82F6", opacity=0.6
        ), row=2, col=1)
        ml_valid = forecast_df.dropna(subset=["ml_peak_error_kw"])
        if len(ml_valid):
            fig.add_trace(go.Bar(
                x=ml_valid["date"], y=ml_valid["ml_peak_error_kw"],
                name="RF Peak Error", marker_color="#10B981", opacity=0.6
            ), row=2, col=1)

        fig.update_layout(
            title="Per-Day Forecast Peak Error Analysis",
            height=700, template="plotly_dark", barmode="group"
        )
        fig.write_html(str(CHARTS_DIR / "forecast_errors.html"))
        safe_write_image(fig, "forecast_errors.png", width=1200, height=700)
        print("Saved forecast error charts (Plotly HTML)")
    if HAS_MATPLOTLIB:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        ax1.plot(forecast_df["date"], forecast_df["actual_peak_kw"], "r-o", label="Actual")
        ax1.plot(forecast_df["date"], forecast_df["hist_predicted_peak_kw"], "b--s", label="Hist Avg")
        ax1.set_title("Daily Peak: Actual vs Forecasted")
        ax1.set_ylabel("Peak (kW)")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)

        ax2.bar(np.arange(len(forecast_df)) - 0.2, forecast_df["hist_peak_error_kw"], 0.4,
                label="Hist Avg Error", color="#3B82F6", alpha=0.6)
        ax2.set_title("Per-Day Peak Error (kW)")
        ax2.set_ylabel("Error (kW)")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(str(CHARTS_DIR / "forecast_errors.png"), dpi=150)
        plt.close()
        print("Saved forecast error charts (matplotlib)")


# %% [markdown]
# ## 6. Equipment Utilization Analysis

# %%
def plot_equipment_utilization(summary):
    """Bar chart of equipment utilization frequency."""
    usage = summary.get("equipment_utilization_count", {})
    if not usage:
        print("No equipment utilization data")
        return

    equip_ids = list(usage.keys())
    counts = [usage[eid] for eid in equip_ids]

    if HAS_PLOTLY:
        fig = go.Figure(go.Bar(
            x=equip_ids, y=counts,
            marker_color=px.colors.qualitative.Set2[:len(equip_ids)],
            text=counts, textposition="auto"
        ))
        fig.update_layout(
            title="Equipment Utilization in DR Plans (30-Day Period)",
            xaxis_title="Equipment ID", yaxis_title="Times Used in Plans",
            template="plotly_dark", height=400
        )
        fig.write_html(str(CHARTS_DIR / "equipment_utilization.html"))
        safe_write_image(fig, "equipment_utilization.png", width=900, height=400)
        print("Saved equipment utilization chart (Plotly HTML)")
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(equip_ids, counts, color=plt.cm.Set2(np.linspace(0, 1, len(equip_ids))))
        ax.set_title("Equipment Utilization in DR Plans (30-Day Period)")
        ax.set_xlabel("Equipment ID")
        ax.set_ylabel("Times Used")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(str(CHARTS_DIR / "equipment_utilization.png"), dpi=150)
        plt.close()
        print("Saved equipment utilization chart (matplotlib)")


# %% [markdown]
# ## 7. Thermal Model Comparison

# %%
def plot_thermal_model_comparison():
    """Compare 1R1C vs 2R2C thermal predictions for a sample zone."""
    from planner.thermal_model import simulate_1r1c, simulate_2r2c, get_thermal_params

    params = get_thermal_params("OFFICE_A", T_outdoor=35.0)
    T_init = 23.0
    reduction_kw = 15.0  # moderate HVAC reduction

    pred_1r1c = simulate_1r1c(T_init, params, reduction_kw, duration_minutes=60, comfort_max=24, safety_limit=27)
    pred_2r2c = simulate_2r2c(T_init, None, params, reduction_kw, duration_minutes=60, comfort_max=24, safety_limit=27)

    # Linear approximation for comparison
    linear_traj = [T_init + reduction_kw * 0.06 * (t / 60) for t in range(61)]

    time_mins = list(range(len(pred_1r1c.T_trajectory)))

    if HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_mins, y=pred_1r1c.T_trajectory,
                                  name="1R1C Model", line=dict(color="#3B82F6", width=2)))
        fig.add_trace(go.Scatter(x=time_mins, y=pred_2r2c.T_trajectory,
                                  name="2R2C Model", line=dict(color="#10B981", width=2)))
        fig.add_trace(go.Scatter(x=list(range(61)), y=linear_traj,
                                  name="Linear Approximation (old)", line=dict(color="#EF4444", dash="dash")))
        fig.add_hline(y=24, line_dash="dot", line_color="#FCD34D",
                      annotation_text="Comfort Max (24°C)")
        fig.add_hline(y=27, line_dash="dash", line_color="#EF4444",
                      annotation_text="Safety Limit (27°C)")
        fig.update_layout(
            title=f"Thermal Model Comparison: OFFICE_A Zone (HVAC reduction = {reduction_kw} kW)",
            xaxis_title="Time (minutes)", yaxis_title="Temperature (°C)",
            template="plotly_dark", height=450
        )
        fig.write_html(str(CHARTS_DIR / "thermal_model_comparison.html"))
        safe_write_image(fig, "thermal_model_comparison.png", width=900, height=450)
        print(f"1R1C: {pred_1r1c.T_initial}°C → {pred_1r1c.T_final}°C (max {pred_1r1c.T_max}°C)")
        print(f"2R2C: {pred_2r2c.T_initial}°C → {pred_2r2c.T_final}°C (max {pred_2r2c.T_max}°C)")
        print(f"Linear: {T_init}°C → {linear_traj[-1]:.1f}°C")
        print("Saved thermal model comparison chart (Plotly HTML)")
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_mins, pred_1r1c.T_trajectory, "b-", label="1R1C Model", linewidth=2)
        ax.plot(time_mins, pred_2r2c.T_trajectory, "g-", label="2R2C Model", linewidth=2)
        ax.plot(range(61), linear_traj, "r--", label="Linear (old)", linewidth=1.5)
        ax.axhline(y=24, color="#FCD34D", linestyle=":", label="Comfort Max")
        ax.axhline(y=27, color="#EF4444", linestyle="--", label="Safety Limit")
        ax.set_title(f"Thermal Model Comparison: OFFICE_A (HVAC reduction = {reduction_kw} kW)")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(CHARTS_DIR / "thermal_model_comparison.png"), dpi=150)
        plt.close()
        print("Saved thermal model comparison chart (matplotlib)")


# %% [markdown]
# ## 8. Summary Statistics Tables

# %%
def print_summary_tables(daily_df, summary):
    """Print formatted summary tables."""
    print("\n" + "=" * 80)
    print("OADRP EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\nDataset: {summary['n_days']} days, {summary['n_observations']} observations")
    print(f"Peak threshold: {summary['peak_threshold_kw']} kW")
    print(f"Peak event days: {summary['n_peak_event_days']}")

    # Event severity distribution
    if "event_severity_distribution" in summary:
        print("\nEvent Severity Distribution:")
        for sev, count in summary["event_severity_distribution"].items():
            print(f"  {sev:10s}: {count} days")

    print("\n--- Peak Reduction Performance ---")
    print(f"{'Metric':<35s} {'Baseline':>10s} {'Cost-First':>12s} {'Occ-First':>12s}")
    print("-" * 72)
    print(f"{'Avg Peak (kW)':<35s} {summary['avg_peak_kw']['baseline']:>10.1f} {summary['avg_peak_kw']['cost_first']:>12.1f} {summary['avg_peak_kw']['occupant_first']:>12.1f}")
    print(f"{'Total Demand Charge (₹)':<35s} {summary['total_demand_charge']['baseline']:>10.0f} {summary['total_demand_charge']['cost_first']:>12.0f} {summary['total_demand_charge']['occupant_first']:>12.0f}")
    print(f"{'Total Energy Cost (₹)':<35s} {summary['total_energy_cost']['baseline']:>10.0f} {summary['total_energy_cost']['cost_first']:>12.0f} {summary['total_energy_cost']['occupant_first']:>12.0f}")
    print(f"{'Avg Comfort Violation (min/day)':<35s} {summary['avg_comfort_violation_min_per_day']['baseline']:>10.1f} {summary['avg_comfort_violation_min_per_day']['cost_first']:>12.1f} {summary['avg_comfort_violation_min_per_day']['occupant_first']:>12.1f}")

    # Divergence analysis
    if "divergence_analysis_by_severity" in summary:
        print("\n--- Divergence Analysis by Severity ---")
        for sev, data in summary["divergence_analysis_by_severity"].items():
            print(f"\n  {sev} ({data['n_days']} days):")
            print(f"    Plans identical: {data['plans_identical_pct']}%")
            print(f"    Avg comfort divergence: {data['avg_comfort_divergence']}")
            print(f"    Avg cost divergence: ₹{data['avg_cost_divergence']}")

    # Forecast errors
    print("\n--- Forecast Error ---")
    hist_errs = summary["forecast_error"]["historical_average"]
    ml_errs = summary["forecast_error"]["random_forest"]
    print(f"{'Model':<25s} {'MAE (kW)':>10s} {'RMSE (kW)':>10s} {'Peak Error (kW)':>15s} {'Peak Error %':>12s}")
    print("-" * 72)
    print(f"{'Historical Average':<25s} {hist_errs['mae']:>10.1f} {hist_errs['rmse']:>10.1f} {hist_errs['peak_error_kw']:>15.1f} {hist_errs['peak_error_pct']:>12.1f}%")
    print(f"{'RandomForest':<25s} {ml_errs['mae']:>10.1f} {ml_errs['rmse']:>10.1f} {ml_errs['peak_error_kw']:>15.1f} {ml_errs['peak_error_pct']:>12.1f}%")

    print(f"\nTotal evaluation runtime: {summary['total_runtime_s']:.1f}s")


# %% [markdown]
# ## 9. Run All Analysis

# %%
def run_notebook():
    """Run all analysis and generate all charts."""
    print("Loading results...")
    load_df, occ_df, daily_df, summary, forecast_df, moderate_df = load_results()

    print("\nGenerating charts...")
    plot_peak_comparison(daily_df, summary)
    plot_severity_distribution(daily_df)
    plot_divergence_analysis(daily_df)
    plot_moderate_event_divergence(moderate_df)
    plot_forecast_errors(forecast_df)
    plot_equipment_utilization(summary)
    plot_thermal_model_comparison()

    print_summary_tables(daily_df, summary)

    print(f"\nAll charts saved to: {CHARTS_DIR}")
    return daily_df, summary


if __name__ == "__main__":
    run_notebook()
