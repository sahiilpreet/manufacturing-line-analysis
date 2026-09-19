"""
Process-Parameter Sensitivity Study
====================================

Extends the manufacturing-line simulation into a parameter-sensitivity
analysis: many line configurations are simulated, each with different
process parameters (cycle-time variability, buffer sizes, MTBF/MTTR at the
bottleneck stage, inspection reject rate). For each configuration, the
resulting 30-day throughput and overall defect rate are recorded.

A regression model (Random Forest) is then fit to PREDICT throughput and
defect rate FROM the process parameters, and the parameters are ranked by
feature importance -- the same "predict an outcome from process settings,
then rank which settings matter most" pattern used broadly in process
engineering to prioritize where to focus process control effort.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from bottleneck_simulation import StageConfig, ProductionLineSimulation, baseline_line


def sample_configuration(rng: np.random.Generator) -> dict:
    """Randomly sample one process-parameter configuration within realistic ranges."""
    return {
        "subassy_cv":        rng.uniform(0.10, 0.35),   # cycle-time variability at bottleneck stage
        "subassy_mtbf":      rng.uniform(400, 900),      # mean time between failures (min)
        "subassy_mttr":      rng.uniform(40, 120),        # mean time to repair (min)
        "subassy_buffer":    rng.integers(2, 9),           # output buffer size at bottleneck
        "machining_buffer":  rng.integers(3, 12),           # buffer feeding the bottleneck
        "inspection_reject": rng.uniform(0.00, 0.05),        # final-inspection reject rate
    }


def build_line_from_params(params: dict) -> list:
    stages = baseline_line()
    for s in stages:
        if s.name == "Sub-Assembly":
            s.cycle_time_cv = params["subassy_cv"]
            s.mtbf = params["subassy_mtbf"]
            s.mttr = params["subassy_mttr"]
            s.buffer_capacity = int(params["subassy_buffer"])
        if s.name == "Machining":
            s.buffer_capacity = int(params["machining_buffer"])
        if s.name == "Inspection":
            s.scrap_rate = params["inspection_reject"]
    return stages


def run_configuration(params: dict, duration_minutes: float, seed: int) -> dict:
    stages = build_line_from_params(params)
    sim = ProductionLineSimulation(stages, dt=1.0, seed=seed)
    sim.run(duration_minutes)
    report = sim.stage_report(duration_minutes)

    entered = report["Items Completed"].iloc[0]  # items that started at Machining
    final_good = report["Items Good"].iloc[-1]    # good units that exited Inspection

    # True defect rate = items actually scrapped at any stage, divided by
    # items that entered the line. Computed from each stage's own
    # completed-vs-good counts (i.e. actual scrap events), NOT from
    # (entered - final_good): that naive difference also counts items still
    # in-flight in buffers/queues at the end of the simulation window, which
    # are not defective, just not finished yet -- using it would overstate
    # the defect rate substantially on a finite-horizon run.
    total_scrapped = (report["Items Completed"] - report["Items Good"]).sum()
    defect_rate = total_scrapped / entered if entered > 0 else 0.0

    result = dict(params)
    result["throughput"] = final_good
    result["defect_rate"] = defect_rate
    return result


def run_sensitivity_study(n_configs: int, duration_minutes: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_configs):
        params = sample_configuration(rng)
        result = run_configuration(params, duration_minutes, seed=1000 + i)
        rows.append(result)
    return pd.DataFrame(rows)


def fit_and_rank(df: pd.DataFrame, target: str, feature_cols: list):
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return model, mae, r2, importances


if __name__ == "__main__":
    DURATION = 30 * 24 * 60   # 30 days
    N_CONFIGS = 150

    print("Running parameter sweep across", N_CONFIGS, "configurations...")
    df = run_sensitivity_study(N_CONFIGS, DURATION, seed=7)
    df.to_csv("/home/claude/sim/sensitivity_data.csv", index=False)
    print(df.describe().to_string())

    feature_cols = ["subassy_cv", "subassy_mtbf", "subassy_mttr",
                     "subassy_buffer", "machining_buffer", "inspection_reject"]

    print("\n" + "=" * 70)
    print("MODEL: Predicting THROUGHPUT from process parameters")
    print("=" * 70)
    model_t, mae_t, r2_t, imp_t = fit_and_rank(df, "throughput", feature_cols)
    print(f"Test MAE: {mae_t:.2f} units | Test R^2: {r2_t:.3f}")
    print("\nFeature importance ranking (throughput):")
    print(imp_t.to_string())

    print("\n" + "=" * 70)
    print("MODEL: Predicting DEFECT RATE from process parameters")
    print("=" * 70)
    model_d, mae_d, r2_d, imp_d = fit_and_rank(df, "defect_rate", feature_cols)
    print(f"Test MAE: {mae_d:.4f} | Test R^2: {r2_d:.3f}")
    print("\nFeature importance ranking (defect rate):")
    print(imp_d.to_string())

    with open("/home/claude/sim/sensitivity_summary.txt", "w") as f:
        f.write("THROUGHPUT MODEL\n")
        f.write(f"Test MAE: {mae_t:.2f} units | Test R^2: {r2_t:.3f}\n")
        f.write("Feature importance:\n" + imp_t.to_string() + "\n\n")
        f.write("DEFECT RATE MODEL\n")
        f.write(f"Test MAE: {mae_d:.4f} | Test R^2: {r2_d:.3f}\n")
        f.write("Feature importance:\n" + imp_d.to_string() + "\n")

    imp_t.to_csv("/home/claude/sim/throughput_importance.csv")
    imp_d.to_csv("/home/claude/sim/defect_importance.csv")
