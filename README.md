# Manufacturing Line Analysis: Bottleneck Identification & Process-Parameter Sensitivity Study

A discrete-event simulation of a 4-stage vehicle-component manufacturing line 
(Machining → Sub-Assembly → Final Assembly → Inspection), built to:
1. Identify the throughput-limiting stage and quantify the gain from a targeted intervention
2. Predict line throughput and defect rate from process parameters and rank their influence

## Key Results
- Identified Sub-Assembly as the bottleneck (84.7% utilization, confirmed by upstream blocking)
- A 30% repair-time reduction + buffer resizing yielded a 4.79% throughput gain
- A Random Forest model predicting throughput from process parameters achieved R² = 0.72;
  repair time and failure frequency at the bottleneck explained 81% of the variance

## How to run
pip install numpy pandas matplotlib scikit-learn

python bottleneck_simulation.py      # Part A: baseline vs improved scenario
python make_chart.py                 # generates bottleneck_chart.png

python sensitivity_study.py          # Part B: 150-config parameter sweep + RF model
python make_sensitivity_chart.py     # generates sensitivity_chart.png

## Full write-up
See report.md for methodology, results, and validity notes.
