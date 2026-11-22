# Pocket CRM: mortgage PD model

This is my small end-to-end probability of default project using Freddie Mac single-family loan-level data. It starts with a WoE logistic scorecard and then tests an XGBoost challenger. I kept the same loan splits and the same later cohort for both models. Their inputs differ: the scorecard uses 7 selected fields and XGBoost uses all 13 origination fields.

## The question

For loans originated in **2005 Q1**, take the ones still active and either current (`00`) or one month late (`01`) in **December 2006**. Did they reach **180+ days past due** during **2007**? The candidate predictors are fields recorded **at origination**. December payment status decides which loans enter the cohort; it is saved for the audit but is never a model input.

For the out-of-time check, I use **2006 Q1 originations**: **December 2007** eligibility and **2008** outcomes during the housing crisis. These are different loans from the 2005 Q1 development group. An observed 180+ month counts as default. A nondefault needs a complete observed year or an accepted payoff. I leave gaps and other unresolved exits without a target rather than turning them into zeros.

## Run it

Put the six notebooks and `cohort.py` in one project directory, then run the notebooks in number order. All six notebooks use the same data-folder rule: use `POCKET_CRM_DATA` if it is set; otherwise use `E:\CRM Projects\Data` if that folder exists; otherwise use `./data` relative to the notebook working directory. The same notebook files can be used locally or from a Git clone. The raw Freddie Mac files are:

- `orig_2005Q1.txt` and `perf_2005Q1.txt`
- `orig_2006Q1.txt` and `perf_2006Q1.txt`

The files are not included here. Each origination/performance pair may be together in the selected data directory or one of its subfolders. On a computer with a different data location, set `POCKET_CRM_DATA` to the directory that contains the files (or their subfolders) before starting Jupyter. For example, in PowerShell:

```powershell
$env:POCKET_CRM_DATA = 'D:\freddie-data'
```

Or in a Unix shell:

```bash
export POCKET_CRM_DATA=/path/to/freddie-data
```

If `POCKET_CRM_DATA` is unset, the notebooks use the E: folder when it exists and `./data` otherwise. Intermediate Parquet files, model objects, tables and diagnostics go into the selected data directory and its `pocket1_model/` subdirectory. You will need Python with pandas, NumPy, pyarrow, scikit-learn, optbinning, xgboost and joblib. The code expects the Freddie Mac Release 47 column layout used in Notebook 1.

| Notebook | Job |
| --- | --- |
| `01_cohort.ipynb` | Make the 2007 development cohort and audit missing outcomes. |
| `02_woe_feature_screen.ipynb` | Split the loans, fit WoE bins on training, and screen features using IV and correlation. |
| `03_logistic_scorecard.ipynb` | Fit the logistic scorecard; report AUC, KS, Gini, PD error and score points. |
| `04_2008_oot_check.ipynb` | Score the new 2006 Q1 loans in 2008 with the frozen scorecard and inspect bin PSI. |
| `05_xgboost_pd_challenger.ipynb` | Fit XGBoost on raw origination fields, with five training folds, a small parameter comparison and a calibration check. |
| `06_model_comparison.ipynb` | Compare both models on 2008, including calibration, paired bootstrap differences, segments, score PSI and TreeSHAP. |

`cohort.py` holds the cohort and outcome rules for Notebooks 1 and 4, the numeric field list, and the tree model's feature preparation. The 2007 and 2008 cohorts follow the same code.

## A few choices I made

The default rate is low, so I stratify the development split and keep the bins broad. I report both ranking and PD accuracy: a good AUC does not by itself mean the probabilities are close to the observed rate. The logistic model gives me a simple scorecard baseline. XGBoost is evaluated on the same loans but uses more input fields; its encoder is fitted separately inside each cross-validation fold. The validation split is used for XGBoost tuning and calibration. The held-out development test and the 2008 cohort are used to assess the fitted models.

DTI 999 gets a separate bin in the scorecard and is treated as missing by XGBoost. I keep the bin tables, split assignments, loan-level PDs and drift tables so I can trace the comparison back to the loans.

## AI assistance

I used AI as a coding assistant for parts of the notebooks and this README. I am responsible for the modeling choices and how I interpret the results.
