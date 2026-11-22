"""Cohort rules and feature preparation shared by the notebooks."""

import pandas as pd


ORIG = {
    0: "fico", 7: "occupancy_status", 8: "cltv", 9: "dti",
    10: "original_upb", 11: "ltv", 12: "original_interest_rate",
    13: "channel", 17: "property_type", 19: "loan_id",
    20: "loan_purpose", 21: "original_loan_term", 23: "seller_name",
    29: "interest_only_indicator",
}
PERF = {0: "loan_id", 1: "month", 3: "dpd", 8: "zbc"}
NUMERIC = ["fico", "cltv", "dti", "original_upb", "ltv",
           "original_interest_rate", "original_loan_term"]


def clean_origination(loans):
    loans[NUMERIC] = loans[NUMERIC].apply(pd.to_numeric, errors="coerce")
    for f, code in {"fico": 9999, "cltv": 999, "ltv": 999}.items():
        loans[f] = loans[f].mask(loans[f].eq(code))
    # DTI 999 stays for the WoE special bin.
    return loans


def features_from(loans, features):
    """Prepare the saved origination fields for the XGBoost encoder."""
    x = loans[features].copy()
    numeric = [f for f in NUMERIC if f in features]
    categories = [f for f in features if f not in NUMERIC]
    for f in numeric:
        x[f] = pd.to_numeric(x[f], errors="coerce").astype("float32")
    if "dti" in x:
        x["dti"] = x["dti"].mask(x["dti"].eq(999))
    for f in categories:
        x[f] = x[f].astype("string").fillna("__MISSING__").astype(str)
    return x


def build_cohort(data_dir, quarter, snapshot_month, outcome_year):
    """Return eligible loans, their origination fields, and observed outcome."""
    target = f"default_180_plus_{outcome_year}"
    year_end = f"{outcome_year}12"
    orig_file = data_dir / f"orig_{quarter}.txt"
    perf_file = data_dir / f"perf_{quarter}.txt"

    parts = []
    for chunk in pd.read_csv(perf_file, sep="|", header=None, usecols=list(PERF),
                             dtype="string", chunksize=400_000):
        rows = chunk.loc[chunk[1].between(snapshot_month, year_end)]
        if not rows.empty:
            parts.append(rows)
    perf = pd.concat(parts, ignore_index=True).rename(columns=PERF)
    perf = perf.apply(lambda s: s.str.strip())
    perf["zbc"] = perf["zbc"].replace("", pd.NA)
    if perf.duplicated(["loan_id", "month"]).any():
        raise ValueError("Repeated loan-month rows")

    snap = perf.loc[(perf["month"] == snapshot_month)
                    & perf["dpd"].isin(["00", "01"])
                    & perf["zbc"].isna(), ["loan_id", "dpd"]]
    snap = snap.rename(columns={"dpd": "snapshot_dpd"})

    orig = pd.read_csv(orig_file, sep="|", header=None, usecols=list(ORIG),
                       dtype="string").rename(columns=ORIG)
    orig = orig.apply(lambda s: s.str.strip())
    loans = snap.merge(orig, on="loan_id", how="left", validate="one_to_one",
                       indicator=True)
    if loans["_merge"].ne("both").any():
        raise ValueError("Snapshot loan missing origination row")
    loans = clean_origination(loans.drop(columns="_merge"))

    out = perf.loc[perf["month"].between(f"{outcome_year}01", year_end)
                   & perf["loan_id"].isin(snap["loan_id"])].copy()
    out["default"] = out["dpd"].isin([f"{i:02d}" for i in range(6, 100)])
    out["good"] = out["dpd"].isin([f"{i:02d}" for i in range(6)])
    out["credit_exit"] = (out["zbc"].isin(["02", "03", "09", "15", "16", "96"])
                          | out["dpd"].eq("RA"))

    exit_row = (out.loc[out["zbc"].notna()].sort_values("month")
                .drop_duplicates("loan_id").set_index("loan_id")[["month", "zbc"]])
    out["stop"] = out["loan_id"].map(exit_row["month"]).fillna(year_end)
    coverage = out.loc[out["month"] <= out["stop"]].groupby("loan_id").agg(
        months=("month", "nunique"), all_good=("good", "all"))
    events = out.groupby("loan_id").agg(default=("default", "any"),
                                         credit_exit=("credit_exit", "any"))
    y = (snap[["loan_id"]].merge(events, left_on="loan_id", right_index=True, how="left")
         .merge(coverage, left_on="loan_id", right_index=True, how="left")
         .merge(exit_row, left_on="loan_id", right_index=True, how="left"))
    expected = y["month"].fillna(year_end).str[-2:].astype(int)
    verified_zero = (y["months"].fillna(0).eq(expected) & y["all_good"].fillna(False)
                     & (y["zbc"].isna() | y["zbc"].eq("01"))
                     & ~y["credit_exit"].fillna(False))
    y[target] = pd.Series(pd.NA, index=y.index, dtype="Int8")
    y.loc[y["default"].fillna(False), target] = 1
    y.loc[verified_zero & ~y["default"].fillna(False), target] = 0

    return loans.merge(y[["loan_id", target]], on="loan_id", how="left",
                       validate="one_to_one")
