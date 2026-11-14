"""
Data validation stage.

Applies the rules in rules.py to users -> items -> interactions (in that order,
because interaction referential integrity depends on the validated user/item
sets) and writes:

  data/validated/<table>.csv          rows with no error
  data/errors/<table>_errors.csv      quarantined rows + error_reason
  data/errors/<table>_warnings.csv    kept rows that raised warnings
  reports/validation_report.json|md   counts per rule, pass rates
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.utils import get_logger, load_config, p, save_table, write_json
from src.validation import rules as R

log = get_logger("validation")


def apply_rules(df: pd.DataFrame, rules: list[R.Rule]):
    df = df.reset_index(drop=True)
    err_reasons = pd.Series([[] for _ in range(len(df))])
    warn_reasons = pd.Series([[] for _ in range(len(df))])
    summary = []
    for rule in rules:
        mask = rule.check(df).fillna(False).astype(bool)
        target = err_reasons if rule.severity == "error" else warn_reasons
        for i in mask[mask].index:
            target[i].append(rule.name)
        summary.append({"rule": rule.name, "severity": rule.severity,
                        "description": rule.description, "failed_rows": int(mask.sum())})
    is_err = err_reasons.map(bool)
    errors = df[is_err].assign(error_reason=err_reasons[is_err].map(";".join))
    warns = df[~is_err & warn_reasons.map(bool)].assign(
        warning_reason=warn_reasons[~is_err & warn_reasons.map(bool)].map(";".join))
    return df[~is_err].copy(), errors, warns, summary


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    v = cfg["validation"]
    src = p(cfg, "raw") / "ingested"
    out_valid, out_err, reports = p(cfg, "validated"), p(cfg, "errors"), p(cfg, "reports")
    read = lambda t: pd.read_csv(src / f"{t}.csv", dtype=str, keep_default_na=False,  # noqa: E731
                                 na_values=[""])

    report = {"run_at": datetime.now().isoformat(timespec="seconds"), "tables": {}}
    valid = {}
    plan = [("users", lambda: R.user_rules(v)),
            ("items", lambda: R.item_rules(v)),
            ("interactions", lambda: R.interaction_rules(
                v, set(valid["users"].user_id), set(valid["items"].item_id)))]
    for table, make_rules in plan:
        df = read(table)
        ok, errors, warns, summary = apply_rules(df, make_rules())
        valid[table] = ok
        save_table(ok, out_valid, table, cfg)
        errors.to_csv(out_err / f"{table}_errors.csv", index=False)
        warns.to_csv(out_err / f"{table}_warnings.csv", index=False)
        report["tables"][table] = {
            "input_rows": int(len(df)), "valid_rows": int(len(ok)),
            "error_rows": int(len(errors)), "warning_rows": int(len(warns)),
            "pass_rate": round(len(ok) / max(1, len(df)), 4), "rules": summary,
        }
        log.info("%-12s input=%d valid=%d errors=%d warnings=%d",
                 table, len(df), len(ok), len(errors), len(warns))

    write_json(report, reports / "validation_report.json")
    (reports / "validation_report.md").write_text(_to_markdown(report), encoding="utf-8")
    return report


def _to_markdown(rep: dict) -> str:
    lines = ["# Data Validation Report", "", f"Run at: `{rep['run_at']}`", "",
             "Severity **error** = record quarantined to `data/errors/<table>_errors.csv`. "
             "Severity **warning** = record kept, issue fixed in the preparation stage "
             "and listed in `data/errors/<table>_warnings.csv`.", "",
             "## Summary", "",
             "| Table | Input rows | Valid rows | Quarantined | With warnings | Pass rate |",
             "|---|---:|---:|---:|---:|---:|"]
    for t, s in rep["tables"].items():
        lines.append(f"| {t} | {s['input_rows']:,} | {s['valid_rows']:,} | {s['error_rows']:,} "
                     f"| {s['warning_rows']:,} | {s['pass_rate']:.2%} |")
    for t, s in rep["tables"].items():
        lines += ["", f"## {t}", "", "| Rule | Severity | Description | Failed rows |",
                  "|---|---|---|---:|"]
        for r in s["rules"]:
            lines.append(f"| `{r['rule']}` | {r['severity']} | {r['description']} | {r['failed_rows']:,} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run()
