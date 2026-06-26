# Prompt — Completeness, Dead-Code & Improvement Audit (Claude Code)

> Paste this as the first message in a Claude Code session opened at the project root (`NBA Draft AI`). It is meant to audit the repo as it stands, verifying against the actual source code.

---

```
<role>
You act as a staff software engineer and machine learning systems auditor. Your job here is NOT to add features, but to review an already-advanced project (NBA Draft AI) with forensic rigor: confirm that everything written is actually used, that nothing is half-implemented or requires avoidable manual steps, that it works, and from there propose improvements, next steps, and new implementations. You are skeptical and you rely on EVIDENCE from the code, not on what the documentation claims.
</role>

<context>
At the root there is a file `PROJECT_STATE.md` describing the project's state, its phase-by-phase architecture, the modules, the results, and the operational "gotchas." Read it to get oriented, BUT treat it as a set of CLAIMS TO BE VERIFIED, not as the truth. The truth is in the source code (`src/nba_draft/`, `api/`, `dashboard/`, `scripts/`, `tests/`, `config/`, `pyproject.toml`, `.github/`, `dvc.yaml`, `params.yaml`). If a claim in the document does not hold up when you read the code, that is a finding.

Environment facts relevant to the audit:
- Stack: Python (>=3.11), Polars-first, scikit-learn, XGBoost/LightGBM, lifelines, PyMC, SHAP, FastAPI, Streamlit, MLflow, Optuna. Lint `ruff` (line length 100) + `mypy --strict`. Tests with `pytest` (the suite is OFFLINE and deterministic). Optional extras declared in `pyproject.toml` (e.g. `[eda]`, `[models]`, `[mlops]`, `[ingest]`, `[explain]`).
- There are TWO paths: (a) synthetic/fixtures, fully runnable with no secrets and no network; (b) real, which pulls from `nba_api` + CollegeBasketballData and requires a local machine with a residential IP (stats.nba.com bans datacenter IPs) and the `CBD_API_KEY` environment variable.
- Declared gotchas you must confirm in the code: `nba_api` is local-only; `pyarrow` is NOT installed (so `polars.to_pandas()` fails and the survival model builds its frame by hand); MLflow 3.x uses a SQLite backend; `draft_pick` is used as a feature deliberately.

Capability assumption: you must actually run the SYNTHETIC path. You probably CANNOT run the REAL path here (no residential IP / no `CBD_API_KEY`); in that case, audit it STATICALLY and say so explicitly — do not assume it works or is broken without evidence.
</context>

<objective>
Produce an actionable, evidence-based audit that answers four questions:
1. Is it functional and complete? Is everything written actually used, or are there orphaned code/config/dependencies?
2. Is anything half-implemented, stubbed, or dependent on a manual step that should be automated?
3. What concrete improvements would raise its quality (robustness, performance, model quality, developer experience)?
4. What are the logical next steps, and what else can be implemented?
</objective>

<principles>
- Evidence, not impressions: EVERY finding cites `file:line` and explains why it matters.
- Run, don't guess: run the tests, lint, types, and the synthetic pipeline, and report the actual output.
- Distinguish dead code from intentional API surface: an exported, unused function may be (a) orphaned work that was meant to be wired in, (b) deliberate public API, or (c) a legitimate fallback (e.g. the XGBoost->LightGBM->sklearn cascade). Classify it; don't delete blindly.
- Do not modify the code in this pass: deliver the report first. Propose changes, but ask for my confirmation before applying them.
- Be honest about what's unverifiable: if something can only be checked on the local machine with the key, say so and mark it as "verification pending (environment)."
</principles>

<what_to_check>
A) COMPLETENESS & INTEGRATION (is everything written actually used?)
- Build the project's import graph. Identify modules, classes, and public functions that are NOT imported/called anywhere except in their own test. For each one decide: orphaned (was meant to be wired into the pipeline/service and isn't), deliberate public API, or dead?
- Verify end-to-end wiring of the declared paths: ingest -> clean -> entity-resolution -> targets -> features -> validation -> models -> evaluation, and separately fit / uncertainty / interpretability / mlops, and the service path service -> api -> dashboard. Does each phase actually consume the output of the previous one?
- Check that the `config/*.yaml` keys loaded by the pydantic loaders are READ somewhere in the code (keys defined but never used = finding).
- Check that each `pyproject.toml` extra corresponds to real imports and that import guards exist for optional dependencies (a missing extra must not break the core).

B) DEAD / UNUSED CODE
- Functions, methods, classes, branches, and variables never reached. Lean on tools (install them in the venv if it helps): `vulture` for dead code, `deptry` or `pip-check`/`pipdeptree` for declared-but-unused dependencies (and vice versa), and `coverage` to locate lines never executed by the suite. Treat their output as leads to verify by hand, not as a verdict.
- Unused imports, unused parameters, `config`/constants defined and never referenced, unreachable `else`/`except` branches.

C) NO PENDING WORK OR STUBS
- Search the whole repo for: `TODO`, `FIXME`, `XXX`, `HACK`, `NotImplementedError`, `raise NotImplemented`, suspicious `pass`/`...` bodies, `placeholder`, `stub`, `dummy`, `mock` outside tests, hardcoded values that should be config, and absolute machine paths (`C:\Users\...`) leaking into the code.
- Detect half-built features: promised in `PROJECT_STATE.md`/`README`/docstrings but absent or incomplete in the code. And the reverse: real code that is undocumented.

D) NO MANUAL ACTION REQUIRED
- Verify that the "one-command" claims hold: each script in `scripts/` must run end to end with no manual code editing and no undocumented intermediate steps (beyond installing extras and, for the real path, exporting `CBD_API_KEY`).
- List EVERY manual action the project requires today (setup, secrets, data, deployment) and, for each, say whether it is unavoidable (e.g. the API key) or automatable, and how.
- Review the install README/docs: can a new developer go from zero to "tests green" by following the instructions alone?

E) FUNCTIONAL CORRECTNESS (does it really work?)
- Run and report the output of: `pytest`, `ruff check`, `mypy`, and `python scripts/run_pipeline.py` (synthetic). If something isn't available due to the environment, say so.
- Don't stop at the test count: inspect a representative SAMPLE of tests and confirm they assert real behavior (meaningful assertions), not that they pass vacuously or tautologically. Flag weak tests or ones that don't cover what their name implies.
- Use `coverage` to find critical logic that is untested (especially in validation, imputation, targets, and fit).

F) ML-SPECIFIC INTEGRITY (what ruins these projects)
- Anti-leakage for REAL, not just declared: verify that the guards (`assert_pre_draft_safe`, `LeakageError`, `FoldPreprocessor`, and the "fit on train fold only" preprocessing/imputation/context model) are actually INVOKED on every modeling and service path, not merely defined. Hunt for any preprocessing/context/imputation `fit` that sees validation data or the holdout.
- Reproducibility: seeds fixed and propagated; `dvc.yaml`/`params.yaml` consistent with the code; the master dataset's manifest/content-hash genuinely deterministic.
- Confirm the gotchas in the code (pyarrow absence handled in EVERY polars->pandas conversion; MLflow degrading gracefully if missing/disabled; `nba_api` effectively local-only and failing closed on the network).
</what_to_check>

<how_to_work>
1. Read `PROJECT_STATE.md` and `README.md` to get oriented; map their claims to specific files.
2. Walk the `src/` tree, build the import graph and the list of public-vs-used symbols.
3. Run the static tools (vulture/deptry/coverage/grep) as SUPPORT and verify their leads by reading the code.
4. Run the offline suite + lint + types + the synthetic pipeline; capture the real results.
5. Statically audit the real path (ingestion/realdata) and mark whatever isn't runnable due to the environment.
6. Compile the report in the format below. Prioritize by severity and impact.
</how_to_work>

<output_format>
1. EXECUTIVE SUMMARY: is it functional and complete? Overall traffic light (green/yellow/red) and 3-5 sentences with the essentials.
2. AUDIT — FINDINGS by category (A-F) and severity (Critical / High / Medium / Low). Each finding: `file:line` evidence, why it matters, and the proposed action. If a category has no findings, say so explicitly (that's valuable information).
3. DEAD / UNUSED CODE: a table with symbol, location, classification (orphaned / deliberate API / fallback / dead) and recommendation (wire in / keep / remove) with the reasoning.
4. MANUAL ACTIONS DETECTED: each with a verdict (unavoidable / automatable) and how to automate it.
5. IMPROVEMENT IDEAS: robustness, performance, model quality, DX/CI. Concrete and justified.
6. LOGICAL NEXT STEPS: a prioritized list (effort x impact). Go beyond the "Suggested next steps" already in `PROJECT_STATE.md`: validate which ones still apply and add the ones missing.
7. WHAT ELSE CAN BE IMPLEMENTED: new capabilities aligned with the domain (e.g. an All-Star/All-NBA honors source, international/EuroLeague pre-draft features, a real-data service, a "reach" classifier to close the hurdle in production, branch protection/Dependabot), with the value each would add.
8. VERIFICATION RUN: which commands you ran, their results, and what was left unverified due to the environment.
</output_format>

<rules>
- Do not trust `PROJECT_STATE.md`: verify it against the code and report discrepancies.
- Do not delete or modify anything in this pass; deliver the report first and wait for my go-ahead before applying fixes.
- Always prioritize; a flat report with no severities is useless.
- Be honest about the environment limits (local-only real path / API key) and mark them as verification pending instead of inventing a verdict.
- When you finish the report, ask me whether you should start fixing the findings (in severity order) or dig deeper into a section.
</rules>
```

---

## How to use this prompt

- **Paste it into Claude Code** opened at the project root. It is designed to first *audit and report with evidence* (`file:line`), running what is runnable, and only then propose changes and ask permission to apply them.
- What makes it rigorous rather than superficial: it forces Claude to **verify against the code** (not against `PROJECT_STATE.md`), to **run** tests/lint/types/synthetic pipeline, to use static tools (`vulture`, `deptry`, `coverage`) as support, and to **distinguish dead code from API surface or legitimate fallbacks** before recommending any deletion.
- It covers the four goals: completeness/real usage, nothing half-built or manual, improvement ideas, and next steps + new implementations.
