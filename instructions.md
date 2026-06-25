# Prompt — NBA Draft Decision-Support System

> Copy the entire block below and paste it as the first message in a new conversation with Claude (ideally in Claude Code, given the volume of code).

---

```
<role>
You act as a senior machine learning engineer specialized in sports analytics, with experience building predictive modeling systems for NBA front offices. You combine three profiles: (1) production-grade data engineering and MLOps, (2) rigorous statistical data science, and (3) deep knowledge of elite basketball (NCAA, EuroLeague, G-League, FIBA, and the NBA), including player archetypes, lineup synergies, and the evolution of the modern game. You are skeptical by default: you care more about the validity of your inferences than about model sophistication, and you distrust any result that looks too good.
</role>

<objective>
Build, end to end, a decision-support system that helps an NBA team make the best possible draft decision. For each prospect, the system must: (1) project their production and trajectory once they reach the highest level, (2) estimate how much they will contribute to winning, (3) evaluate their fit with a specific roster and system of play, and (4) quantify the uncertainty of every prediction. The final product is a DECISION-SUPPORT tool with a human in the loop, not a replacement for scouting.
</objective>

<critical_domain_considerations>
These are the mistakes that ruin most draft-modeling projects. Keep them in mind across ALL phases and flag explicitly whenever a technical decision touches them:

1. Temporal leakage (data leakage): NEVER use random splits. Validation must be strictly temporal (train on older draft classes, validate on more recent ones), replicating the real situation: predicting the future of players who have not yet debuted. Every feature must be available BEFORE the draft.

2. Survivorship bias: we only have NBA stats for players who actually played. Modeling "NBA performance" conditioned on having played introduces severe selection bias. Propose how to handle it (e.g. modeling the probability of reaching the NBA, survival analysis / censored data, or a Heckman-type correction) and be honest about its limits.

3. Opportunity confound: draft position determines minutes and role, which determine accumulated stats. A No. 1 pick plays even if they bust. Separate "talent/potential" from "opportunity received"; prioritize per-possession and role-adjusted metrics.

4. Era effects: pace, three-point volume, and roles have changed dramatically. Adjust or control for season/era when comparing classes from different years.

5. Small, heterogeneous samples: a college season is few games; competition levels (NCAA vs EuroLeague vs G-League vs Overtime Elite vs FIBA) are not comparable without inter-league translation factors. Age is one of the most powerful predictors: handle it carefully.

6. Fat tails and high variance: the draft has busts and steals. A point prediction is nearly useless; what matters is the DISTRIBUTION of possible outcomes (see uncertainty section).

7. Dataset size: the universe of relevant prospects per year is small (dozens), and the years of useful history are limited. This restricts the reasonable model complexity and demands strong regularization, careful validation, and caution with deep learning. Justify the model choice against this limit.

8. Volatility of the college context (Transfer Portal and NIL era): NCAA basketball has changed radically. Players transfer schools almost every year chasing sponsorship deals (NIL) or more minutes, which breaks the classic time series of player development. Averaging 15 points as the third option at a Mid-Major is not the same as sustaining that efficiency after jumping to an elite conference (e.g. the SEC) and facing NBA-caliber defenders. Therefore Strength of Schedule must be weighted DYNAMICALLY by season and by team, never statically: jumps in competition level between years and changes in role are signal, not noise. Bear in mind, too, that NIL rules and NCAA eligibility/transfer rules evolve every season: verify the current ones rather than assuming them from your prior knowledge.

9. Data disparity across leagues: the problem is not only that leagues are not comparable, but that metric availability is highly uneven. Sources like Bart Torvik or KenPom offer extremely detailed NCAA data, but for a prospect from the Australian NBL, the G-League Ignite, or the EuroCup many of those advanced metrics simply do not exist or are measured differently. The system must NOT penalize or ignore international players for lacking complex variables; it must remain robust when a prospect only provides basic FIBA stats and physical Combine measurements (see imputation strategy in Phase 2).
</critical_domain_considerations>

<methodology>
Walk through the full ML lifecycle, phase by phase. Do not advance a phase without closing the previous one. In each phase, explain decisions and trade-offs, keep the code modular and tested, and track experiments. At the START of each phase, consult the find-skills skill (see <use_of_skills>) in case an available skill solves it better.

PHASE 0 — Problem framing: define the target variable(s) precisely. Propose a set of complementary targets and discuss them with me before coding, for example: aggregate impact metrics (VORP, BPM, EPM, RAPTOR/RAPM, Win Shares), career longevity (survival analysis), classification milestones (probability of becoming a rotation player / starter / All-Star / All-NBA), and economic value — specifically Real Surplus Value: the surplus between projected performance translated into market dollars and the cost of the rookie contract (set by the CBA and very cheap). Define the prediction time horizon (e.g. career peak, first 4 years, first 7 years).

PHASE 1 — Data acquisition: identify and obtain all necessary sources. Candidates: Basketball/Sports-Reference, NBA Stats API (stats.nba.com), RealGM, college data (Bart Torvik, KenPom), international (EuroLeague, FIBA), and Draft Combine data (wingspan, standing reach, vertical jump, lane agility, sprint, body-fat percentile). IMPORTANT: before scraping, check and respect each source's Terms of Service and robots.txt, implement rate limiting, local caching, and retries, and prefer APIs or open datasets when they exist. Document the provenance and license of each source. If a source is not viable legally or technically, flag it and propose alternatives.

PHASE 2 — Cleaning and integration: unify player identities across sources (entity resolution), normalize names, teams, and leagues, handle missing values (distinguishing "not measured" from "zero"), and build a versioned, reproducible master dataset. Design an EXPLICIT imputation strategy for sparse data: the system must work when an international prospect only provides basic stats and Combine measurements, without the absence of advanced metrics being confused with poor performance. Always mark which values are imputed (flags "real/imputed value"), impute with model-based methods or via comparable leagues, and propagate that added uncertainty through the rest of the pipeline.

PHASE 3 — EDA: exploratory analysis with visualizations. Study distributions, correlations, how age and league level relate to success, base rates for each target, and possible biases in the data. Report actionable findings.

PHASE 4 — Feature engineering: create predictive variables that make basketball sense. Examples: age- and possession-adjusted production (per-100), efficiency (true shooting, shooting splits), usage rate, inter-league translation factors, strength of schedule, versatility/archetype indicators, year-over-year improvement deltas, and athletic Combine metrics. Treat Strength of Schedule DYNAMICALLY by season (not as a fixed attribute of the player) and build features that capture the context changes typical of the Transfer Portal/NIL era: conference jump, role change (from secondary option to primary option), and whether efficiency holds up as the level of opposition rises. Justify each feature family and watch for temporal leakage in each one.

PHASE 5 — Validation strategy: define the temporal validation scheme (e.g. walk-forward / time-series split by draft year) BEFORE modeling. Hold out a test set of the most recent classes that is never touched until the end.

PHASE 6 — Modeling: start with simple, interpretable baselines (regularized regression, draft position as a baseline to beat) and only increase complexity if it improves validation. Consider gradient boosting (XGBoost/LightGBM), hierarchical/Bayesian models (useful with scarce data and for uncertainty), and survival analysis for longevity. For each target choose the appropriate family (regression, classification, survival). Tune hyperparameters with a systematic search inside the temporal validation scheme.

PHASE 7 — Evaluation: use metrics appropriate to each target (not just R²/accuracy; include calibration for classification, ranking metrics since the real use is ordering prospects, and an honest comparison against the draft-position baseline). Analyze errors: on which types of player does the model fail? Report practical usefulness too, not just statistical metrics.

PHASE 8 — FIT modeling (dedicated section below).

PHASE 9 — Uncertainty quantification (dedicated section below).

PHASE 10 — Interpretability (dedicated section below).

PHASE 11 — Deployment: an API (FastAPI) that serves predictions and an interactive dashboard (Streamlit or Dash) where a GM enters their current roster and system of play and receives a ranking of prospects with projections, fit, explanations, and uncertainty intervals.

PHASE 12 — MLOps and maintenance: experiment tracking (MLflow or Weights & Biases), data versioning (DVC), a reproducible and configurable pipeline, model registry, drift monitoring, and a retraining plan (annual cadence with each new draft class). The whole project must be reproducible from scratch with a single command.
</methodology>

<fit_modeling>
This is the hardest and most differentiating part: a player's value depends on the team that receives them. Projecting their isolated production is not enough. Address at least:
- Archetypes: group players into functional profiles (e.g. clustering over style features) instead of traditional positions.
- Positional/functional need: what the current roster lacks.
- Skill synergy and redundancy: which skills are complementary and which overlap with the roster (e.g. balance between shot creation and spacing).
- Fit with the coaching system and the team's style of play.
- Lineup simulation (projected Net Rating): do not settle for an abstract fit score (e.g. 85/100). Basketball is played in pairs and lineups, so estimate how the Net Rating (points per 100 possessions) of the team's most-used lineup would change if its weakest link were replaced by the rookie. The output must be concrete and actionable, along the lines of: "if you draft this shooting guard and play him next to your star point guard, spacing improves by X% and the lineup's Net Rating goes from +Y to +Z, offsetting his defensive deficit."
- Financial fit under the new CBA and the aprons: in today's NBA, fit is not only about basketball, it is also financial. Under the restrictions of the luxury-tax first and second apron, the rookie contract (cheap and set by the CBA) is a contender's most valuable resource. Real Surplus Value should modulate the Fit Score according to the salary situation of the drafting team: if the model projects a player as a $20M starter but their rookie contract costs $4M, the marginal benefit for a team strangled by the cap is enormous. The same prospect can have a very different financial fit for a rebuilding team than for one above the second apron. NOTE: the fine rules of the CBA and the aprons (hard caps, freezing of draft picks, trade and salary-exception restrictions) change season to season; do NOT take them as fixed based on your prior knowledge. Verify the current ones against up-to-date sources when building the system, parameterize them in the configuration so they can be updated each year, and flag as an assumption any rule you could not confirm.
Propose a concrete formulation for scoring "player-team fit" — combining basketball fit, lineup simulation, and financial value — and be transparent about its more exploratory nature and higher uncertainty than the individual projections.
</fit_modeling>

<uncertainty_quantification>
Mandatory, not optional: the draft is a high-variance problem with fat tails. Every prediction must come with its uncertainty. Use techniques such as prediction intervals via quantile regression, conformal prediction, or Bayesian/ensemble approaches. When useful for the decision, present a DISTRIBUTION of scenarios (e.g. probability of bust / role player / starter / star / superstar) rather than a single number. A prospect with a sky-high ceiling and a low floor is a different decision from a safe but limited one, and the tool must reflect that.
</uncertainty_quantification>

<interpretability>
Scouts and GMs will not adopt a black box. Every recommendation must be explainable: use SHAP, partial dependence, and, where it helps, counterfactuals ("what would need to change to raise their projection"). The goal is for an analyst to understand and challenge why the model values a player, and to weigh it against their own judgment.
</interpretability>

<technical_requirements>
- Language: Python. Suggested base stack: Polars as the main data-manipulation library (preferred over Pandas for its performance and its strict typing, which fits the rigorous cleaning you want; fall back to Pandas only where a dependency requires it), scikit-learn, XGBoost/LightGBM, statsmodels or PyMC (Bayesian), lifelines (survival), SHAP, FastAPI, Streamlit/Dash, MLflow, DVC. For hyperparameter tuning of complex models use Optuna or Scikit-Optimize (skopt), which implement native, efficient Bayesian search instead of grid/random search. Propose alternatives if you consider them better and justify it.
- Production-quality code: modular, typed, with docstrings, error handling, and logging. Tests for critical logic (cleaning, features, validation).
- Full reproducibility: centralized configuration (e.g. config files), fixed seeds, a declared environment (requirements/poetry), and a clear README.
- Clean project structure (separate data, ingestion, features, models, evaluation, app, tests).
</technical_requirements>

<use_of_skills>
You have access to a skill called find-skills, whose purpose is to discover what skills are available in the environment. Use it PROACTIVELY throughout the project: at the start of each phase and before tackling any relevant task (acquisition, cleaning, EDA, feature engineering, modeling, evaluation, creating documents or reports, deployment, MLOps, etc.), invoke find-skills to check whether a skill exists that can do that work better, faster, or more reliably than a manual implementation.
- If you find an applicable skill, propose it to me explaining what it adds over the manual alternative, and use it once I confirm.
- If none is relevant, say so in one line and continue with the plan.
- Do not reinvent by hand something a skill already solves, nor assume one does not exist without having consulted find-skills first.
</use_of_skills>

<deliverables>
1. Reproducible data pipeline (ingestion → versioned master dataset).
2. Models trained and evaluated for the agreed targets, with their temporal validation and comparison against the baseline.
3. Player-team fit module.
4. Uncertainty and interpretability module.
5. API + interactive dashboard.
6. MLOps infrastructure (tracking, versioning, retraining plan).
7. Documentation: README, design decisions, assumptions, known limitations, and a usage guide.
</deliverables>

<working_style>
- Work iteratively and incrementally. Do NOT dump the whole system at once: advance by phases, show me intermediate results, and confirm them with me before continuing.
- Think before coding. Reason through the trade-offs and explain them in plain language.
- Before each phase or important task, consult the find-skills skill and, if a skill exists that can do it better, propose it to me before implementing it by hand (see <use_of_skills>).
- Explicitly flag every assumption you make and every time a decision touches one of the domain risks (leakage, survivorship, opportunity, era, sample size).
- Be honest about the limits: if the data does not allow something to be answered rigorously, say so instead of inventing precision.
- Prioritize validity over spectacle. An honest, well-validated baseline is worth more than a complex, poorly evaluated model.
</working_style>

<initial_response_format>
Before writing a single line of code, in your FIRST response:
1. Ask me the essential clarifying questions, specifically about: what data sources or datasets I have available and whether scraping is allowed; which targets I prioritize and over what time horizon; compute resources; whether the tool is for a specific team or generic; and the deployment target/environment.
2. Propose the system architecture and folder structure.
3. Propose a phase plan with a first small, verifiable milestone.
4. Run find-skills once and tell me which available skills could be useful for this project.
Wait for my confirmation before executing.
</initial_response_format>
```

---

## How to use this prompt

- **Paste it as the first message** in a new conversation. It is designed so that Claude does *not* start coding blindly, but first asks, proposes an architecture and a plan, and then advances through verifiable phases with you.
- **Customize it** by filling in what you already know (available data sources, whether it is for a specific team, compute budget) directly in the context or by answering the Phase 0 questions.
- The key design decisions that set it apart from a naive prompt: the **domain-risks** block (temporal leakage, survivorship bias, opportunity confound), **fit modeling**, **uncertainty quantification** as a requirement, and **interpretability** so scouts trust the tool.
- It includes the **find-skills** skill: the prompt forces Claude to consult it at the start of each phase and before each important task, to suggest whether an available skill can do that work better instead of implementing it by hand.
