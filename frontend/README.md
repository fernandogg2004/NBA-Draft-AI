# Front Office — Draft Command Center (frontend)

React + TypeScript + Vite + Tailwind frontend for **NBA Draft AI**, implementing the
**Apex Front Office** design system from the Google Stitch "War Room" project. It is a thin
presentation layer over the existing FastAPI service (`api/main.py`) — all model logic stays
in Python.

## Screens

| Route | Screen | Data |
| --- | --- | --- |
| `/board` | Draft Board | `GET /prospects` |
| `/prospect/:id` | Prospect Detail | `GET /prospects` |
| `/explain/:id` | Explainability (SHAP + counterfactual) | `GET /explain/{id}`, `GET /counterfactual/{id}` |
| `/team-fit` | Team Fit & Simulator | `POST /fit` |
| `/compare` | Comparison | `GET /prospects` |

Real model output backs the SHAP attribution, the **counterfactual**, the **synergy sub-scores**,
the **lineup Net-Rating** before→after, the **skill radar** + **archetype** (feature-derived),
**age**, **wingspan**, **peak percentile**, and **projected career value $**.

With a real serving artifact (`NBA_DRAFT_AI_MASTER`, see the root README), the board shows the
**2026 class post-draft**: real **photos**, **drafting team**, **position**, **combine
measurements**, the **actual pick**, and a **steal/reach** chip (model rank vs. where the player was
actually drafted). On the synthetic demo these real-only fields fall back to icons / "not measured".
A ribbon reminds users the board is **pre-draft projections** (no 2026 outcomes yet) and the skill
mapping is exploratory.

## Run it

You need two processes: the API and the Vite dev server.

```bash
# 1) API (from the repo root, in the project venv)
uvicorn api.main:app --reload --port 8000      # serves the synthetic demo board by default

# 2) Frontend (from frontend/)
npm install        # first time only
npm run dev        # http://localhost:5173
```

The dev server proxies `/api/*` → `http://127.0.0.1:8000` (see `vite.config.ts`), so the browser
makes same-origin requests and there's no CORS to configure in development.

To serve a **real** board, point the API at a persisted serving dir before launching uvicorn:

```bash
export NBA_DRAFT_AI_MASTER=/path/to/serving   # written by scripts/run_real_pipeline.py
```

## Configuration

| Env var | Where | Purpose |
| --- | --- | --- |
| `VITE_API_TARGET` | dev server | Proxy target for `/api` (default `http://127.0.0.1:8000`) |
| `VITE_API_BASE` | build | API base URL when not using the proxy (e.g. a deployed origin) |
| `NBA_DRAFT_AI_CORS` | API | Comma-separated allowed origins (default localhost:5173) |

## Build

```bash
npm run build      # tsc -b && vite build  → dist/
npm run preview    # serve the production build locally
```

## Design system

`tailwind.config.js` carries the Apex tokens verbatim from Stitch (colors, Inter + JetBrains
Mono type scale, radius/spacing). The 5-tier outcome scale and the brand orange/blue accents
live there too. Tonal-layer and hairline helpers are in `src/index.css`.
