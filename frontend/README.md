# AgentTriage React frontend

Separate React/Vite application for the post-training inference and human-in-the-loop interface. It does not change or depend on the existing Python project files.

## Run

```powershell
npm.cmd install
npm.cmd run dev
```

Open the local address printed by Vite (normally `http://127.0.0.1:5173`).

## Backend integration contract

Set `VITE_API_BASE_URL` (for example in a local `.env`) to point to the API. The client sends:

```json
POST /api/v1/predictions
{
  "source": "GitHub", "project": "vscode", "title": "...",
  "description": "...", "component": "", "priority": ""
}
```

It expects `prediction_id`, `model_version`, `model_prediction`, `model_confidence`, `prediction_summary`, `shap_values`, and an `audit` array. Low-confidence cases are submitted as separate `POST /api/v1/reviews` records; model prediction and confidence remain immutable in the request.

Without `VITE_API_BASE_URL`, the app uses visibly disclosed preview data for UI testing only.
