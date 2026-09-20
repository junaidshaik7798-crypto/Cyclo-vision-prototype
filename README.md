# CYCLO-VISION

AI-powered tropical cyclone intelligence from satellite imagery. CYCLO-VISION accepts cyclone images, detects and classifies storms, estimates wind speed and pressure, assesses risk, produces an explainability heatmap, and generates a prototype forecast track with an uncertainty cone.

> Research prototype only. Results are estimates and must not be used as operational weather warnings.

## Features

- Upload JPG, JPEG, PNG, or TIFF satellite images up to 10 MB.
- Analyze bundled demo cyclone samples.
- Optional storm-centre input as latitude and longitude.
- Cyclone detection and IMD-style intensity classification.
- Estimated wind speed in knots and pressure in hPa.
- Confidence score, risk level, and risk factors.
- Explainability heatmap overlay for the analyzed image.
- Forecast track points from +6 h to +48 h with an uncertainty cone.
- Historical reference and calibration information from IBTrACS/reference data.
- Health, data-source, live IBTrACS, and reference-dataset views.
- Optional PostgreSQL persistence for analysis history.

## Project Structure

```text
backend/              FastAPI API, analysis services, ML pipeline, and tests
frontend/src/         React and Vite dashboard
frontend/standalone/  No-build input page and separate results page
data/demo/            Bundled sample satellite images
data/reference/       Reference dataset manifest and raw data location
models/               Trained model output location
database/             Database schema and initialization scripts
scripts/              Demo-data, database, and verification utilities
```

## Requirements

- Python 3.11 or newer
- Node.js 18 or newer and npm, for the React frontend
- Docker Desktop, optional, for PostgreSQL
- PyTorch is optional. Without model weights, the API runs in `demo` mode.

## Quick Start

### 1. Install backend dependencies

From the repository root in PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the backend with the environment's Python directly or adjust the local execution policy.

### 2. Start the API

Keep the terminal in `backend/`:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation is available at:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/redoc
- http://127.0.0.1:8000/api/health

### 3. Start the React frontend

Open a second terminal from the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173. Vite proxies `/api` and `/data` requests to the backend at port 8000.

Useful frontend commands:

```powershell
npm run build    # TypeScript check and production build
npm run lint     # TypeScript check without emitting files
npm run preview  # Serve the production build locally
```

## Separate Results Page

The standalone frontend provides the complete upload-to-results workflow without a frontend build step.

With the backend running, open a second terminal from the repository root:

```powershell
python -m http.server 5173 --directory frontend/standalone
```

Open http://127.0.0.1:5173. Select a bundled sample or upload an image, optionally enter `latitude, longitude`, and choose **Run Cyclone Analysis**. The analysis opens in `results.html` and displays the analyzed image, heatmap, confidence, wind, pressure, risk, forecast track, uncertainty cone, and model details.

The standalone page uses `http://127.0.0.1:8000` by default. The API can be overridden with either:

```text
http://127.0.0.1:5173/?api=http://127.0.0.1:8000
```

or browser local storage key `cyclo_api`.

## Database (Optional)

The API can run without PostgreSQL, but history persistence requires the database.

Start PostgreSQL with Docker Compose:

```powershell
docker compose up -d db
```

The default connection is:

```text
postgresql://cyclo:cyclo_pass@localhost:5432/cyclo_vision
```

To use another connection, create a `.env` file in the repository root:

```env
DATABASE_URL=postgresql://cyclo:cyclo_pass@localhost:5432/cyclo_vision
```

The application attempts database initialization at startup. If PostgreSQL is unavailable, analysis still runs with fallback behavior and reports the database as offline.

## Model Mode

The default configuration is:

```env
ML_MODE=demo
```

Demo mode uses the transparent feature-based inference path. To use trained weights, place the model at `models/cyclo_cnn.pt`, install the CPU PyTorch packages, and configure the root `.env` file:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```env
ML_MODE=model
MODEL_PATH=models/cyclo_cnn.pt
```

Restart the API after changing `.env` values. The health endpoint reports the active ML mode and whether model weights are available.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | API, database, ML, and IBTrACS status |
| `POST` | `/api/analyze` | Analyze an uploaded image using multipart form data |
| `GET` | `/api/analyze/samples` | List bundled demo samples |
| `POST` | `/api/analyze/demo` | Analyze the first bundled sample |
| `POST` | `/api/analyze/demo/{sample_id}` | Analyze a selected bundled sample |
| `GET` | `/api/analyses` | List persisted analysis history |
| `GET` | `/api/analyses/{analysis_id}` | Get one persisted analysis |
| `GET` | `/api/cyclone-track/{analysis_id}` | Get a persisted forecast track |
| `GET` | `/api/data-sources` | List configured data sources |
| `GET` | `/api/reference-dataset` | Get the curated reference dataset |

For uploads, `/api/analyze` expects a multipart field named `file`. The optional `source` field defaults to `upload`, and `region` may contain a JSON pair such as `[17.5, 88.3]`.

## Analysis Response

An analysis response includes fields such as:

```json
{
	"cyclone_detected": true,
	"classification": "Severe Cyclonic Storm",
	"confidence": 0.86,
	"estimated_wind_speed_knots": 72.0,
	"estimated_pressure_hpa": 968.0,
	"intensity_category": "Severe Cyclonic Storm",
	"risk_level": "High",
	"risk_factors": [],
	"explainability": { "heatmap_png_b64": "..." },
	"track": [],
	"center": { "lat": 17.5, "lon": 88.3 },
	"inference_mode": "demo"
}
```

The exact values depend on the image, active inference mode, calibration data, and available reference data. Confidence is not a guarantee of forecast accuracy. Single-image infrared intensity estimates have meaningful uncertainty, so results should be reviewed by a qualified meteorologist.

## Tests

From `backend/`:

```powershell
python -m pytest -q
```

The tests run the FastAPI application in-process and do not require a live server. The frontend can be checked with:

```powershell
cd ..\frontend
npm run build
```

## Troubleshooting

- **API unreachable:** confirm the backend is running on port 8000 and that the frontend is using the same API URL.
- **Database offline:** start `docker compose up -d db`, or continue without history persistence.
- **Model unavailable:** confirm `ML_MODE=model`, `MODEL_PATH`, and the model file; otherwise use the default demo mode.
- **No demo images:** confirm files exist under `data/demo/` and restart the API.
- **CORS errors:** serve the frontend through port 5173 instead of opening the HTML file directly with `file://`.