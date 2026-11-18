# PathSnap

PathSnap reconstructs directed road routes from GPS observations. The evaluator
selects one of the ten loaded trips, sends it to the prediction server and views
the chosen arcs and evaluation metrics. The UI runs independently from the API.

Start with the [teammate handoff](CONTEXT.md) for setup, demo steps, and project status.

## Run on two laptops

On Laptop 1, create a Python 3.11+ environment and install the backend
dependencies. Start the API using the directory directly containing the ten
trip folders. Each trip may have a server-side .route file for scoring.

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python -m api.main --trips "C:\path\to\evaluator_test" --host 0.0.0.0 --port 8000

On Laptop 2, open the same repository and point the evaluator UI at Laptop 1's
LAN IP. The UI server uses Python's standard library.

    python -m ui.app --api http://192.168.1.20:8000 --port 8501

Replace the example IP with Laptop 1's actual address. Both laptops need to be
on the same network, and Laptop 1 must accept incoming TCP on the chosen port.
Open http://127.0.0.1:8501 on Laptop 2. The in-page map works without tile-server
access. Its overlays represent raw GPS observations and predicted directed arcs.

## Demo and evidence

1. Rehearse with student_test first. Use the reserved evaluator directory only
   for the final session. Start the API and confirm that it lists ten trip IDs.
2. From Laptop 2, pick a trip in the UI, select Reconstruct route, and check
   the overlay, metrics, and Laptop 1 terminal request log.
3. Capture screenshots of the working UI and the receiving laptop's terminal.
   A concise real response can also be printed from Laptop 2 in PowerShell:

       $api = 'http://192.168.1.20:8000'
       $trip = (Invoke-RestMethod "$api/trips")[0]
       $result = Invoke-RestMethod "$api/predict/$trip" -Method Post
       [pscustomobject]@{ trip_id = $result.trip_id; n_points = $result.n_points; metrics = $result.metrics } | ConvertTo-Json -Depth 4

4. With the UI up, run the ten-trip HTTP smoke check:

       python -m scripts.validate --ui http://127.0.0.1:8501

The API contract is GET /trips and POST /predict/{trip_id}. Responses expose
GPS, predicted arcs and aggregate metrics. The scoring service reads a
reference separately after inference. Missing references return unavailable
metric values. Inference time in the UI excludes scoring and HTTP latency.
The service caches geometry, the candidate index, and the road graph for
two recently used trips.

GET /trips and POST /predict are implemented in FastAPI for the demo.
Use --reference-http only for local development when FastAPI is unavailable.
The supplied local evidence uses a direct service rehearsal, because local
HTTP requests were blocked and FastAPI dependencies could not be installed.
The browser preview used an intercepted real service response. It is not a
two-laptop API screenshot.

## Reproduce the experiments

Keep working files under the ignored data directory. Replace RAW with the
directory containing train and student_test. Commands below are run from the
repo root. CPU is sufficient. Set OPENBLAS_NUM_THREADS=1 for comparable timing.

    python -m pip install -r requirements-experiments.txt
    python experiments/p1/p1.py --data "RAW" --out data/processed/p1
    python -m experiments.p2.p2 --p1 data/processed/p1 --raw "RAW" --out data/processed/p2 --trials 10
    python -m experiments.p2.plots --root data/processed/p2
    python -m experiments.p2.fit_diagnostics --p1 data/processed/p1 --raw "RAW" --root data/processed/p2
    python -m experiments.p3.p3 --raw "RAW" --p1 data/processed/p1 --p2 data/processed/p2/results/scores --out data/processed/p3
    python -m experiments.p3.maps --raw "RAW" --results data/processed/p3 --out maps
    python -m experiments.p3.preview --raw "RAW" --results data/processed/p3 --out maps
    python report/generate.py --out PathSnap_Technical_Report.pdf

Install the optional experiments requirements for Folium, PyArrow or Optuna.
The recorded results use Gaussian-process Bayesian tuning, gzip CSV and
Leaflet. Installing Optuna changes the Bayesian sampler. Install Folium and
rerun the map-generation line to produce the required Folium versions before
final submission. Interactive report maps and the comparison PNG are under
maps. Reference layers in these report maps use the student-test split only.

The checked-in results contain summaries, per-model/fold diagnostics and
tuning logs. The reproduction commands regenerate per-candidate tables,
scores, per-point references and decoded points. Keep the raw student and
reserved datasets out of the public repo.

## Repository contents

- api/ and ui/ provide the two deployment entry points.
- src/, configs/, and models/ provide the shared map matching pipeline.
- experiments/ has reproducible P1-P3 experiment code.
- results/ contains the recorded CSV/JSON results and model figures.
- report/ has the consolidated report generator and UI preview.

Review the technical report with the team, regenerate the final Folium maps,
capture the two-laptop evidence, and publish the repository from the team's
GitHub account.
