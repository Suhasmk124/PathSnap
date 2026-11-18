# PathSnap teammate handoff

## Where to start

PathSnap matches each noisy GPS observation to a directed road arc and
reconstructs a route. The classifier and decoder configuration are included.
Running the demo does not require retraining. Clone this repository onto both
laptops. Get the original hackathon dataset from the team or organizers and
extract it on the API laptop. The repository keeps the raw data offline.

Point the API at the directory that directly contains the numbered trip
folders. Each trip folder contains matching .track, .nodes, and .arcs files.
The API also reads its .route file for scoring after inference. The folder for
a practice run is student_test. Use the ten reserved trips for the final demo.

## 1. Run on one Windows laptop

From the repository root, open PowerShell and run:

    python -m venv .venv
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
    & '.\.venv\Scripts\python.exe' -m api.main --trips 'C:\path\to\student_test' --host 127.0.0.1 --port 8001

Replace the input path with your extracted student_test directory. Keep the
first terminal open. In a second PowerShell window at the repo root:

    & '.\.venv\Scripts\python.exe' -m ui.app --api http://127.0.0.1:8001 --port 8502

Open http://127.0.0.1:8502. Select 00000076 and reconstruct the route. For
that trip the expected Top-1 is about 76.18% for the classifier and 89.03% with
Viterbi. The displayed runtime varies by laptop and whether the trip is cached.

## 2. Switch to two laptops

On Laptop 1 use the same backend command with --host 0.0.0.0. Find its LAN
IPv4 address with ipconfig. Both laptops need to be on the same network.
On Laptop 2 open the repo in PowerShell and run the UI with its own Python.
The UI only needs the standard library.

    python -m ui.app --api http://LAPTOP1_IP:8001 --port 8502

Replace LAPTOP1_IP with the real address. Open http://127.0.0.1:8502 on
Laptop 2. To diagnose a connection issue, run this on Laptop 2:

    Invoke-RestMethod 'http://LAPTOP1_IP:8001/trips'

A practice run should list all ten student-test IDs. If the UI starts but
cannot reach the API, confirm the LAN IP, the backend's host and port, and
the Windows firewall rule for the demo network. The live ten-trip smoke
check can be run from the repo root using the environment from step 1:

    & '.\.venv\Scripts\python.exe' -m scripts.validate --ui http://127.0.0.1:8502

## What has been built

| Stage | Implementation and evidence |
| --- | --- |
| P1 | Loaded 80 training and 10 student-test trips. Built local metre geometry, data diagnostics, monotone GPS-to-reference alignment, and a spatial index. Chose 25 m and up to 30 candidates using training trips. See experiments/p1 and results/p1. |
| P2 | Compared five classifiers with four trip-grouped folds. Ran 10 Bayesian trials and selected a tuned shallow MLP. The selected model, scaler, and feature list are in models and configs. See experiments/p2 and results/p2. |
| P3 | Built directed-road HMM/Viterbi decoding with gap, stop and unreachable-transition handling. It raised student-test Top-1 from 80.27% to 86.95% and continuity from 77.27% to 89.08% using common time boundaries. See src/hmm.py, experiments/p3 and results/p3. |
| P4 | Integrated candidate generation, model scoring, decoding, server-side reference scoring, and the evaluator UI. The direct service matched the offline metrics on all ten student-test trips. A browser check used a recorded service response. See api, ui and results/p4. |

The API exposes GET /trips and POST /predict/{trip_id}. Responses contain GPS
points, predicted arcs and metrics. Raw arc IDs are local to each trip, so join
candidate or prediction rows using both trip ID and observation/arc identifiers.
The API gives evaluator reference data only to the server-side scoring step.

## Before the final evaluation

- Run the real two-laptop FastAPI rehearsal and capture the evaluator UI and
  receiving laptop's API terminal. The recorded local tests covered direct
  inference and the browser separately.
- Switch the API input to the reserved evaluator trip folders when provided.
  Keep those data and labels off GitHub.
- Use the README reproduction commands to regenerate the required Folium
  report maps. The package includes Leaflet versions and a Folium-capable
  generator. The consolidated PDF can be regenerated with report/generate.py.
