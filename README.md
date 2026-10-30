# HAMV-CC Lab

A lightweight local Plotly Dash research workbench based on the supplied revised manuscript, **Reliability-Aware Fusion for Incomplete Multi-View Clustering: Bounded Gating and Evidence-Grounded Evaluation**.

## Start on Windows

1. Install **Python 3.12 (64-bit)**, including the Python launcher.
2. Extract this ZIP completely.
3. Double-click `start_windows.bat`. The first start creates a local virtual environment and installs dependencies; it needs internet access.
4. Open **http://127.0.0.1:8050**. Keep the terminal open. Press Ctrl+C to stop.

The existing Python 3.8 environment used for the old manuscript experiment is too old for this application's pinned packages. Use a separate Python 3.12 environment; do not overwrite the experiment environment.

Manual setup (Windows):

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

macOS / Linux (Python 3.11+; tested with 3.12):

```sh
sh start.sh
```

Or create a virtual environment, install `requirements.txt`, and run `python app.py`. Change a busy port with `python app.py --port 8051`. The server binds to localhost with debugging disabled. This is a local research application, not an authenticated public deployment.

## Four workspaces

- **Results:** filter recorded Linux/Windows runs by condition, ARI/NMI, and method; inspect tables; export selected CSV data. Method error bars are standard deviations of five split means, each over eight seeds. The primary paired-effect graph always shows ΔARI, even when the method metric is NMI. Its saved intervals and Holm-adjusted p values are conditional diagnostics, not calibrated population-level inference.
- **Gate lab:** change structural/visual/text reliability, mask available views, and compare a minimum-share policy with manual strength. Implements the manuscript's stable masked softmax exactly. All-missing inputs are explicitly unassignable. Bounds are analytical; no trained reliability predictor is implied.
- **Documents:** inspect saved baseline clusters for 199 records, K=2/3, PCA plots, internal metrics, and feature ablations. Run a separate CPU K-means experiment on selected bundled features or a custom CSV. New outputs never overwrite manuscript results.
- **Evidence:** see evidence boundaries, the manuscript's environment comparison, original source archive paths, and SHA-256 checksums.

Plotly's camera button exports SVG plots. The new-run export is a ZIP with assignments and configuration metadata. Uploads and new results are held in page memory, never automatically written to disk; switching tabs or refreshing resets that workspace. Download results you want to retain before leaving the tab. The saved result files are read-only in the application.

## Custom feature CSV

UTF-8 CSV, maximum 3 MB, 4–2,000 rows, 64 columns. Example:

```csv
document_id,ink_ratio,edge_density,mean_intensity
doc-001,0.15,0.20,0.85
doc-002,0.16,0.22,0.82
doc-003,0.75,0.81,0.25
doc-004,0.79,0.78,0.22
```

`document_id` must be unique; row identifiers are generated if it is absent. Select meaningful numeric features. Obvious identifiers, byte counts, dimensions, and columns named `label`, `target`, or `cluster` are excluded from automatic candidates. Other numeric metadata may still require manual deselection. Missing/infinite selected values are rejected; constant columns are removed with an explicit count. No silent imputation is performed.

Fresh run: StandardScaler → KMeans (Lloyd, n_init=10, chosen seed). Full-SVD PCA is used only to visualize the fitted partition. Silhouette and Davies–Bouldin are computed in standardized feature space. Above 500 rows, silhouette uses a seeded 500-row subset including at least one item from every cluster; the UI reports this count. No semantic accuracy is computed without labels. For new datasets, choose K and features on development data before confirmatory evaluation.

## Efficiency and dependencies

Six direct dependencies; no PyTorch, TensorFlow, OCR service, Redis, database, remote models, external CSS, or paid API. Dash serves its JS and Plotly locally. Saved data uses bounded process-level caches; pages render on demand. User uploads stay in per-page memory, not shared mutable server globals. K-means executes only on Run, with one numerical thread, bounded input dimensions, and bounded silhouette work. Plotting and filters do not rerun experiments. The interface uses plain Unicode equations and requires no CDN; it works offline after dependency installation.

The ZIP contains application source and compact original result files, not a virtual environment. Dependency installation takes much more disk space than this ZIP. Exact historical experiment scores are preserved, not recalculated using today's dependencies.

## Scientific scope

The gate uses `u_m = kappa * (2*r_m - 1)` and exponentials shifted by the maximum available logit. Missing weights equal zero and available weights sum to one. The floor rule is `kappa = 0.5 * log((1/rho - 1)/(M-1))`; one view gets weight one. The fixed-mask L1 perturbation bound is `min(2, 2*kappa*delta)`.

This app implements a gate calculator and exploratory feature clustering. It **does not train the complete contrastive HAMV-CC architecture**, extract OCR or image embeddings, or claim semantic document-clustering performance. The supplied image experiment is incomplete. The saved public results include negative clean-condition evidence and stronger spectral-comparator results; they are not hidden or pooled across environments.

## Sources

- Supplied `HAMVCC_TETC_Revised.tex`: method equations, interpretation, and evidence limitations.
- Supplied manuscript ZIP: Linux `results_v2.json` and `analysis_v2.json`.
- `results2.zip`: Windows public-digit outputs and feature summaries.
- `HAMVCC_full199_results.zip`: document feature matrix, saved assignments, metrics, and PCA.
- `data/provenance.json`: exact source archive/member, byte count, and checksum for every bundled source artifact.
- Dash usage reference: https://dash.plotly.com/minimal-app

## Verification

```sh
python -m unittest discover -s tests -v
```

Checks cover gate masking, all-missing rejection, floor/sensitivity bounds, source-file hashes, split-level effects, cross-environment discrepancy counts, feature validation, fresh K-means, callbacks, and HTTP routes. See `VALIDATION.md` for the execution record. `tests/browser_smoke.py` is optional browser validation and requires Playwright plus Chromium, which are deliberately excluded from runtime dependencies.

## Enhanced diagnostics (new)

The **05 Diagnostics** tab adds aligned Linux–Windows numerical discrepancy analysis for ARI/NMI, tolerance and condition filters, per-method discrepancy ranking, downloadable paired audit, a synthetic orthogonal-coordinate invariance sandbox, and an analytic bounded-gate Jacobian heatmap. Recorded evidence is immutable. The synthetic sandbox is explicitly not a rerun of the paper or a validation of document semantics. To run: `pip install -r requirements.txt && python app.py`.

## New: Experiment studio (tab 06)
Upload a UTF-8 CSV (up to 3 MB; 4–2,000 rows) with two to six groups of numerical columns using `view__feature` naming, e.g. `visual__0,visual__1,text__0,text__1`. Optional `available_visual` and `available_text` binary columns specify availability; nonfinite feature entries imply an unavailable view. Optional `label` is used **only** for final ARI/NMI scoring. Duplicate document IDs are rejected. All-missing rows are excluded and reported, not imputed. The app supports three synthetic corruption families, train-only proxy regression, bounded-gate concatenation, masked concatenation, aligned sum fusion, a 15-run ablation grid, an optional PyTorch encoder experiment, and downloadable JSON manifests. Runs are exploratory and separate from immutable reference results.

**Scientific scope:** The optional neural module is a small entropy-balanced encoder demonstration, not the manuscript's full contrastive + Jensen–Shannon consistency + balance training objective. Reliability training uses synthetic feature preservation, not verified semantic correctness. The dataset upload, corruption, and regression experiments are new implementations; do not interpret their results as manuscript reproductions. Inputs stay in browser session storage, and computations run on the local Dash server. Avoid uploading confidential data to a remotely hosted instance.

## V3 complete neural research module

The Experiment Studio's **Train complete HAMV-CC neural model** action now trains
view-specific 2-layer encoders, bounded reliability-conditioned residual maps,
shared-coordinate bias-free projections, masked softmax gating, a shared cluster
head, symmetric two-augmentation NT-Xent, reliability-weighted Jensen-Shannon
consistency, and entropy-based cluster balancing. The loss is
`lambda_instance * NT-Xent + lambda_consistency * weighted_JS + lambda_balance * (mean_entropy - entropy_of_mean)`.

A deterministic train/validation/test partition is generated for each run.
Feature normalization and K-means centroid fitting use **training data only**;
validation selects a neural checkpoint without labels. The test split is not
scored unless the **Run held-out TEST once** option is selected. Note that the
app's synthetic corruption/reliability-proxy experiment is exploratory; it is
not the frozen public-text manuscript protocol or a proof of semantic validity.

The model saves a `.pt` checkpoint in `checkpoints/` with weights, split IDs,
normalization statistics, k-means centers, configuration and data SHA-256.
Checkpoints are local artifacts and may contain sensitive learned parameters.
Never load a checkpoint from an untrusted source using unrestricted pickle.
To customize training outside Dash, import `NeuralConfig` and `train_neural`
from `neural.py`. The model uses CPU by default. Labels are used only for
ARI/NMI evaluation, not for training or checkpoint selection. No repeated
independent-test tuning should be performed.

The implementation is a research approximation of the manuscript's mathematical
architecture, not a numerical reproduction of its fixed 20 Newsgroups runs.

## 07 — Document clustering (new)

Upload a ZIP of image files (PNG/JPG/TIFF/BMP/WebP), UTF-8 text files, or text-based PDFs. The archive is capped at 45 MB compressed, 180 MB expanded, 1,500 files, and 12 MB per item. SHA-256 exact duplicates are excluded. Text PDFs are parsed with `pypdf`; scanned PDFs need an external OCR conversion first. Image OCR is optional and requires the Tesseract executable installed on your computer, plus `pytesseract`.

Set a range of candidate K values and press **Find optimal K and cluster documents**. The app reports the silhouette curve, inertia elbow, Davies–Bouldin diagnostic in the JSON, per-cluster counts, and a sortable document table with filename, hash, source type, OCR status, text preview, view availability, reliability proxies, and fusion weights. Export CSV assignments and JSON run metadata.

**Scientific limitations:** This is an exploratory image-structure/appearance and TF-IDF text feature pipeline using transparent heuristic reliability proxies, **not** a pretrained semantic visual model, a calibrated reliability regressor, or a full neural HAMV-CC replication. All uploaded documents are used for K selection and clustering; the silhouette is an in-sample descriptive metric. No independent test evaluation or semantic label validation is performed. The manuscript's fail-closed private test protocol remains separate. Do not report these exploratory scores as reproduced manuscript findings.


### Tab rendering fix (V4.1)
The earlier app inserted tab content using a server callback. That meant inputs
for other tab callbacks were not present in the initial layout, which could
leave the interface blank or cause callback dependency errors. V4.1 embeds
each tab's content directly in `dcc.Tab(children=...)`, ensuring components
are available when Dash initializes. Clear browser cache / hard-refresh after
restarting the app.

## ZIP upload troubleshooting (V4.4)

In tab 07, the **local path** field reports whether the Dash server can find the ZIP. Paste the complete path and press **Enter** to commit it, then select **Load ZIP from local path**. This option reads files from the *server's* filesystem, not a remote browser's filesystem. Use it only on a trusted local server. For example:

```cmd
python check_zip.py "C:\Users\YourName\Documents\funds.zip"
```

This preflight tool shows archive validity, supported document types and extraction limits. The browser upload remains available for smaller archives. The ZIP is not required to contain only documents; unsupported files are ignored. A dataset of FUNSD-style image/JSON pairs is not automatically a semantic clustering benchmark: JSON annotations are not consumed by this proxy pipeline.
