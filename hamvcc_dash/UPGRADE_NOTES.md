# HAMV-CC Streamlit — Manuscript-aligned upgrade

## New interface features
- Native ZIP upload **and** multi-file document selection.
- Silhouette-based K recommendation with elbow and Davies–Bouldin diagnostics.
- Interactive 2D PCA visualization of the **full-dimensional** fused clustering result.
- Per-cluster indicative TF-IDF keywords, with no claim of validated semantic labels.
- Document inspector with availability, reliability proxies, SHA-256, cluster, and OCR text.
- Eight-seed partition-stability audit (pairwise ARI); the seed-pair values are dependent.
- CSV/JSON export and archive hash manifest.

## Scientific scope
The uploaded-document workflow is a transparent **proxy-feature exploratory analysis**. It is not a trained neural HAMV-CC model and does not reproduce manuscript test-set metrics. Availability is binary, reliability is continuous, and all-missing documents are rejected. Gated block concatenation avoids adding independently defined view coordinate systems. The maximum silhouette K is descriptive and must not be interpreted as a semantic taxonomy.

## Further recommended work
1. Integrate checkpoint-loaded trained neural encoders, calibrated reliability estimators, and a strict feature-schema validator.
2. Add a frozen development/test manifest and explicit test-unlock workflow, with independent labels when available.
3. Add corruption-family challenge sets and matched availability-only, fixed-prior, shuffled, inverted, and random controls.
4. Add raw-image robustness tests and OCR quality audits; currently the existing saved-feature diagnostics are not raw-image tests.
5. Add CPU/GPU resource monitoring, cancellable jobs, resumable preprocessing caches, and user access controls before public deployment.
