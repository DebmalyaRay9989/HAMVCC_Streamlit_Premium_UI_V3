# Validation record

Executed 26 October 2026, Linux, Python 3.12.14. Runtime dependencies are pinned in requirements.txt. The Windows launcher is supplied but has not been executed on Windows in this environment.

## Automated numerical / application checks

Five unittest cases passed, covering multiple configurations:

- Exact zero missing-view weights, sum-to-one normalization, all-missing rejection, binary-mask validation, finite scores, and stable evaluation through κ=1000.
- Minimum-share rule at ρ=0.1 and 100 randomized fixed-mask perturbation checks.
- SHA-256 verification of every source data file; unique run keys and 160 condition/split/seed rows per environment.
- Recomputed five split-level primary effects against both saved analyses. Verified exactly 6,806 differing aligned ARI/NMI values above 1e−10.
- Fresh K-means on the 199 bundled records reproduced silhouette 0.2622665950041013; validation rejects empty/constant feature selections and invalid K. CSV parser checks include valid input and wrong extension.
- Result callbacks exercised for both environments, all four conditions, both metrics, and an empty method selection. Saved document callbacks exercised for four methods and both K settings. HTTP routes and all four tab-render callback responses returned 200.

## Browser checks

Playwright 1.51.0 / Chromium 134 (test tools only, not runtime requirements):

- Four tabs render; plots and metric cards load.
- Gate reports κ=0.752 at the three-view floor rule and rejects all-missing input.
- Saved-data CSV download and fresh-assignment ZIP download succeed.
- Bundled-data clustering renders and reports silhouette 0.2623.
- Custom CSV upload loads four records and runs clustering.
- Results page fits a 390 px viewport without horizontal document overflow.
- No JavaScript page errors during the final test.
- Desktop results-page screenshot was inspected for layout and chart readability.

## Scope

These checks validate the application and consistency with supplied artifacts. They do not constitute a new scientific replication, validation of the complete contrastive architecture, or semantic document-clustering assessment. No public deployment or Windows runtime test was performed. Dash DataTable is pinned with Dash 3.4.0; a future Dash major upgrade may require migration.
