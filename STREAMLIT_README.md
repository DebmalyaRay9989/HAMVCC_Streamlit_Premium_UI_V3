# HAMV-CC Streamlit research workbench

This is a Streamlit replacement for the Dash user interface. All seven workspaces are available in `streamlit_app.py`. The underlying scientific Python modules and bundled data are preserved. The former Dash `app.py` and `direct_upload.py` have been removed from this distribution to prevent accidental launching of the wrong framework.

## Windows launch

Double-click `run_streamlit.bat`, or run:

```cmd
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the URL printed by Streamlit (typically http://localhost:8501). To upload, use **07 · Archive upload & clustering → Browse files**. Select the ZIP from Windows Explorer, then click **Import and inspect archive**, followed by **Find best K and cluster documents**. No local path input is needed.

## Archive processing

ZIP archives up to 500 MB can be sent by the Streamlit uploader, subject to Streamlit's server upload limit (configured in `.streamlit/config.toml`). The underlying ingestion module enforces 1,500 entries, 180 MB uncompressed, and 12 MB per file. Supported files: PNG/JPG/JPEG/TIF/TIFF/BMP/WebP/TXT/MD/PDF. PDFs are text-extraction only; image OCR uses pytesseract and a separately installed Tesseract executable. Unsupported files are ignored; exact duplicates are excluded by SHA-256.

The exploratory clustering uses lightweight structural/visual image features and text TF-IDF, a bounded reliability proxy gate, and block concatenation. Silhouette maximization selects K; the elbow curve is diagnostic. This is **not** a trained neural reproduction of the manuscript's published results. Saved manuscript benchmarks remain distinct from user-uploaded results.

## Research modules

`core.py`, `diagnostics.py`, `research.py`, `neural.py`, and `document_pipeline.py` remain importable. The Experiment Studio and Evidence & Provenance interface pages have been removed as requested. Research modules remain available in Python for compatibility.

## Troubleshooting

Run from the directory containing `streamlit_app.py`. For OCR, install Tesseract separately and ensure `tesseract.exe` is on PATH. If a ZIP is rejected, inspect its compressed and uncompressed sizes and file types. For large archives, OCR may take time. Do not deploy an unrestricted uploader publicly without authentication and resource limits.
