# HAMV-CC Streamlit UI refresh

Five workspaces retained. Added Manrope/DM Sans typography (system-font fallbacks), improved hierarchy, dark research sidebar, clear upload panel, consistent metrics, cards, spacing, accessible contrast, and responsive layout. No change to research computation or data files. A small-sample guard now explains why silhouette K selection cannot run with fewer than three documents.

Run: `python -m streamlit run streamlit_app.py`.

Fonts load from Google Fonts if internet access is available; otherwise system fonts are used. Streamlit UI styling uses selectors that may need adjustment with future Streamlit versions.
