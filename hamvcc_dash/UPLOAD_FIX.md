# V4.5 ZIP path input fix
The local ZIP path input previously used `debounce=True`. Dash did not send a freshly pasted value to the server until Enter or focus loss, so clicking Load could send an empty State. The field now uses `debounce=False`; the callback receives the current path without requiring Enter.

1. Run `python app.py` in this folder.
2. Go to tab 07. Paste a complete path such as `C:\\Users\\...\\funds.zip`.
3. Wait until the validation line displays `File found`, then click **Load ZIP from local path**.
4. Click **Find optimal K and cluster documents**.

If the validation line says `file not found`, use Windows Explorer > Copy as path and paste the result, including the `.zip` extension. The ZIP must be accessible to the Python process on the server computer.

Browser runtime validation was not performed in the build environment; Python compilation and ZIP integrity were checked.
