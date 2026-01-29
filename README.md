# navin

This repository contains a small ML project (Flask app and training script).

Files:
- `app.py` — application
- `trainmodel.py` — training script
- `news_data.csv` — dataset (ignored in repo)

To run:
1. Create and activate a virtual environment
2. Install dependencies listed in `requirements.txt` (if provided)
3. Run `python app.py`

Deployment / Model hosting
--------------------------

This app expects `model.pkl` and `vectorizer.pkl` in the repository root. Best practices:

- For small models you can commit them to the repository (not recommended for large files).
- Preferred: host model files on a public URL (S3, GitHub Releases, or similar) and set the following environment variables in your deployment platform:
	- `MODEL_URL` — public URL to `model.pkl`
	- `VECT_URL` — public URL to `vectorizer.pkl`

The app will attempt to download these files at startup if they are missing locally.

Alternative: use Git LFS for large model files and configure your deployment to fetch LFS objects.

Automated Releases (recommended)
-------------------------------

This repository includes a GitHub Actions workflow that, when `model.pkl` and/or `vectorizer.pkl` are present in a commit, will create a GitHub Release and upload the model files as release assets. The release will be tagged `model-<short-sha>`.

To use a release asset as your `MODEL_URL`/`VECT_URL`, the public download URL pattern is:

`https://github.com/<owner>/<repo>/releases/download/<tag>/model.pkl`

Example:

`MODEL_URL=https://github.com/NAVIN-TG/navin/releases/download/model-abc123/model.pkl`

This approach keeps the large binaries out of the repository history while making them easily downloadable by the app.

