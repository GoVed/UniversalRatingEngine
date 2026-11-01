# UniversalRatingEngine - GitHub Pages Deployment Guide 🌐

This directory (`docs/`) contains the complete standalone, client-side web application for **UniversalRatingEngine**, featuring interactive searching, filtering, macro cross-genre charts, side-by-side movie comparison, and title inspection across 19,669 movies and 31.6M ratings.

---

## 🚀 How to Enable GitHub Pages (30-second setup)

1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "Add UniversalRatingEngine pipeline and Web UI"
   git push origin main
   ```

2. Open your GitHub Repository in your browser.
3. Click on **Settings** (top tabs) $\to$ **Pages** (in the left sidebar under "Code and automation").
4. Under **Build and deployment**:
   - **Source**: `Deploy from a branch`
   - **Branch**: Select `main` (or your active branch)
   - **Folder**: Select `/docs`
5. Click **Save**.

Your interactive UI will be live within 1–2 minutes at:
`https://<your-username>.github.io/<your-repo-name>/`

---

## 📁 Web Assets Included in `docs/`

- `index.html`: Fully self-contained, responsive single-page web app with Tailwind CSS, Chart.js, and Lucide icons (no npm build required).
- `data/movies_32m.json`: Compact JSON containing all 19,669 calibrated titles from MovieLens 32M (only 0.9 MB gzipped!).
- `data/movies_1m.json`: Curated MovieLens 1M dataset (0.39 MB).
- `data/genre_summary_32m.json` & `1m.json`: Cross-genre aggregate statistics and deflation multipliers.
- `data/universal_ratings_32m.csv` & `1m.csv`: Full raw exportable CSV datasets for researchers.

---

## 🖥️ Local Testing

You can preview the Web UI locally right now:

```bash
# Start a local web server inside the docs folder
python3 -m http.server 8000 --directory docs
```
Then open [http://localhost:8000](http://localhost:8000) in your browser!
