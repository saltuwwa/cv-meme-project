# Deploy Qazaq Meme Vision on Render

This project uses **`backend/main_deploy.py`** (not `main.py`) so your local dev UI stays unchanged.

---

## 1. Test locally

From the **repository root** (`cv-meme-project/`):

```bash
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements_deploy.txt
uvicorn backend.main_deploy:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000 — you should see the deploy UI with the instruction card.

Health check: http://127.0.0.1:8000/health → `{"status":"ok"}`.

---

## 2. Commit deploy files

```bash
git add backend/main_deploy.py backend/templates/index_deploy.html backend/static/style_deploy.css backend/static/app_deploy.js backend/requirements_deploy.txt backend/__init__.py render.yaml DEPLOY_RENDER.md
git status
git commit -m "Add Render deploy entrypoint and UI"
```

---

## 3. Include the model (~43 MB)

If `models/meme_classifier.pth` is listed in `.gitignore`, force-add it:

```bash
git add -f models/meme_classifier.pth
git commit -m "Track trained model for Render deploy"
```

Alternatively, remove `models/*.pth` from `.gitignore` only if you want the model tracked without `-f` every time.

---

## 4. Push to GitHub

```bash
git push origin main
```

---

## 5. Create a Web Service on Render

1. Sign in at [render.com](https://render.com).
2. **New** → **Web Service** → connect your GitHub repo.
3. Configure:
   - **Root Directory:** leave empty (repo root).
   - **Runtime:** Python 3.
   - **Build Command:**  
     `pip install -r backend/requirements_deploy.txt`
   - **Start Command:**  
     `uvicorn backend.main_deploy:app --host 0.0.0.0 --port $PORT`
4. Choose **Free** plan if available (or upgrade if builds time out — PyTorch install can be slow).
5. Deploy.

**Blueprint:** You can also use **New → Blueprint** and point Render at `render.yaml` in the repo.

---

## 6. After deploy

- Open your public Render URL (HTTPS).
- Browsers require **HTTPS** for `getUserMedia` on most setups — Render provides HTTPS by default.
- Add the URL to your **README**, **portfolio**, and **CV**.

---

## Notes

- First build may take several minutes (torch + torchvision download).
- Free tier services spin down after idle; cold starts can be slow.
- If the free plan is unavailable in your region/account, pick the smallest paid instance.
