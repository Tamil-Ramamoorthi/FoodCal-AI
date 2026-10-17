# Deploying FoodCal AI

FoodCal AI is deployed for free like this:

| Part | Where it runs | Address |
|---|---|---|
| React frontend (Vite build) | Vercel, project `food-cal-ai` | <https://food-cal-ai-sigma.vercel.app> |
| FastAPI backend, EfficientNet-B0, SQLite history | Your Windows PC | `http://127.0.0.1:8000` (local only) |
| Public HTTPS access to the backend | Cloudflare Quick Tunnel (`cloudflared`) | `https://<random-words>.trycloudflare.com` |

Visitors' browsers load the frontend from Vercel and call the backend
through the tunnel. **The site only works while your PC is on, connected to
the internet, and both the backend and the tunnel are running.** Nothing is
port-forwarded and the Windows firewall stays as it is: `cloudflared` makes
an outgoing connection to Cloudflare, and the backend listens only on
`127.0.0.1`.

A paid alternative that runs everything on Render is described at the end
([Alternative: Render](#alternative-render)).

## 1. Required software

| Software | Used for | Check |
|---|---|---|
| Python environment `.venv-gpu` (Python 3.12, PyTorch with CUDA) | Backend | `.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"` |
| `cloudflared` | The tunnel | `cloudflared --version` |
| Node.js 24 and npm | Local frontend development and builds only | `node --version` |
| Vercel CLI (optional) | Changing Vercel settings from a terminal | `vercel --version` |

Install `cloudflared` if it is missing (no Cloudflare account is needed for a
Quick Tunnel):

```powershell
winget install --id Cloudflare.cloudflared
```

Open a new PowerShell window afterwards so `cloudflared` is on the `PATH`.
It installs to `C:\Program Files (x86)\cloudflared\cloudflared.exe`.

## 2. Windows setup (once)

All commands run in PowerShell from the project folder:

```powershell
cd "C:\Users\tamil\OneDrive\Documents\FoodCal AI"
```

The backend uses the existing `.venv-gpu` environment and files that are
already in the repository: the checkpoint
`models\efficientnet_b0\food_efficientnet_b0_best.pth` (34 classes, read from
the checkpoint), `backend\app\services\nutrition_reference.json` and the
evaluation reports. History is stored in `storage\foodcal.db` with
thumbnails in `storage\thumbnails\`; both are created on first start and are
never committed. It uses the GPU when CUDA is available and the CPU
otherwise.

**Admin token.** Changing or deleting saved history (feedback, delete one,
clear all) through the public URL needs a token that only you know.
Requests made directly on this PC (the local dev frontend) do not. Create
the token once and store it for your Windows user:

```powershell
$token = .\.venv-gpu\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
[Environment]::SetEnvironmentVariable("FOODCAL_ADMIN_TOKEN", $token, "User")
$token   # copy this; you enter it on the website in step 7
```

Only PowerShell windows opened **after** this see the variable. Without a
token, the public site can still predict and show history, but every change
or deletion through it is refused (HTTP 403). To read the token again later:
`[Environment]::GetEnvironmentVariable("FOODCAL_ADMIN_TOKEN", "User")`. Never
put it in the repository or in Vercel.

**CORS.** The backend allows browser requests from
`https://food-cal-ai-sigma.vercel.app` and the local dev servers
(`http://localhost:5173`, `http://127.0.0.1:5173`, and the same on port
5174) by default. Setting `FOODCAL_CORS_ORIGINS` (comma-separated) replaces
that whole list, so include the Vercel URL if you set it.

## 3. Start the backend

Window 1:

```powershell
cd "C:\Users\tamil\OneDrive\Documents\FoodCal AI"
.\.venv-gpu\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete.`, then check
<http://127.0.0.1:8000/>. It should report `"classes": 34`,
`"device": "cuda"` (or `"cpu"`), `"history_storage": "ok"` and
`"nutrition_reference": "ok"`. The API documentation is at
<http://127.0.0.1:8000/docs>.

- Keep `--host 127.0.0.1`. `0.0.0.0` would also expose the backend to your
  local network.
- Run only one backend. A second one cannot start while port 8000 is in use
  (`[WinError 10048]`); stop the old one first.
- Do not use `--reload` for the deployment.

## 4. Start the Cloudflare Quick Tunnel

Window 2 (the backend must be running):

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

After a few seconds it prints a box with the public URL, for example:

```
Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):
https://example-random-words-here.trycloudflare.com
```

- **Leave this window open.** Closing it or pressing Ctrl+C ends the tunnel.
- **The URL is random and changes every time the tunnel starts.** It cannot
  be chosen or kept. Use the URL this run printed, never an older one.
- Check it from any browser: `https://<your-url>/` shows the same JSON as
  step 3.
- Only the backend API is reachable through the tunnel. The database,
  model and other project files are not served (they return 404), and the
  frontend dev server is not exposed.

## 5. Configure Vercel

The Vercel project `food-cal-ai` is linked to the GitHub repository and
deploys `main` automatically. Its settings, already in place:

| Setting | Value |
|---|---|
| Root directory | `frontend` |
| Framework preset | Vite |
| Build command | `npm run build` |
| Output directory | `dist` |
| Environment variable | `VITE_API_BASE_URL` (Production) = the tunnel URL from step 4 |

Vite builds `VITE_API_BASE_URL` into the JavaScript bundle, so **every
change to it needs a redeploy**. A production build fails on purpose if it is
missing, not `https://`, or a local address, so the deployed site can never
call `localhost`. Local development (`npm run dev`) still uses
`http://127.0.0.1:8000`.

**Set or update the URL in the dashboard:** vercel.com → project
`food-cal-ai` → Settings → Environment Variables → `VITE_API_BASE_URL` →
Edit → paste `https://<your-url>.trycloudflare.com` (no trailing `/`) →
Save. Then redeploy (step 6).

**Or with the Vercel CLI**, from any folder linked to the project
(`vercel link --project food-cal-ai` creates a `.vercel` folder there):

```powershell
vercel env rm VITE_API_BASE_URL production --yes
vercel env add VITE_API_BASE_URL production   # paste the URL when asked
```

## 6. Redeploy the frontend

- **After changing `VITE_API_BASE_URL`:** dashboard → Deployments → the
  latest Production deployment → ⋯ → Redeploy. Or with the CLI:
  `vercel ls food-cal-ai` to find the latest Production deployment URL, then
  `vercel redeploy <that-url> --target production`.
- **After changing frontend code:** push to `main` on GitHub; Vercel builds
  and deploys it with the current `VITE_API_BASE_URL`.

A deploy takes under a minute. To confirm which backend the live site uses,
open its Settings page: "Backend API" shows the URL.

## 7. Test the public application

1. Open <https://food-cal-ai-sigma.vercel.app>. The sidebar shows
   "FastAPI Engine: Online" and the top bar "EfficientNet-B0".
2. Analyze Food → upload a food photo → a prediction with confidence and
   device appears, with a nutrition estimate.
3. Prediction History lists the new record. Nutrition and Model Evaluation
   load.
4. Settings → Admin token → paste the token from step 2 → Save token. This
   browser can now verify, delete and clear history; other visitors cannot.
   It is stored only in this browser.

Without the token, deleting through the public site shows "A valid admin
token is needed…" and nothing is deleted.

## 8. When the backend goes offline

The site keeps loading (it is static on Vercel) but shows "API Offline". It
re-checks the backend every 5 seconds.

- **Backend window closed or crashed:** start it again (step 3). The tunnel
  keeps its URL while its own window stays open, so the site turns Online by
  itself within a few seconds. No Vercel change is needed.
- **Tunnel window closed:** start a new tunnel (step 4). It gets a **new
  URL**, so update `VITE_API_BASE_URL` (step 5) and redeploy (step 6).

## 9. When the PC shuts down, sleeps or restarts

The backend and the tunnel stop, and the site shows "API Offline". Saved
history stays on disk. After the PC is back:

1. Start the backend (step 3).
2. Start the tunnel (step 4). The URL will be new.
3. Update `VITE_API_BASE_URL` and redeploy (steps 5–6).

To keep the site reachable, stop Windows from sleeping while it should be
online (Settings → System → Power).

## 10. When the tunnel URL changes

This happens every time `cloudflared tunnel --url …` starts. Until Vercel is
redeployed with the new URL, the site keeps calling the old, dead address and
shows "API Offline". Update the variable (step 5), redeploy (step 6), and
reload the page.

A fixed URL needs a *named* Cloudflare Tunnel, which needs a Cloudflare
account and a domain you control. That is not covered here.

## 11. Security limitations

- **Anyone with the tunnel URL can use the API.** There are no user
  accounts. Visitors can upload photos (each prediction is saved to *your*
  history with a small thumbnail), and read all saved history, including
  thumbnails and original filenames. The API documentation (`/docs`) is
  public too.
- **Changes need the admin token.** Feedback, deleting a record and clearing
  all history (`DELETE /api/history?confirm=all`) are refused for requests
  through the tunnel unless they carry the token in the
  `X-FoodCal-Admin-Token` header, and refused entirely if
  `FOODCAL_ADMIN_TOKEN` is not set. Requests made directly on this PC are
  trusted. The token is a single shared secret: keep it private, and change
  it (step 2, then restart the backend) if it leaks. It is stored in the
  browser's local storage on the devices where you enter it.
- **CORS is not access control.** It stops other websites from calling the
  API from a visitor's browser, but not direct requests (e.g. `curl`).
- **Limits.** Uploads must be images of at most 10 MB that decode correctly.
  Predictions through the tunnel are limited to 20 per minute per visitor
  address (HTTP 429 beyond that). Requests on this PC are not limited.
- **Error messages** returned by the API contain no file paths or stack
  traces; details appear only in the backend window.
- **Quick Tunnels are meant for testing.** They have no uptime guarantee and
  no access control of their own. For anything beyond a personal demo, use a
  named tunnel with Cloudflare Access in front of it.
- **The project folder is in OneDrive.** `storage\` (history and
  thumbnails) is therefore synced to your OneDrive account, although it is
  never committed to Git.

## 12. Stop the services

Press Ctrl+C in the tunnel window, then in the backend window (or close
both windows). The public site then shows "API Offline". Nothing is deleted.

## 13. Preserve and back up history

History is `storage\foodcal.db` plus `storage\thumbnails\`. Never delete
`storage\` or the database to "clean up"; use Clear All History in the app
instead, which keeps the database itself.

Back up (safe while the backend runs; `backups\` is ignored by Git):

```powershell
cd "C:\Users\tamil\OneDrive\Documents\FoodCal AI"
$dest = "backups\storage_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path $dest | Out-Null
.\.venv-gpu\Scripts\python.exe -c "import sqlite3, sys; src = sqlite3.connect('storage/foodcal.db'); dst = sqlite3.connect(sys.argv[1] + '/foodcal.db'); src.backup(dst); dst.close()" $dest
Copy-Item storage\thumbnails "$dest\thumbnails" -Recurse
```

Restore: stop the backend, copy `foodcal.db` and `thumbnails` from the backup
into `storage\` (replacing what is there), then start the backend.

---

## Alternative: Render

The repository also contains a `render.yaml` for running both parts on
Render instead. This is not the current deployment. Render's free plan
cannot hold the model and loses history on restart (see below), so it needs
a paid plan.

| Service | Type | What it runs |
|---|---|---|
| `foodcal-ai-api` | Web service (Python) | FastAPI + the EfficientNet-B0 checkpoint on CPU |
| `foodcal-ai` | Static site | The React frontend built by Vite |

The frontend calls the API directly from the browser, so the two services
need each other's URLs: the frontend's `VITE_API_BASE_URL` and the API's
`FOODCAL_CORS_ORIGINS`.

### Backend (`foodcal-ai-api`)

| Setting | Value |
|---|---|
| Root directory | repository root (the backend reads `models/`, `reports/` and `data/metadata/` from there) |
| Python | 3.12, from `.python-version` (Render's default is newer and untested here) |
| Build command | `pip install -r backend/requirements.txt` |
| Start command | `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health check | `/` |
| Plan | Standard (2 GB RAM); see [Memory](#memory-and-plan) |

`backend/requirements.txt` holds only the runtime packages, pinned to the
tested versions, with CPU-only `torch==2.11.0+cpu` and
`torchvision==0.26.0+cpu` from the PyTorch CPU index. The root
`requirements.txt` (training and evaluation tools) is not used by the
deployment.

| Variable | Value | Purpose |
|---|---|---|
| `FOODCAL_CORS_ORIGINS` | the static site URL, e.g. `https://foodcal-ai.onrender.com` (comma-separate several; a trailing `/` is ignored) | Browsers may call the API only from these origins. |
| `FOODCAL_ADMIN_TOKEN` | a random secret (as a Render secret) | Needed to change or delete history: every request reaches Render through its proxy, so none count as local. |
| `FOODCAL_DATA_DIR` | `/var/data/foodcal` | Where the SQLite history and thumbnails are stored (created on startup) |
| `OMP_NUM_THREADS` | `1` | Keeps PyTorch to the instance's CPU allowance |
| `FOODCAL_LOW_CONFIDENCE_THRESHOLD` | optional, default `50` | Unchanged existing setting |

The deployment relies on these committed files: the checkpoint
`models/efficientnet_b0/food_efficientnet_b0_best.pth` (16 MB),
`backend/app/services/nutrition_reference.json`,
`models/efficientnet_b0/evaluation_report.json`,
`data/metadata/verification_report.json` and
`reports/real_world/real_world_evaluation.json`.

### Frontend (`foodcal-ai`)

Root directory `frontend`, build command `npm ci && npm run build`, publish
directory `dist`, `NODE_VERSION=24`, and `VITE_API_BASE_URL` = the API's
public URL (a redeploy is needed after changing it).

### First deployment

1. Push the branch to GitHub.
2. In Render: **New → Blueprint**, pick the repository. Render reads
   `render.yaml` and asks for the `sync: false` values. On the first run the
   URLs are not known yet: enter the default ones Render will assign,
   `https://foodcal-ai-api.onrender.com` and `https://foodcal-ai.onrender.com`,
   and fix them in step 4 if they differ.
3. Wait for both services to deploy. Open each service to see its real URL.
4. If a URL differs from what you entered, update `FOODCAL_CORS_ORIGINS` on
   the API (it restarts) and `VITE_API_BASE_URL` on the static site
   (redeploy it).
5. Check: `https://<api>/` returns `"nutrition_reference": "ok"` and
   `"history_storage": "ok"`; an uploaded photo returns a prediction and
   nutrition.

### Memory and plan

Measured locally with the deployment packages (CPU-only PyTorch, one worker,
Windows; Linux figures will differ somewhat):

| State | Peak memory |
|---|---|
| Idle, model loaded | ~325 MB |
| Small image (dataset size) | ~336 MB |
| 12 MP phone photo | ~466 MB |
| 27 MP photo | ~640 MB |

Uploads are decoded at full resolution before the model's resize, so memory
grows with the photo's pixel count.

- **Free (512 MB, 0.1 CPU): not suitable.** Too little memory for large
  photos; it spins down after 15 minutes idle; it cannot have a disk, so
  history is wiped on every restart and spin-down.
- **Starter (512 MB, 0.5 CPU): risky.** Large photos can exceed the limit.
- **Standard (2 GB, 1 CPU): recommended.** Check current prices on
  <https://render.com/pricing>.

### Storage and persistence

- **With a disk** (1 GB at `/var/data`, paid plans only): history and
  thumbnails survive deploys and restarts.
- **Without a disk:** the filesystem is ephemeral, and **all history and
  feedback is lost on every deploy, restart or spin-down.**
- Thumbnails are about 15 KB each, so 1 GB holds tens of thousands.
