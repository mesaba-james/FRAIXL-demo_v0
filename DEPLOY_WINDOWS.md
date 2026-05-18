# Deployment Guide — FRAIXL-Air Demo
## For Windows + Anaconda + GitHub + Streamlit Community Cloud

This guide takes you from a zip file to a public URL David can click. About
30 minutes total. Two phases: run it on your laptop first to confirm it
works, then put it on the internet.

---

## Phase 1 — Run it locally (10 minutes)

You're doing this *before* hosting it so you can see it working and catch
any issues in a place where it's easy to fix things.

### Step 1.1 — Unzip the app

1. You should have downloaded `fraixl_app.zip`.
2. Right-click it → **Extract All** → choose a simple location like
   `C:\fraixl_app`. Don't put it in OneDrive or a path with spaces if you
   can avoid it.
3. Open the folder. You should see `app.py`, `requirements.txt`, a
   `samples` folder with one CSV inside it, and a few other Python files.

### Step 1.2 — Open Anaconda Prompt and install the libraries

1. **Start menu → Anaconda Prompt** (not regular Command Prompt).
2. Navigate to the app folder. Type this exactly and hit Enter:
   ```
   cd C:\fraixl_app
   ```
   (Adjust the path if you put it somewhere else.)
3. Install the four libraries the app needs. Type this and hit Enter:
   ```
   pip install streamlit pandas plotly anthropic
   ```
   This will take 1–2 minutes. You'll see a lot of text scroll by. Wait
   until you get a normal prompt back.

### Step 1.3 — Launch the app

1. Still in the Anaconda Prompt in the same folder, type:
   ```
   streamlit run app.py
   ```
2. A browser tab opens automatically pointing at `http://localhost:8501`.
3. You should see the FRAIXL-Air demo with the bundled flight already loaded.

### Step 1.4 — Try it

- Click through the four tabs.
- Move some sliders in the sidebar — the state distributions should change.
- If you want to test the Claude analyst panel, paste your Anthropic
  API key in the sidebar and click "Generate analyst summary with Claude"
  on tab 4.

### Step 1.5 — Stop the app

In the Anaconda Prompt window, press **Ctrl+C**. The server stops.

**If anything broke in Phase 1, stop here and tell me what you saw.**
Don't move on to hosting if local doesn't work — we fix it first.

---

## Phase 2 — Host it on Streamlit Community Cloud (20 minutes)

### Step 2.1 — Create a GitHub repository

1. Go to [github.com](https://github.com) and sign in.
2. Click the **+** in the top right → **New repository**.
3. Fill in:
   - **Repository name:** `fraixl-air-demo` (or whatever you like)
   - **Description:** "FRAIXL-Air Landing Phase Twin demonstrator"
   - **Public** (Streamlit Cloud free tier requires public repos)
   - **DO NOT** check "Add a README" — we have our own
   - Leave everything else default
4. Click **Create repository**.
5. You'll land on a page that says "Quick setup". Leave this tab open;
   we'll come back to it.

### Step 2.2 — Upload the app files via the web (no git commands needed)

1. On the GitHub page from step 2.1, look for the link that says
   **"uploading an existing file"** (in the "or push an existing repository"
   section, or near the top there should be an "uploading an existing file"
   link).
2. You'll get a drag-and-drop zone.
3. **Important**: In Windows Explorer, open your `C:\fraixl_app` folder,
   select all of these (Ctrl+A or selectively pick):
   - `app.py`
   - `fraixl_extract.py`
   - `fram_metadata.py`
   - `parameter_dictionary.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - The `samples` folder
   - The `.streamlit` folder
4. Drag them all into the GitHub drag-and-drop zone.
5. Wait for the files to upload (sample CSV is 7MB so this takes a moment).
6. Scroll down. In the commit message box, type: `Initial commit`.
7. Click **Commit changes**.

You should now see all the files listed in your repository.

### Step 2.3 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **Sign in with GitHub** (top right). Authorize when prompted.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in the form:
   - **Repository:** `your-username/fraixl-air-demo`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pick something memorable, e.g. `fraixl-air-demo`
     (final URL will be `https://fraixl-air-demo.streamlit.app`)
5. **Before clicking Deploy**, click **Advanced settings**:
   - **Python version:** 3.11 (or 3.12 — both fine)
   - **Secrets:** paste this, replacing the placeholder:
     ```
     ANTHROPIC_API_KEY = "sk-ant-your-actual-key-here"
     ```
   - This is where your Anthropic key goes server-side. David will never
     see it; the app just uses it transparently.
6. Click **Deploy**.

Streamlit Cloud now spins up your app. You'll see a log streaming —
it's installing the packages from `requirements.txt`. This takes 2–4
minutes the first time.

When it's done, you'll see your app running. Test it.

### Step 2.4 — Send David the URL

You're done. The URL is `https://YOUR-APP-NAME.streamlit.app`.
Send it to him in an email like:

> Hi David,
>
> I've built a working prototype of the FRAIXL-Air Landing Phase Twin
> against real 737 FDR data — implementing your v0.3 metadata equations
> and the data-to-model dictionary, exercised on an actual ATL approach.
> The app is live at: https://YOUR-APP-NAME.streamlit.app
>
> Four tabs walk through: the parameter dictionary (downloadable as JSON),
> the flight profile with event markers, the v0.3 derived function states
> including the 1000ft gate assessment, and a calibration/discrepancy
> view with an optional Claude-generated analyst summary that demonstrates
> the bounded LLM layer your proposal describes.
>
> The thresholds are deliberately provisional — they're a placeholder
> until calibration against a corpus of reviewed normal landings is done.
> The sliders in the sidebar let you see how the states shift as the
> calibration profile changes.
>
> Source code is at https://github.com/YOUR-USERNAME/fraixl-air-demo if
> you want to look under the hood.

---

## Adding more sample flights later

When you get more FDR CSVs you want David (or others) to try:

1. On the GitHub repo page, click into the `samples` folder.
2. Click **Add file** → **Upload files**.
3. Drag the new CSV in. Give it a descriptive name like
   `737_LGA_high_energy.csv` so people can tell flights apart in the
   dropdown.
4. Commit. Streamlit Cloud auto-redeploys within a minute, and the new
   flight appears in the sidebar dropdown.

**Privacy reminder:** Before adding any new flight, confirm with whoever
manages the Delta FOQA program that you're cleared to publish it. Even
anonymized FDR data has handling rules at most airlines. The bundled
sample doesn't include tail number or flight number identifiers, but the
lat/lon track does show the ATL approach — worth a quick internal check
before this URL leaves your hands.

---

## Troubleshooting

### "pip is not recognized"
You opened regular Command Prompt instead of Anaconda Prompt. Close it
and open **Anaconda Prompt** from the Start menu.

### Local: "streamlit: command not found"
The `pip install` didn't finish or didn't run in the Anaconda
environment. Reopen Anaconda Prompt and try `pip install streamlit`
again.

### Local: browser opens but page shows an error
Look at the Anaconda Prompt window — it'll show a Python traceback.
Copy the bottom 10–15 lines and paste them to me; I can diagnose from
that.

### Streamlit Cloud: deploy fails with "ModuleNotFoundError"
`requirements.txt` is missing something or didn't upload. On GitHub,
check the file is in the root of the repo. If it's missing, re-upload it.

### Streamlit Cloud: app loads but "Sample flight" dropdown is empty
The `samples` folder didn't upload, or uploaded as an empty folder.
GitHub doesn't always preserve empty folders. Go to the repo, navigate
to the `samples` folder — if it's missing, click "Add file" → "Upload
files" and drag in `737_ATL_nominal.csv` and commit.

### Streamlit Cloud: Claude panel button does nothing / errors
The secret didn't save. Go to the app's **Settings → Secrets** in
Streamlit Cloud and re-paste:
```
ANTHROPIC_API_KEY = "sk-ant-your-actual-key-here"
```
Hit Save. The app restarts automatically.

### "I don't see the .streamlit folder when uploading"
Windows hides folders starting with a dot by default. In File Explorer:
**View → Show → Hidden items** (toggle it on). The `.streamlit` folder
will appear.

---

## What you can change later

Everything's editable on GitHub. The sidebar slider defaults, the
threshold ranges, the LLM prompt for the analyst summary — all live in
`app.py` and `fram_metadata.py`. Edit on GitHub directly (click any file,
click the pencil icon), commit, and Streamlit Cloud redeploys automatically.

If you want me to make changes for you, send me what you want different
and I'll produce updated files.
