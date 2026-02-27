# ScoreBorga 2.5 ⚽🔮

A powerful football prediction engine that fetches data from **Sportmonks API** and **Odds API**, processes analytics, and posts polished weekend predictions to **Telegram** across the **Top 7 European Leagues**.

## ✨ Features

- **Top 7 Predictions**: Only the 7 highest-confidence predictions are sent to Telegram each weekend — no per-league cap
- **Heuristic Totals & Best Total**: Each match includes Over 1.5 / 2.5 / 3.5 probabilities (Poisson-based) and a single "Best Total" pick
- **Ensemble ML Model**: Combines Logistic Regression, Random Forest, and XGBoost via soft voting for sharper probability estimates
- **Hybrid Prediction Mode**: Combines statistical analysis with the ensemble ML model for the best results
- **ML Model Training**: Trains on past 3 seasons of historical data (configurable)
- **Multiple Prediction Modes**: Choose between stats-only, ML-only, or hybrid
- **Real-time Odds Integration**: Incorporates live betting odds for enhanced accuracy
- **Automated Weekend Predictions**: Scheduled to run every Friday

---

## 🏆 Supported Leagues
| # | League | Country |
|---|--------|---------|
| 1 | Premier League | 🏴 England |
| 2 | La Liga | 🇪🇸 Spain |
| 3 | Bundesliga | 🇩🇪 Germany |
| 4 | Serie A | 🇮🇹 Italy |
| 5 | Ligue 1 | 🇫🇷 France |
| 6 | Eredivisie | 🇳🇱 Netherlands |
| 7 | Primeira Liga | 🇵🇹 Portugal |

---

## 🧠 Prediction Modes

ScoreBorga 2.5 supports three prediction modes:

| Mode | Description |
|------|-------------|
| `stat` | Statistics-based weighted scoring using form, H2H, home advantage, and odds |
| `ml` | Ensemble ML model (LR + RF + XGBoost, soft voting) trained on past 3 seasons |
| `hybrid` | **Default** - Combines both approaches for sharper predictions |

### Hybrid Mode (Recommended)
The hybrid mode blends statistical analysis with ensemble machine learning predictions:
- Uses a soft-voting ensemble of Logistic Regression, Random Forest, and XGBoost classifiers
- Combines recent form, head-to-head records, home advantage, and odds
- Configurable ML weight (default: 50% ML + 50% statistics)
- Automatically trains on first run using past 3 seasons of data

---

## 📊 Telegram Output

Each weekend, ScoreBorga 2.5 sends the **top 7 highest-confidence predictions** to your configured Telegram channel, regardless of which league they come from. Each prediction includes:

- Match details (teams, league, kickoff time)
- Predicted outcome and confidence level
- Odds (home / draw / away)
- **Best Total** pick (e.g. `Over 2.5 — 68.4%`)
- Totals breakdown: `O1.5 | O2.5 | O3.5` probabilities
- Reasoning summary

---

## ⚽ Heuristic Totals

Totals probabilities are computed without using any external totals markets. The engine uses a simple **expected-goals (xG) approach**:

1. Estimate home xG = average of home attack and away defence averages
2. Estimate away xG = average of away attack and home defence averages
3. Sum to get expected total goals (`λ`)
4. Apply a **Poisson distribution** to compute `P(goals > 1.5)`, `P(goals > 2.5)`, and `P(goals > 3.5)`

The line with the highest probability is selected as the **Best Total** for the match.

---

## 🤖 Ensemble ML Model

The ML predictor uses a **soft-voting ensemble** of three classifiers:

| Model | Notes |
|-------|-------|
| Logistic Regression | Fast, interpretable baseline |
| Random Forest | Robust tree-based classifier |
| XGBoost | Gradient boosting; requires `xgboost` package |

Probabilities from all available models are averaged (soft vote) before the final prediction is made. If XGBoost is not installed the engine warns and proceeds with the remaining models.

---

## 🏗️ Project Structure
```
ScoreBorga2.5/
├── config/
│   └── settings.py           # API keys, league IDs, config
├── data/
│   ├── sportmonks.py         # Sportmonks API client
│   ├── odds_api.py           # Odds API client
│   └── historical.py         # Historical data fetcher for ML training
├── engine/
│   ├── predictor.py          # Core prediction logic (stat/ml/hybrid + totals)
│   ├── ml_model.py           # Ensemble ML model (LR + RF + XGBoost)
│   ├── analytics.py          # Stats & analytics processing
│   └── polisher.py           # Polish predictions for Telegram
├── leagues/
│   └── top7.py               # Top 7 European leagues definitions
├── models/
│   └── ml_predictor.pkl      # Trained ML model (generated at runtime)
├── scheduler/
│   ├── weekend_runner.py     # Weekend prediction scheduler (CLI entry point)
│   └── schedule_utils.py     # Timezone-aware next-run calculation helpers
├── output/
│   ├── telegram_bot.py       # Telegram bot dispatcher
│   └── dispatcher.py         # Main output dispatcher (top-7 filter)
├── tests/
│   ├── test_predictor.py     # Unit tests for predictor and totals heuristic
│   ├── test_ml_model.py      # Unit tests for ensemble ML model
│   └── test_schedule_utils.py  # Unit tests for timezone-aware scheduling
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/kw97ksvxf9-bit/ScoreBorga2.5.git
cd ScoreBorga2.5
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `xgboost` is included in `requirements.txt`. If it cannot be installed in your environment, the engine will automatically run the ensemble without XGBoost and log a warning.

### 3. Configure environment variables
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 4. Run the prediction engine
```bash
# Run predictions immediately (hybrid mode by default)
python scheduler/weekend_runner.py --run-now

# Start the weekly scheduler (runs every Friday at PREDICTION_RUN_TIME in TIMEZONE)
python scheduler/weekend_runner.py

# Run immediately on startup, then continue scheduling weekly
python scheduler/weekend_runner.py --run-once-then-schedule
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SPORTMONKS_API_KEY` | Sportmonks API key (required) | - |
| `ODDS_API_KEY` | Odds API key (required) | - |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (required) | - |
| `TELEGRAM_CHAT_ID` | Telegram chat ID (required) | - |
| `TIMEZONE` | Timezone for scheduling (used for next-run calculation) | `Europe/London` |
| `PREDICTION_RUN_TIME` | Time to run predictions (HH:MM in `TIMEZONE`) | `09:00` |
| `PREDICTION_MODE` | Prediction mode: `stat`, `ml`, or `hybrid` | `hybrid` |
| `HISTORICAL_SEASONS` | Number of past seasons for ML training | `3` |
| `ML_WEIGHT` | ML weight in hybrid mode (0.0-1.0) | `0.5` |

## Scheduler CLI Reference

| Command | Behaviour |
|---------|-----------|
| `python scheduler/weekend_runner.py` | Start the timezone-aware weekly scheduler (every Friday at `PREDICTION_RUN_TIME` in `TIMEZONE`) |
| `python scheduler/weekend_runner.py --run-now` | Run the pipeline once immediately and exit |
| `python scheduler/weekend_runner.py --run-once-then-schedule` | Run the pipeline immediately, then enter the weekly schedule loop |

> **Timezone handling**: Next-run times are calculated in `TIMEZONE` (e.g. `Europe/London`) and converted to the correct wall-clock instant, so the job runs at the correct local time regardless of the server's system timezone.

---

## 🔑 Required API Keys
- **Sportmonks API** → [sportmonks.com](https://sportmonks.com)
- **Odds API** → [the-odds-api.com](https://the-odds-api.com)
- **Telegram Bot Token** → [@BotFather](https://t.me/BotFather) on Telegram

---

## 🚀 Deploying on Render

### Option A — Blueprint (recommended)

1. Push this repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New → Blueprint** and connect your repository.
3. Render will detect `render.yaml` and pre-fill the worker service configuration.
4. Set the four secret environment variables when prompted (marked `sync: false`):
   - `SPORTMONKS_API_KEY`
   - `ODDS_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
5. Deploy — the worker starts and runs predictions every Friday at 09:00 Europe/London time (timezone-aware).

### Option B — Manual service

1. In the [Render Dashboard](https://dashboard.render.com/), click **New → Background Worker**.
2. Connect your GitHub repository and choose **Docker** as the runtime.
3. Set the environment variables:
   | Key | Value |
   |-----|-------|
   | `SPORTMONKS_API_KEY` | *(your key)* |
   | `ODDS_API_KEY` | *(your key)* |
   | `TELEGRAM_BOT_TOKEN` | *(your token)* |
   | `TELEGRAM_CHAT_ID` | *(your chat id)* |
   | `TIMEZONE` | `Europe/London` |
   | `PREDICTION_RUN_TIME` | `09:00` |
   | `PREDICTION_MODE` | `hybrid` |
   | `HISTORICAL_SEASONS` | `3` |
   | `ML_WEIGHT` | `0.5` |
4. Click **Create Background Worker** — Render will build the Docker image and start the service.

---

## 📄 License
MIT
