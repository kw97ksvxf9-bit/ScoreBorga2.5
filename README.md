# ScoreBorga 2.5 ⚽🔮

A powerful football prediction engine that fetches data from **Sportmonks API** and **Odds API**, processes analytics, and posts polished weekend predictions to **Telegram** across the **Top 7 European Leagues**.

## ✨ Features

- **Hybrid Prediction Mode**: Combines statistical analysis with machine learning for sharper predictions
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
| `ml` | Machine learning model trained on past 3 seasons of historical match data |
| `hybrid` | **Default** - Combines both approaches for sharper predictions |

### Hybrid Mode (Recommended)
The hybrid mode blends statistical analysis with machine learning predictions:
- Uses a Random Forest classifier trained on historical data
- Combines recent form, head-to-head records, home advantage, and odds
- Configurable ML weight (default: 50% ML + 50% statistics)
- Automatically trains on first run using past 3 seasons of data

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
│   ├── predictor.py          # Core prediction logic (stat/ml/hybrid modes)
│   ├── ml_model.py           # Machine learning model (Random Forest)
│   ├── analytics.py          # Stats & analytics processing
│   └── polisher.py           # Polish predictions for Telegram
├── leagues/
│   └── top7.py               # Top 7 European leagues definitions
├── models/
│   └── ml_predictor.pkl      # Trained ML model (generated at runtime)
├── scheduler/
│   └── weekend_runner.py     # Weekend prediction scheduler
├── output/
│   ├── telegram_bot.py       # Telegram bot dispatcher
│   └── dispatcher.py         # Main output dispatcher
├── tests/
│   ├── test_predictor.py     # Unit tests for predictor
│   └── test_ml_model.py      # Unit tests for ML model
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

### 3. Configure environment variables
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 4. Run the prediction engine
```bash
# Run predictions immediately (hybrid mode by default)
python scheduler/weekend_runner.py --run-now

# Start the scheduler (runs every Friday at 09:00)
python scheduler/weekend_runner.py
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
| `TIMEZONE` | Timezone for scheduling | `Europe/London` |
| `PREDICTION_RUN_TIME` | Time to run predictions | `09:00` |
| `PREDICTION_MODE` | Prediction mode: `stat`, `ml`, or `hybrid` | `hybrid` |
| `HISTORICAL_SEASONS` | Number of past seasons for ML training | `3` |
| `ML_WEIGHT` | ML weight in hybrid mode (0.0-1.0) | `0.5` |

---

## 🔑 Required API Keys
- **Sportmonks API** → [sportmonks.com](https://sportmonks.com)
- **Odds API** → [the-odds-api.com](https://the-odds-api.com)
- **Telegram Bot Token** → [@BotFather](https://t.me/BotFather) on Telegram

---

## 📬 Telegram Output
Predictions are automatically posted to your configured Telegram channel/group every weekend with detailed match analysis.

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
5. Deploy — the worker starts and runs predictions every Friday at 09:00 Europe/London time.

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
