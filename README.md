# ScoreBorga 2.5 ⚽🔮

A powerful football prediction engine that fetches data from **Sportmonks API** and **Odds API**, processes analytics, and posts polished weekend predictions to **Telegram** across the **Top 7 European Leagues**.

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

## 🏗️ Project Structure
```
ScoreBorga2.5/
├── config/
│   └── settings.py           # API keys, league IDs, config
├── data/
│   ├── sportmonks.py         # Sportmonks API client
│   └── odds_api.py           # Odds API client
├── engine/
│   ├── predictor.py          # Core prediction logic
│   ├── analytics.py          # Stats & analytics processing
│   └── polisher.py           # Polish predictions using external engines
├── leagues/
│   └── top7.py               # Top 7 European leagues definitions
├── scheduler/
│   └── weekend_runner.py     # Weekend prediction scheduler
├── output/
│   ├── telegram_bot.py       # Telegram bot dispatcher
│   └── dispatcher.py        # Main output dispatcher
├── tests/
│   └── test_predictor.py     # Unit tests
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
python scheduler/weekend_runner.py
```

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
4. Click **Create Background Worker** — Render will build the Docker image and start the service.

---

## 📄 License
MIT
