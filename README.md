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

## 🚀 Deploying on DigitalOcean

### Option A — App Platform (recommended)

1. Push this repository to GitHub.
2. In the [DigitalOcean App Platform](https://cloud.digitalocean.com/apps), create a new app and point it at your repository.
3. DigitalOcean will detect `.do/app.yaml` and pre-fill the service configuration.
4. Set the four secret environment variables in the App Platform dashboard:
   - `SPORTMONKS_API_KEY`
   - `ODDS_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
5. Deploy — the worker starts and runs predictions every Friday at 09:00 UTC.

Alternatively, use the [doctl](https://docs.digitalocean.com/reference/doctl/) CLI:
```bash
doctl apps create --spec .do/app.yaml
```

### Option B — Droplet (Docker)

1. Provision a Ubuntu Droplet and install Docker:
   ```bash
   apt-get update && apt-get install -y docker.io docker-compose-plugin
   ```
2. Copy the project to the Droplet and create your `.env` from the template:
   ```bash
   cp .env.example .env
   # Fill in your API keys
   ```
3. Build and start the container:
   ```bash
   docker compose up -d --build
   ```
4. View live logs:
   ```bash
   docker compose logs -f
   ```

---

## 📄 License
MIT
