AI Even Odd Bot 🤖🎲


AI Even Odd is a Python trading bot that predicts even/odd outcomes using Hidden Markov Models (HMM).
It connects to trading platforms, analyzes historical data, and executes trades automatically based on AI predictions.

🚀 Features

🔹 Predicts even/odd outcomes using HMM-based AI

🔹 Automatically loads and analyzes historical trade data

🔹 Logs trades to trades.csv for review

🔹 Configurable via environment variables (.env)

🔹 Lightweight, easy to deploy, and runs in a virtual environment

🗂 Project Structure
algotrading/
├── evenodd.py        # Main bot script
├── .gitignore        # Git ignore file
├── .env              # Environment variables (not tracked)
├── trades.csv        # Trade logs (ignored in git)
├── venv/             # Python virtual environment (ignored in git)
├── README.md         # This file
├── requirements.txt  # Python dependencies

⚙️ Installation

Clone the repository

git clone https://github.com/keyadaniel56/algotrading.git
cd algotrading

Set up a virtual environment

python3 -m venv venv
source venv/bin/activate

Install dependencies

pip install -r requirements.txt

Create .env file

DERIV_TOKEN=your_api_token_here

⚠️ Keep .env private — do not commit it to GitHub.

🏁 Usage

Run the bot:

python evenodd.py

The bot will:

Connect to the trading platform  
Load historical trade data  
Predict even/odd outcomes using AI  
Log executed trades to trades.csv

🔧 Recommended Settings

Python 3.10 or higher  
Stable internet connection  
Always run in a virtual environment  
Keep .env secure and private

💡 Tips

Review trades.csv regularly to monitor performance  
Adjust AI parameters in evenodd.py as needed  
Add logging to track bot activity

🤝 Contributing

Contributions are welcome:

Fork the repository  
Create a feature branch (git checkout -b feature/your-feature)  
Commit changes (git commit -m "Add feature")  
Push (git push origin feature/your-feature)  
Submit a pull request

⚠️ Disclaimer

Trading involves risk. This bot is for educational and personal use only.  
Do not expose your API token or credentials publicly.

---

# Deriv Digits Over/Under bot (AI + reasoning)

This is a small **educational** algo-trading bot for Deriv **Digits** contracts:

- **Digits Over 5** (wins when last digit is 6–9)  
- **Digits Under 4** (wins when last digit is 0–3)

It uses a lightweight online-learning model with online updates from recent tick digits and prints **reasoning** for each decision (estimated win prob, edge vs payout quote, key signals, and risk gates).

## Safety first

- Start with a **DEMO** token.  
- Keep `DRY_RUN=1` until you confirm the printed reasoning + proposals look right.  
- This bot **can lose money**. There is no guaranteed edge in random digits.

## Setup

```bash
cd /home/cdk/deriv-digits-bot
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `DERIV_TOKEN` (demo token recommended).

## Run

```bash
source .venv/bin/activate
python3 run.py
```

## What “AI reasoning” means here

For each trade opportunity, the bot prints:

- model probabilities for **OVER_5** and **UNDER_4**  
- expected value estimate (edge) from Deriv payout quote  
- why it traded or skipped (thresholds, cooldown, max loss, take profit, etc.)

## Notes

- The bot requests a proposal from Deriv before buying; **edge** is computed from the quoted payout.  
- Contracts are **1 tick** by default (`DURATION=1`, `DURATION_UNIT=t`).
