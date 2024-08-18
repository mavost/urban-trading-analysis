# Technical Analysis

## General startup🚀

1. Clone repo and change into project root directory.
2. Create a virtual environment `python3 -m venv python_env`.
3. Activate it `source python_env/bin/activate`.
4. (Optionally): skip transformers and friends, initially, by commenting them out in `requirements.txt`,
i.e., *torch, torchvision, torchaudio, transformers*.
5. Install dependencies `pip install -r requirements.txt`, considering sequential approach, due to package size.

### Startup for trader bot notebook🚀

1. Create an [Alpaca](https://alpaca.markets/) paper trading account and generate API credentials.
2. Copy `.env-dist` file to `.env` and update the `ALP_API_KEY` and `ALP_API_SECRET` with values from your Alpaca account.
3. Use the jupyter notebook: `src/tradingbot.ipynb` to run the trading bot showcase interactively.

## Disclaimers

### Trader bot

- [Youtube](https://www.youtube.com/watch?v=c9OjEThuJjY)
- [Github](https://github.com/nicknochnack/MLTradingBot)

Build a trader bot which looks at sentiment of live news events and trades appropriately.

👨🏾‍💻 Author: Nick Renotte  
📅 Version: 1.x  
📜 License: This project is licensed under the MIT License

## Other References 🔗

[Lumibot](https://github.com/Lumiwealth/lumibot): trading bot library, makes lifecycle stuff easier.
