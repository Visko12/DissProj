# Discord Bot for Moderation

This is a discord bot that can check messages for toxic stuff and also does sentiment analysis. It can delete bad messages and warn people if they say something not nice.

## Features

- Checks for toxic messages (using a model)
- Can tell if a message is positive, negative or neutral
- Deletes toxic messages
- Warns users
- Logs stuff
- Has some commands

## What you need

- Python 3.8 or higher
- Discord bot token
- Some python packages (see requirements.txt)

## How to install

1. Clone the repo:
```
git clone <repo-url>
cd <repo-folder>
```
2. Install the stuff:
```
pip install -r requirements.txt
```
3. Make a `.env` file in the main folder, put this in it:
```
DISCORD_TOKEN=your_token_here
MODEL_PATH=./models/toxicity_model.h5
SENTIMENT_MODEL_PATH=./models/sentiment_model.h5
```

## Training the models

### Toxicity Model

The bot uses the Jigsaw dataset for training. To train:
```
python train_model.py
```
This will make the model and save it.

### Sentiment Model

To train the sentiment model:
```
python train_sentiment_model.py
```
This will save the model too.

## Running the bot

Make sure you have the models in the `models` folder:
- toxicity_model.h5
- sentiment_model.h5
- tokenizer.pkl

Then run:
```
python bot.py
```

The bot will start and connect to Discord.

## Logging

The bot saves logs in the `logs` folder. There are logs for messages, bot stuff, and training.

## Testing

You can run tests with:
```
python test_bot.py
```

## Folders

```

 
