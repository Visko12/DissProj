import discord
from discord.ext import commands
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import traceback
from collections import defaultdict

#set up dotenv::environment
load_dotenv()

#configure logging
def setup_LogIn():
    #creates directory if it deosnt exist
    os.makedirs('logs', exist_ok=True)
    
    #creates a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/discord_bot_{timestamp}.log'
    
    #configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

#set up logger
logger = setup_LogIn()

class ContentModerator:
    def __init__(self):
        self.logger = logging.getLogger('ContentModerator')
        self.logger.setLevel(logging.INFO)
        
        #creates directory if it doesnt exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        log_file = f'logs/moderator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        #sets up tracking of warnings
        self.warnings = defaultdict(int)  #this tracks the amount of warnings per user
        self.last_warning_time = defaultdict(datetime)  #this tracks when users got their last warning
        
        self.logger.info("Loading models and tokenizer....")
        
        try:
            #loads toxicity model
            model_path = 'models/toxicity_model_final.keras'
            self.logger.info(f"Loading model from: {model_path}")
            self.toxicity_model = tf.keras.models.load_model(model_path)
            self.logger.info("Toxicity model loaded successfully")
            
            #loads tokenizer
            tokenizer_path = 'models/tokenizer.pkl'
            self.logger.info(f"Loading tokenizer from: {tokenizer_path}")
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            self.logger.info("Tokenizer loaded successfully")
            
            #verifies model architecture
            self.logger.info("Model architecture:")
            self.toxicity_model.summary(print_fn=self.logger.info)
            
            #verifies tokenizer
            self.logger.info(f"Tokenizer vocabulary size: {len(self.tokenizer.word_index)}")
            
            # Verify model with test cases
            test_cases = [
                "you are stupid",
                "hello world",
                "i hate you",
                "this is a test",
            ]
            self.logger.info("Verifying model with test cases:")
            for test in test_cases:
                sequence = self.tokenizer.texts_to_sequences([test])
                self.logger.info(f"Tokenized sequence for '{test}': {sequence}")
                padded = pad_sequences(sequence, maxlen=50)
                score = self.toxicity_model.predict(padded, verbose=0)[0][0]
                self.logger.info(f"Test case '{test}': {score:.4f}")
            
            # Initialize sentiment model
            try:
                sentiment_model_path = 'models/sentiment_model_final.h5'
                self.logger.info(f"Loading sentiment model from: {sentiment_model_path}")
                self.sentiment_model = tf.keras.models.load_model(sentiment_model_path)
                self.logger.info("Sentiment model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Sentiment model not loaded: {e}")
                self.sentiment_model = None
            
            self.logger.info("Models and tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def create_warning_embed(self, user, warning_count, message_content):
        colours = {
            1: 0x00ff00,  #green for the first warning
            2: 0xffa500,  #amber for the second warning
            3: 0xff0000   #red for the third warning
        }
        titles = {
            1: "First Warning",
            2: "Second Warning",
            3: "Final Warning"
        }
        
        descriptions = {
            1: "This is your first warning. Please be mindful of your language!!!",
            2: "This is your second warning. One more violations will result in a kick!!!",
            3: "This is your final warning. One more violation will result in a kick!!!"
        }
        
        embed = discord.Embed(
            title=titles[warning_count],
            description=descriptions[warning_count],
            color=colours[warning_count]
        )
        
        embed.add_field(name="User", value=user.mention, inline=True)
        embed.add_field(name="Warning Count", value=f"{warning_count}/3", inline=True)
        embed.add_field(name="Message Content", value=message_content, inline=False)
        embed.set_footer(text=f"Warning issued at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return embed

    def reset_warnings(self, user_id):
        self.warnings[user_id] = 0
        self.last_warning_time[user_id] = datetime.now()
        self.logger.info(f"Reset warnings for user {user_id}")

    def log_message(self, message, toxicity_score=None, action_taken=None, warning_count=None, sentiment=None, sentiment_confidence=None):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'platform': 'discord',
            'channel': getattr(message.channel, 'name', None),
            'author_name': getattr(message.author, 'name', None),
            'author_id': getattr(message.author, 'id', None),
            'message': getattr(message, 'content', None),
            'toxicity_score': toxicity_score,
            'action_taken': action_taken,
            'warning_count': warning_count,
            'sentiment': sentiment,
            'sentiment_confidence': sentiment_confidence
        }
        os.makedirs('logs', exist_ok=True)
        with open('logs/message_log.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

    def check_sentiment(self, text):
        if not self.sentiment_model or not hasattr(self, 'tokenizer'):
            return 'neutral', 0.0
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
        prediction = self.sentiment_model.predict(padded, verbose=0)[0]
        sentiment_idx = int(np.argmax(prediction))
        sentiments = ['negative', 'neutral', 'positive']
        return sentiments[sentiment_idx], float(prediction[sentiment_idx])

    async def check_message(self, message):
        #skip messages from bots
        if message.author.bot:
            return
        
        print(f"[Discord] Message from {message.author}: {message.content}")
        sequence = self.tokenizer.texts_to_sequences([message.content])
        self.logger.info(f"Tokenized sequence: {sequence}")
        
        padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
        self.logger.info(f"Padded sequence shape: {padded.shape}")
        
        #converts messages to token sequences and then logs it
        sequence = self.tokenizer.texts_to_sequences([message.content])
        self.logger.info(f"Tokenized sequence: {sequence}") 
        
        #predict and logs the toxicity score for the message so it can be used to take action
        toxicity_score = float(self.toxicity_model.predict(padded, verbose=0)[0][0])
        print(f"[Discord] Toxicity score for '{message.content}': {toxicity_score:.4f}")
        self.logger.info(f"Toxicity score: {toxicity_score:.4f}")
        
        #logs messages
        self.logger.info(f"Message from {message.author}: {message.content}")
        self.logger.info(f"Processing message: {message.content}")
        
        #checks and logs the sentiment of the message
        sentiment, sentiment_confidence = self.check_sentiment(message.content)
        print(f"[Discord] Sentiment: {sentiment} (confidence: {sentiment_confidence:.4f})")
        self.logger.info(f"Sentiment: {sentiment} (confidence: {sentiment_confidence:.4f})")
        
        action_taken = None
        warning_count = self.warnings[message.author.id] + 1 if message.author.id in self.warnings else 1
        
        #takes action based on the toxicity score
        if toxicity_score > 0.9:  #set too 0.9 for testing but high toxcity score
            try:
                #deletes the message
                await message.delete()
                print(f"[Discord] Deleted toxic message from {message.author}")
                self.logger.info(f"Deleted toxic message from {message.author}")
                
                #increases users warning count and makrs the messages for deletion
                self.warnings[message.author.id] += 1
                warning_count = self.warnings[message.author.id]
                action_taken = 'deleted'
                
                if warning_count <= 3:
                    #creates and sends warnings
                    warning_embed = self.create_warning_embed(
                        message.author,
                        warning_count,
                        message.content
                    )
                    await message.channel.send(embed=warning_embed)
                    #logs the warning
                    self.log_message(message, toxicity_score, 'warning', warning_count, sentiment, sentiment_confidence)
                    
                    #third warning kicks the user
                    if warning_count == 3:
                        try:
                            await message.author.kick(reason="Received three warnings for toxic behavior")
                            print(f"[Discord] Kicked user {message.author} after three warnings")
                            self.logger.warning(f"Kicked user {message.author} after three warnings")
                            #logs the kick
                            self.log_message(message, toxicity_score, 'kick', warning_count, sentiment, sentiment_confidence)
                            #resets warnings after kick
                            self.reset_warnings(message.author.id)
                        except discord.Forbidden:
                            print(f"[Discord] Missing permissions to kick user {message.author}")
                            self.logger.error(f"Missing permissions to kick user {message.author}")
                            await message.channel.send(f"⚠️ Unable to kick {message.author.mention} due to missing permissions.")
                            # Reset warnings since we couldn't kick
                            self.reset_warnings(message.author.id)
                #logs the incident
                self.logger.warning(
                    f"Toxic content detected - User: {message.author}, "
                    f"Channel: {message.channel}, "
                    f"Content: {message.content}, "
                    f"Toxicity Score: {toxicity_score:.4f}, "
                    f"Warning Count: {warning_count}, "
                    f"Sentiment: {sentiment}, "
                    f"Sentiment Confidence: {sentiment_confidence:.4f}"
                )
                #logs the deletion and takes action if needed
                self.log_message(message, toxicity_score, action_taken, warning_count, sentiment, sentiment_confidence)
            except discord.Forbidden:
                print(f"[Discord] Missing permissions to moderate in channel {message.channel}")
                self.logger.error(f"Missing permissions to moderate in channel {message.channel}")
            except Exception as e:
                print(f"[Discord] Error handling toxic message: {str(e)}")
                self.logger.error(f"Error handling toxic message: {str(e)}")
        else:
            print(f"[Discord] Message from {message.author} passed toxicity check (score: {toxicity_score:.4f})")
            self.logger.info(f"Message from {message.author} passed toxicity check (score: {toxicity_score:.4f})")
            self.log_message(message, toxicity_score, 'none', warning_count, sentiment, sentiment_confidence)


#set up content moderator
moderator = ContentModerator()

#set up bot with command !
intents = discord.Intents.all()  # Enable all intents
bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    logger.info(f'Bot is ready! Logged in as {bot.user.name} (ID: {bot.user.id})')
    logger.info('------')
    for guild in bot.guilds:
        logger.info(f'Connected to guild: {guild.name}')
        bot_member = guild.get_member(bot.user.id)
        if bot_member:
            logger.info(f'Bot permissions: {bot_member.guild_permissions}')

@bot.event
async def on_member_join(member):
    moderator.reset_warnings(member.id)
    logger.info(f"Reset warnings for rejoining member {member.name} (ID: {member.id})")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    #logs the messages that have been recieved
    logger.info(f'Message from {message.author}: {message.content}')
    
    try:
        #checks the content of the messages
        await moderator.check_message(message)
        
        #processes the commands
        await bot.process_commands(message)
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        logger.error(traceback.format_exc())

@bot.command(name='hello')
async def hello(ctx):
    await ctx.send(f'Hello {ctx.author.mention}!')

def main():
    try:
        #gets the bot token from environment variables
        token = os.getenv('DISCORD_TOKEN')
        if not token:
            raise ValueError("No Discord token found in environment variables")
        
        #run the bot
        bot.run(token)
    except Exception as e:
        logger.error(f"Error running bot: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 