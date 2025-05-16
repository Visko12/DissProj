import twitchio
from twitchio.ext import commands
import logging
from models.multimodal_moderator import ContentModerator
import asyncio
from datetime import datetime, timedelta
import json
import os

#set up login
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#define a twitch bot that moderates messages
class TwitchModerator(commands.Bot):
    def __init__(self, token, client_id, nick, prefix, initial_channels):
        super().__init__(token=token, client_id=client_id, nick=nick, prefix=prefix, initial_channels=initial_channels)
        self.content_moderator = ContentModerator()
        self.user_warnings = {}
        self.warning_timeout = 3600
        self.warning_expiry = 86400
        
    async def event_ready(self):
        logger.info(f'Logged in as {self.nick}')
        logger.info(f'Connected to channels: {self.connected_channels}')
        
    async def event_message(self, message):
        if not message.author:
            print("event_message: author is None, skipping message.")
            return
        print(f"event_message: {message.content} from {message.author.name}")
        await self.process_message(message)
        
    def log_message(self, message, text_score=None, action_taken=None, warning_count=None):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'platform': 'twitch',
            'channel': getattr(message.channel, 'name', None),
            'author_name': getattr(message.author, 'name', None),
            'author_id': getattr(message.author, 'id', None),
            'message': getattr(message, 'content', None),
            'toxicity_score': text_score,
            'action_taken': action_taken,
            'warning_count': warning_count
        }
        os.makedirs('logs', exist_ok=True)
        with open('logs/message_log.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

    async def process_message(self, message):
        print(f"Received message: {message.content} from {message.author.name}")  # DEBUG

        try:
            text_score = self.content_moderator.moderate_text(message.content)
            print(f"Model score for '{message.content}': {text_score}")  # DEBUG

            action_taken = None
            warning_count = None
            if text_score > 0.9:
                action_taken = 'timeout'
                await self.take_action(message)
            elif text_score > 0.7:
                action_taken = 'warning'
                warning_count = len(self.user_warnings.get(message.author.id, [])) + 1
                await self.handle_warning(message)
            else:
                action_taken = 'none'

            self.log_message(message, text_score, action_taken, warning_count)

        except Exception as e:
            print(f"Exception in process_message: {e}")  # DEBUG
            logger.error(f"Error processing message: {str(e)}")
            
    async def handle_warning(self, message):
        user_id = message.author.id
        current_time = datetime.now()
        
        #set up warning list for a new user or update it if needed
        if user_id not in self.user_warnings:
            self.user_warnings[user_id] = []
            
        #remove any expired warnings
        self.user_warnings[user_id] = [
            w for w in self.user_warnings[user_id]
            if (current_time - w).total_seconds() < self.warning_expiry
        ]
        
        #add a new warning
        self.user_warnings[user_id].append(current_time)
        warning_count = len(self.user_warnings[user_id])
        
        #always send warning messages
        warning_msg = f"@{message.author.name} Warning {warning_count}/3: Your message was detected as toxic and may violate our community guidelines."
        await message.channel.send(warning_msg)
        
        #log warnings
        self.log_message(message, action_taken='warning', warning_count=warning_count)
        
        #if 3 warnings are issues it should take action
        if warning_count >= 3:
            await self.take_action(message)
            
    async def take_action(self, message, reason="Multiple violations of community guidelines"):
        try:
            #simulate a timeout by sending a /timeout command if the bot is a mod
            timeout_seconds = self.warning_timeout
            timeout_command = f"/timeout {message.author.name} {timeout_seconds} {reason}"
            await message.channel.send(timeout_command)
            
            #send a notification message
            timeout_msg = f"@{message.author.name} has been timed out for {timeout_seconds//60} minutes due to: {reason}"
            await message.channel.send(timeout_msg)
            
            #log timeout
            self.log_message(message, action_taken='timeout', warning_count=None)
            
            # Clear warnings after timeout
            if message.author.id in self.user_warnings:
                del self.user_warnings[message.author.id]
                
        except Exception as e:
            logger.error(f"Error handling violation: {str(e)}")
            
    def save_warnings(self):
        warnings_data = {
            user_id: [w.isoformat() for w in warnings]
            for user_id, warnings in self.user_warnings.items()
        }
        
        with open('data/twitch_warnings.json', 'w') as f:
            json.dump(warnings_data, f)
    
    def load_warnings(self):
        if os.path.exists('data/twitch_warnings.json'):
            with open('data/twitch_warnings.json', 'r') as f:
                warnings_data = json.load(f)
                
            self.user_warnings = {
                user_id: [datetime.fromisoformat(w) for w in warnings]
                for user_id, warnings in warnings_data.items()
            }

def main():
    #laod dotenv::environment
    from dotenv import load_dotenv
    load_dotenv()
    
    #get twitch credentials:: needed
    token = os.getenv('TWITCH_TOKEN')
    client_id = os.getenv('TWITCH_CLIENT_ID')
    nick = os.getenv('TWITCH_BOT_NICK')
    channels = os.getenv('TWITCH_CHANNELS', '').split(',')
    
    if not all([token, client_id, nick, channels]):
        logger.error("Missing required environment variables. Please check your .env file.")
        return
        
    #create and run the bot
    bot = TwitchModerator(
        token=token,
        client_id=client_id,
        nick=nick,
        prefix='!',
        initial_channels=channels
    )
    
    #load the saved warnings
    bot.load_warnings()
    
    #run the bot
    bot.run()

if __name__ == '__main__':
    main() 