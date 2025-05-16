import os
from dotenv import load_dotenv
from .discord_moderator import run_bot
import logging

#setup LogIn
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    #load dotenv::environment
    load_dotenv()
    
    #get discord token
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("No Discord token found. Please set DISCORD_TOKEN in .env file")
        return
    
    #run the bot
    logger.info("Starting Discord bot....")
    try:
        run_bot(token)
    except Exception as e:
        logger.error(f"Error running Discord bot: {e}")
        raise

if __name__ == '__main__':
    main() 