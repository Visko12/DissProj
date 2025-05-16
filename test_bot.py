import unittest
import discord
from discord.ext import commands
import asyncio
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from bot import ContentModerator

#set up dotenv::environment variables
load_dotenv()

#configure logging
def setup_LogIn():
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/bot_test_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_LogIn()

#creates a mock message for testing
class MockMessage:
    def __init__(self, content, author):
        self.content = content
        self.author = author
        self.deleted = False
        self.warnings = []

    async def delete(self):
        self.deleted = True

    async def channel(self):
        return MockChannel()

#creates a mock author for testing
class MockAuthor:
    def __init__(self, name):
        self.name = name
        self.mention = f"@{name}"

#creates a mock channel for testing
class MockChannel:
    async def send(self, message):
        return message

class TestBot(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.moderator = ContentModerator()
        logger.info("Test environment set up complete")

    #test toxic messages to ensure they are classified as toxic
    def test_toxic_messages(self):
        toxic_messages = [
            "You're a complete idiot!",
            "I hate you so much!",
            "This is the worst thing ever!",
            "You're worthless and stupid!"
        ]
        
        logger.info("\nTesting toxic messages:")
        for message in toxic_messages:
            toxicity_score = self.moderator.check_toxicity(message)
            logger.info(f"Message: {message}")
            logger.info(f"Toxicity score: {toxicity_score:.4f}")
            self.assertGreater(toxicity_score, 0.7, 
                    f"Message should be classified as toxic: {message}")
            

    #test non-toxic messages to ensure they are classified as non-toxic
    def test_non_toxic_messages(self):
        non_toxic_messages = [
            "Hello, how are you?",
            "The weather is nice today.",
            "I like this server.",
            "Thanks for your help!"
        ]
        
        logger.info("\nTesting non-toxic messages:")
        for message in non_toxic_messages:
            toxicity_score = self.moderator.check_toxicity(message)
            logger.info(f"Message: {message}")
            logger.info(f"Toxicity score: {toxicity_score:.4f}")
            self.assertLess(toxicity_score, 0.7, 
            f"Message should not be classified as toxic: {message}")

    #test sentiment analysis to ensure it correctly identifies the sentiment of a message
    def test_sentiment_analysis(self):
        test_messages = [
            ("I love this server!", "positive"),
            ("This is terrible.", "negative"),
            ("The weather is okay.", "neutral"),
            ("I'm really happy with the results!", "positive"),
            ("I'm disappointed with the service.", "negative")
        ]
        
        logger.info("\nTesting sentiment analysis:")
        for message, expected_sentiment in test_messages:
            sentiment, confidence = self.moderator.check_sentiment(message)
            logger.info(f"Message: {message}")
            logger.info(f"Predicted sentiment: {sentiment} (confidence: {confidence:.4f})")
            self.assertEqual(sentiment, expected_sentiment,
                    f"Sentiment should be {expected_sentiment} for: {message}")

    #test message handling to ensure it correctly handles toxic and non-toxic messages
    async def test_message_handling(self):
        test_cases = [
            {
                "message": "You're an idiot!",
                "author": "TestUser1",
                "should_delete": True,
                "should_warn": False
            },
            {
                "message": "I really don't like this server.",
                "author": "TestUser2",
                "should_delete": False,
                "should_warn": True
            },
            {
                "message": "Hello, how are you?",
                "author": "TestUser3",
                "should_delete": False,
                "should_warn": False
            }
        ]
        
        logger.info("\nTesting complete message handling:")
        for case in test_cases:
            message = MockMessage(case["message"], MockAuthor(case["author"]))
            
            #checks the sentiment
            sentiment, sentiment_confidence = self.moderator.check_sentiment(message.content)
            logger.info(f"Sentiment: {sentiment} (confidence: {sentiment_confidence:.4f})")
            
            #checks the toxicity
            toxicity_score = self.moderator.check_toxicity(message.content)
            logger.info(f"\nMessage: {message.content}")
            logger.info(f"Author: {message.author.name}")
            logger.info(f"Toxicity score: {toxicity_score:.4f}")
            
            #simulates message handling
            if toxicity_score > 0.7:
                await message.delete()
                warning = f"{message.author.mention} Your message was removed due to toxic content."
                message.warnings.append(warning)
            
            elif sentiment == 'negative' and sentiment_confidence > 0.7:
                warning = f"{message.author.mention} Your message appears to be negative."
                message.warnings.append(warning)
            
            #verifies the results
            self.assertEqual(message.deleted, case["should_delete"],
                    f"Message deletion status incorrect for: {message.content}")
            self.assertEqual(len(message.warnings) > 0, case["should_warn"],
                    f"Warning status incorrect for: {message.content}")

def run_tests():
    logger.info("Starting bot tests...")
    
    #creates a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBot)
    
    #runs the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    #logs the results
    logger.info("\nTest Results:")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    #runs async tests
    loop = asyncio.get_event_loop()
    loop.run_until_complete(TestBot().test_message_handling())
    
    #runs all tests
    success = run_tests()
    
    #exits with appropriate status code
    exit(0 if success else 1)
    