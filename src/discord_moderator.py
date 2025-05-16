import discord
from discord.ext import commands
import logging
from bot import ContentModerator
import os
from dotenv import load_dotenv
from PIL import Image, ImageFilter
import io
import torch
from src.models.multimodal_moderator import MultiModalModerator

#set up login
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#load dotenv::environment
load_dotenv()
moderator = ContentModerator()

#check if an image is nsfw using the moderator
image_moderator = MultiModalModerator()
image_moderator.eval()

def is_nsfw_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        score = image_moderator.process_image(image_tensor)
        #if score > 0.3, consider nsfw (lowered threshold) for testing....
        return float(score[0][0]) > 0.3

async def blur_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    blurred = image.filter(ImageFilter.GaussianBlur(12))
    output = io.BytesIO()
    blurred.save(output, format='PNG')
    output.seek(0)
    return output

class ViewImageButton(discord.ui.View):
    def __init__(self, original_image_bytes, user):
        super().__init__(timeout=60)
        self.original_image_bytes = original_image_bytes
        self.user = user

    @discord.ui.button(label="View Image", style=discord.ButtonStyle.primary)
    async def view_image(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id == self.user.id:
            file = discord.File(io.BytesIO(self.original_image_bytes), filename="original_image.png")
            await interaction.user.send("Here is the original image you requested:", file=file)
            await interaction.response.send_message("The original image has been sent to your DMs.", ephemeral=True)
        else:
            await interaction.response.send_message("You are not allowed to view this image.", ephemeral=True)

class ModerationBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.all()
        super().__init__(command_prefix='!', intents=intents)

    async def on_ready(self):
        logger.info(f'Bot is ready! Logged in as {self.user} (ID: {self.user.id})')
        logger.info('------')
        for guild in self.guilds:
            logger.info(f'Connected to guild: {guild.name}')
            bot_member = guild.get_member(self.user.id)
            if bot_member:
                logger.info(f'Bot permissions: {bot_member.guild_permissions}')

    async def on_member_join(self, member):
        moderator.reset_warnings(member.id)
        logger.info(f"Reset warnings for rejoining member {member.name} (ID: {member.id})")

    async def on_message(self, message):
        if message.author == self.user:
            return
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                image_bytes = await attachment.read()
                if is_nsfw_image(image_bytes):
                    await message.delete()
                    blurred_bytes = await blur_image(image_bytes)
                    file = discord.File(blurred_bytes, filename="blurred_image.png")
                    view = ViewImageButton(image_bytes, message.author)
                    await message.channel.send(
                        f"⚠️ NSFW content detected and blurred. Click the button below to view the original image (only available to the sender).",
                        file=file,
                        view=view
                    )
                    logger.info(f"Blurred and reposted NSFW image from {message.author}")
                    return
        logger.info(f'Message from {message.author}: {message.content}')
        try:
            await moderator.check_message(message)
            await self.process_commands(message)
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

def run_bot(token):
    bot = ModerationBot()
    bot.run(token) 