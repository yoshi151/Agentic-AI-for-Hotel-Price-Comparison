from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
from browser_use import Agent
import os
import asyncio
import logging

# Load .env variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Pass API key directly to the model
llm = ChatGoogleGenerativeAI(
   model="gemini-2.0-flash",  # use full model path
   google_api_key=api_key
)
#llm = ChatOllama(
#    model="deepseek-r1:14b",  # llama3
#    temperature=1.0,
#)

# Create your Agent
agent = Agent(
    task="""Go to 'https://www.agoda.com/th-th/' 
    and Find available hotel accommodations.
    Location: Phuket, Thailand  
    Check-in Date: 15 May  
    Check-out Date: 18 May  
    Number of Rooms: 1  
    Number of Adults: 3 Adults
    Objective: Search for hotels that are available for the given dates and party size. Provide the hotel names, prices""",
    llm=llm
)

async def main():
    logger.info("Agent starting task...")
    try:
        await agent.run()
        logger.info("Agent completed task.")
    except Exception as e:
        logger.error(f"Error during agent execution: {e}", exc_info=True)

if __name__ == '__main__':
    asyncio.run(main())

"""
SET GEMINI_API_KEY AIzaSyAyejwfHvn6YTmugdZzBE3PNOECbct6MwY 


curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyAyejwfHvn6YTmugdZzBE3PNOECbct6MwY" -H 'Content-Type: application/json' -X POST -d '{"contents": [{"parts": [{"text": "Write a story about a magic backpack."}]}]}'
"""
