"""
FastAPI service for the LiveKit Restaurant Reservation Bot with Twilio integration
"""

import asyncio
import os
import subprocess
from contextlib import asynccontextmanager

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

load_dotenv(override=True)

# ------------ Configuration ------------ #
REQUIRED_ENV_VARS = [
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "LIVEKIT_URL",
    "LIVEKIT_SIP_ENDPOINT",  # This should be your LiveKit SIP endpoint
    "OPENAI_API_KEY",
    "DEEPGRAM_API_KEY",
    "CARTESIA_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN"
]

# Check for required environment variables
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")

# Initialize Twilio client
twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    app.state.aiohttp_session = aiohttp.ClientSession()
    yield
    # Cleanup
    await app.state.aiohttp_session.close()

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/twilio_start_bot", response_class=PlainTextResponse)
async def twilio_start_bot(request: Request):
    """Handle incoming Twilio calls by connecting them to a LiveKit room."""
    try:
        logger.info("Received Twilio webhook call")
        form = await request.form()
        logger.debug(f"Form data: {dict(form)}")
        
        call_sid = form.get("CallSid")
        account_sid = form.get("AccountSid")

        if not call_sid or not account_sid:
            logger.error("Missing CallSid or AccountSid in form data")
            raise HTTPException(status_code=400, detail="Missing CallSid or AccountSid")

        logger.info(f"Call details - ID: {call_sid}, Account: {account_sid}")

        # Generate a unique room name for this call
        room_name = f"call_{call_sid}"

        # Start the restaurant bot as a separate process
        bot_proc = f"python food_ordering_livekit.py --room {room_name}"
        logger.info(f"Starting bot process: {bot_proc}")
        subprocess.Popen(bot_proc.split())

        # Create Twilio response
        response = VoiceResponse()
        
        # Add a pause for the bot to initialize
        response.pause(length=4)
        
        # Connect to LiveKit's SIP endpoint
        sip_uri = os.getenv("LIVEKIT_SIP_ENDPOINT")
        
        # Add custom SIP headers that LiveKit will use to route the call
        sip_headers = {
            "X-Room-Name": room_name,
            "X-Participant-Identity": f"caller_{call_sid}"
        }
        
        # Then dial the SIP URI with headers
        dial = response.dial(timeout=20)
        dial.sip(sip_uri, sip_headers=sip_headers)

        return PlainTextResponse(str(response))

    except Exception as e:
        logger.error(f"Error in twilio_start_bot: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
