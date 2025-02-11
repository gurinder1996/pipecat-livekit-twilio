import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from twilio.rest import Client
from livekit import api
from deepgram import LiveOptions

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    EndFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.livekit import LiveKitParams, LiveKitTransport
from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Initialize Twilio client
twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilioclient = Client(twilio_account_sid, twilio_auth_token)

# Constants
DESIRED_SAMPLE_RATE = 16000

# Type definitions for Flow Results
class PartySizeResult(FlowResult):
    size: int

class TimeResult(FlowResult):
    time: str

# Function handlers for the flow
async def record_party_size(args: FlowArgs) -> FlowResult:
    """Handler for recording party size."""
    size = args["size"]
    # In a real app, this would store the reservation details
    return PartySizeResult(size=size)

async def record_time(args: FlowArgs) -> FlowResult:
    """Handler for recording reservation time."""
    time = args["time"]
    # In a real app, this would validate availability and store the time
    return TimeResult(time=time)

# Flow configuration for the reservation conversation
flow_config: FlowConfig = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "role_messages": [
                {
                    "role": "system",
                    "content": "You are a restaurant reservation assistant for La Maison, an upscale French restaurant. You must ALWAYS use one of the available functions to progress the conversation. This is a phone conversations and your responses will be converted to audio. Avoid outputting special characters and emojis. Be causal and friendly.",
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": "Warmly greet the customer and ask how many people are in their party.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_party_size",
                        "handler": record_party_size,
                        "description": "Record the number of people in the party",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "size": {"type": "integer", "minimum": 1, "maximum": 12}
                            },
                            "required": ["size"],
                        },
                        "transition_to": "get_time",
                    },
                },
            ],
        },
        "get_time": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "Ask what time they'd like to dine. Restaurant is open 5 PM to 10 PM. After they provide a time, confirm it's within operating hours before recording. Use 24-hour format for internal recording (e.g., 17:00 for 5 PM).",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_time",
                        "handler": record_time,
                        "description": "Record the reservation time",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "time": {"type": "string", "pattern": "^([01]\\d|2[0-3]):[0-5]\\d$"}
                            },
                            "required": ["time"],
                        },
                        "transition_to": "end",
                    },
                },
            ],
        },
        "end": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "Thank the customer for their reservation and confirm all the details (party size and time). Then end the conversation.",
                }
            ],
            "functions": [],
        },
    },
}

def generate_token(room_name: str, participant_name: str, api_key: str, api_secret: str) -> str:
    token = api.AccessToken(api_key, api_secret)
    token.with_identity(participant_name).with_name(participant_name).with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
        )
    )
    return token.to_jwt()

def generate_token_with_agent(room_name: str, participant_name: str, api_key: str, api_secret: str) -> str:
    token = api.AccessToken(api_key, api_secret)
    token.with_identity(participant_name).with_name(participant_name).with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
            agent=True,  # This makes livekit client know agent has joined
        )
    )
    return token.to_jwt()

async def join_room_as_agent(room_name: str):
    """Join a LiveKit room as an agent."""
    try:
        url = os.getenv("LIVEKIT_URL")
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")

        if not all([url, api_key, api_secret]):
            raise ValueError(
                "LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in environment variables."
            )

        # Generate token for the bot
        token = generate_token_with_agent(room_name, "Restaurant Bot", api_key, api_secret)

        async with aiohttp.ClientSession() as session:
            transport = LiveKitTransport(
                url=url,
                token=token,
                room_name=room_name,
                params=LiveKitParams(
                    audio_in_channels=1,
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    audio_in_sample_rate=DESIRED_SAMPLE_RATE,
                    audio_out_sample_rate=DESIRED_SAMPLE_RATE,
                    vad_analyzer=SileroVADAnalyzer(),
                    vad_enabled=True,
                    vad_audio_passthrough=True,
                ),
            )

            stt = DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(
                    sample_rate=DESIRED_SAMPLE_RATE,
                    vad_events=True,
                ),
            )

            tts = CartesiaTTSService(
                api_key=os.getenv("CARTESIA_API_KEY"),
                voice_id="794f9389-aac1-45b6-b726-9d9369183238",
                sample_rate=DESIRED_SAMPLE_RATE,
            )

            llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

            context = OpenAILLMContext()
            context_aggregator = llm.create_context_aggregator(context)

            # Create pipeline with all components
            pipeline = Pipeline(
                [
                    transport.input(),
                    stt,
                    context_aggregator.user(),
                    llm,
                    tts,
                    transport.output(),
                    context_aggregator.assistant(),
                ]
            )

            # Create task with pipeline
            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    allow_interruptions=True,
                    enable_metrics=True,
                    enable_usage_metrics=True
                )
            )

            # Initialize flow manager with task, llm, tts, and flow_config
            flow_manager = FlowManager(
                task=task,
                llm=llm,
                tts=tts,
                flow_config=flow_config
            )

            # Register event handlers
            @transport.event_handler("on_first_participant_joined")
            async def on_first_participant_joined(transport, participant):
                logger.debug("Initializing flow")
                await flow_manager.initialize()
                logger.debug("Starting conversation")
                await task.queue_frames([context_aggregator.user().get_context_frame()])

            @transport.event_handler("on_data_received")
            async def on_data_received(transport, data, participant_id):
                logger.info(f"Received data from participant {participant_id}: {data}")
                json_data = json.loads(data)
                await task.queue_frames(
                    [
                        BotInterruptionFrame(),
                        UserStartedSpeakingFrame(),
                        TranscriptionFrame(json_data["text"]),
                        UserStoppedSpeakingFrame(),
                    ]
                )

            # Create runner and run
            try:
                runner = PipelineRunner()
                await runner.run(task)
            except NotImplementedError:
                # Windows doesn't support signal handlers, so we'll just run the task directly
                await task.run()
            except Exception as e:
                logger.error(f"Error running pipeline: {str(e)}")
                raise
    except Exception as e:
        logger.error(f"Error joining room as agent: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LiveKit Restaurant Bot")
    parser.add_argument("--room", required=True, help="Name of the LiveKit room to join")
    args = parser.parse_args()
    
    asyncio.run(join_room_as_agent(args.room))
