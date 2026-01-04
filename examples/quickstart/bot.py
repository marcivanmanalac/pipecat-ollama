#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat Cloud.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

Run the bot using::

    uv run bot.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
#from pipecat.services.cartesia.tts import CartesiaTTSService
#from pipecat.services.deepgram.stt import DeepgramSTTService
#from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

#ADDED STT, TTS, & LLM Services
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.piper.tts import PiperTTSService
#ADDED 100% local audio
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
#ADDED TTS local web server setup
import aiohttp

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    
    # Create a session for the HTTP requests
    async with aiohttp.ClientSession() as session:
        
        # 1. Configure LLM (Ollama)
        llm = OLLamaLLMService(
            model="llama3.2:1b",
            base_url="http://localhost:11434/v1"
        )

        # 2. Configure STT (Whisper - Local)
        #stt = WhisperSTTService() 
        # Force Whisper to use the CPU to avoid NVIDIA driver errors
        stt = WhisperSTTService(device="cpu")

        # 3. Configure TTS (Piper - HTTP Client)
        tts = PiperTTSService(
            aiohttp_session=session,  # <--- REQUIRED
            base_url="http://localhost:5000", # <--- REQUIRED
            voice="en_US-lessac-medium"
        )

        messages = [
            {
                "role": "system",
                "content": "You are a friendly AI assistant. Keep answers short and conversational.",
            },
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(context)

        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")
            messages.append({"role": "system", "content": "Say hello."})
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)

async def bot(runner_args: RunnerArguments):
    # ... existing code ...

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        # ADD THIS BLOCK FOR LOCAL AUDIO:
        "local": lambda: LocalAudioTransportParams(
            audio_out_sample_rate=16000,
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)



async def main():
    """Manual entry point for running locally without the CLI parser."""
    
    # 1. Define the Local Transport directly (No 'create_transport' helper needed)
    #    Note: We use the 'TransportParams' suffix you fixed earlier
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_out_sample_rate=16000,
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        )
    )

    # 2. Create a dummy object for runner_args since run_bot expects it
    #    (This just tells the runner to handle Ctrl+C strictly)
    class DummyRunnerArgs:
        handle_sigint = True
    
    runner_args = DummyRunnerArgs()

    # 3. Run the bot
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    import asyncio
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass