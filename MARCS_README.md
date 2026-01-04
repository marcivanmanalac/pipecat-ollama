pipecat + ollama server

This is how I setup a local speech to speech server using ollama(LLM), pipecat(framework), local speech to text(STT), text-to-speech(TTS) models.

1. Follow quickstart guide all the way through uv sync. 
2. start tts server on terminal using `uv run tts_server.py`.
3. Start bot.py on another terminal using `uv run bot.py`.
4. Speak into mic once your terminal prints 
   `DEBUG    | pipecat.pipeline.task:_wait_for_pipeline_start:605 - PipelineTask#0: StartFrame#0 reached the end of the pipeline, pipeline is now ready.
`


FAQ:
1. You dont't need a seperate environment with this setup. uv takes care of that.
2. The used models are hardcoded in pipecat.examples.quickstart.bot.run_bot()

