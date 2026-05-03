# transcribe-engine

Self-hostable transcription engine for the [Transcribe](https://github.com/artriv/transcribe) webapp.

Bundles `whisper.cpp` (Vulkan / Metal / CPU) + `pyannote.audio` (CPU diarization) into a
single tray-icon binary per OS. Pairs with the hosted webapp via WebRTC; audio never
leaves your machine.

> Status: v0.1.0 in development (BYO-GPU milestone, Phase 7).
> Releases will appear at https://github.com/artriv/transcribe-engine/releases when ready.

## License

MIT — see [LICENSE](./LICENSE).
