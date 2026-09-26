package com.unamentis.core.session

import android.util.Log
import com.unamentis.core.audio.AudioEngine
import com.unamentis.core.config.RecordingMode
import com.unamentis.core.curriculum.CurriculumEngine
import com.unamentis.data.model.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.*
import java.util.concurrent.atomic.AtomicInteger

/**
 * Dependencies for SessionManager.
 *
 * Groups all required services for the session orchestrator.
 *
 * @property audioEngine Audio I/O engine
 * @property vadService Voice activity detection
 * @property sttService Speech-to-text provider
 * @property ttsService Text-to-speech provider
 * @property llmService Language model provider
 * @property curriculumEngine Curriculum progress tracking
 */
data class SessionDependencies(
    val audioEngine: AudioEngine,
    val vadService: VADService,
    val sttService: STTService,
    val ttsService: TTSService,
    val llmService: LLMService,
    val curriculumEngine: CurriculumEngine,
)

/**
 * Core session orchestrator managing voice conversation state machine.
 *
 * State Machine:
 * - IDLE: No active session
 * - USER_SPEAKING: User is speaking (VAD detected speech)
 * - PROCESSING_UTTERANCE: STT finalizing transcription
 * - AI_THINKING: LLM generating response
 * - AI_SPEAKING: TTS playing audio
 * - INTERRUPTED: User barged in during AI speech
 * - PAUSED: Session paused by user
 * - ERROR: Unrecoverable error occurred
 *
 * Conversation Flow:
 * 1. VAD detects speech start → USER_SPEAKING
 * 2. VAD detects 1.5s silence → PROCESSING_UTTERANCE
 * 3. STT finalizes → AI_THINKING
 * 4. LLM starts streaming → AI_SPEAKING (concurrent with TTS)
 * 5. TTS completes → Back to listening (IDLE or USER_SPEAKING)
 *
 * Barge-in:
 * - User can interrupt AI speech after 600ms confirmation window
 * - AI audio stops, LLM generation cancelled
 * - New user utterance processed
 *
 * @property dependencies Required services bundle
 * @property scope Coroutine scope for session lifecycle
 */
class SessionManager(
    private val dependencies: SessionDependencies,
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Default + SupervisorJob()),
) {
    private val audioEngine get() = dependencies.audioEngine
    private val vadService get() = dependencies.vadService
    private val sttService get() = dependencies.sttService
    private val ttsService get() = dependencies.ttsService
    private val llmService get() = dependencies.llmService
    private val curriculumEngine get() = dependencies.curriculumEngine
    private val _sessionState = MutableStateFlow<SessionState>(SessionState.IDLE)
    val sessionState: StateFlow<SessionState> = _sessionState.asStateFlow()

    private val _currentSession = MutableStateFlow<Session?>(null)
    val currentSession: StateFlow<Session?> = _currentSession.asStateFlow()

    private val _transcript = MutableStateFlow<List<TranscriptEntry>>(emptyList())
    val transcript: StateFlow<List<TranscriptEntry>> = _transcript.asStateFlow()

    private val _metrics = MutableStateFlow(SessionMetrics())
    val metrics: StateFlow<SessionMetrics> = _metrics.asStateFlow()

    // Recording mode configuration
    private val _recordingMode = MutableStateFlow(RecordingMode.VAD)
    val recordingMode: StateFlow<RecordingMode> = _recordingMode.asStateFlow()

    // Manual recording state (for PUSH_TO_TALK and TOGGLE modes)
    private val _isManuallyRecording = MutableStateFlow(false)
    val isManuallyRecording: StateFlow<Boolean> = _isManuallyRecording.asStateFlow()

    // Mute state - when true, microphone input is ignored (no VAD/STT processing)
    private val _isMuted = MutableStateFlow(false)
    val isMuted: StateFlow<Boolean> = _isMuted.asStateFlow()

    /**
     * Set the microphone muted state.
     *
     * When muted:
     * - VAD does not trigger speech detection
     * - STT is not started for new utterances
     * - User can still hear AI responses
     * - Barge-in is disabled
     *
     * @param muted true to mute, false to unmute
     */
    fun setMuted(muted: Boolean) {
        _isMuted.value = muted
        Log.i("SessionManager", "Microphone muted: $muted")
    }

    // ==========================================================================
    // CURRICULUM STATE FLOWS (exposed for UI)
    // ==========================================================================

    /**
     * Whether a curriculum is loaded (curriculum mode vs free conversation).
     */
    val isCurriculumMode: StateFlow<Boolean> =
        curriculumEngine.currentCurriculum
            .map { it != null }
            .stateIn(scope, SharingStarted.Eagerly, false)

    /**
     * Current segment index within the topic (0-based).
     */
    val currentSegmentIndex: StateFlow<Int> = curriculumEngine.currentSegmentIndex

    /**
     * Total number of segments in the current topic.
     */
    val totalSegments: StateFlow<Int> =
        curriculumEngine.currentTopic
            .map { it?.transcript?.size ?: 0 }
            .stateIn(scope, SharingStarted.Eagerly, 0)

    /**
     * Whether there is a next topic available in the curriculum.
     */
    val hasNextTopic: StateFlow<Boolean> =
        combine(
            curriculumEngine.currentCurriculum,
            curriculumEngine.currentTopic,
        ) { curriculum, currentTopic ->
            if (curriculum == null || currentTopic == null) {
                false
            } else {
                val currentIndex = curriculum.topics.indexOfFirst { it.id == currentTopic.id }
                currentIndex >= 0 && currentIndex + 1 < curriculum.topics.size
            }
        }.stateIn(scope, SharingStarted.Eagerly, false)

    /**
     * Current audio level in dB (for VU meter visualization).
     * Range is typically -60 (silence) to 0 (maximum).
     */
    val audioLevelDb: StateFlow<Float> =
        audioEngine.audioLevel
            .map { level ->
                // Convert RMS to dB, with floor at -60dB
                if (level.rms > 0) {
                    (20 * kotlin.math.log10(level.rms)).coerceIn(-60f, 0f)
                } else {
                    -60f
                }
            }
            .stateIn(scope, SharingStarted.Eagerly, -60f)

    // Conversation history for LLM context
    private val conversationHistory = mutableListOf<LLMMessage>()

    // Turn timing
    private var currentTurnStartTime = 0L
    private var userSpeechStartTime = 0L
    private var lastSpeechDetectedTime = 0L

    // Barge-in handling
    private var aiSpeechStartTime = 0L
    private val bargeInConfirmationWindowMs = 600L

    // Active jobs for cancellation
    private var sttJob: Job? = null
    private var llmJob: Job? = null
    private var vadJob: Job? = null

    // TTS jobs collection - tracks ALL active TTS jobs to prevent race conditions
    // Each call to synthesizeAndPlay() adds to this list instead of overwriting
    private val ttsJobs = mutableListOf<Job>()

    // Configuration
    private val silenceThresholdMs = 1500L // 1.5 seconds of silence to finalize utterance

    // Speech debouncing - prevent false triggers from brief noise
    private val minimumSpeechFrames = 3 // ~36ms at 12ms frames
    private val consecutiveSpeechFrames = AtomicInteger(0)

    // Audio capture resume configuration
    private val audioResumeDelayMs = 100L // Delay to allow STT microphone release
    private val maxAudioResumeRetries = 3
    private val audioResumeRetryDelayMs = 50L

    /**
     * Set the recording mode for this session.
     *
     * @param mode The recording mode to use
     */
    fun setRecordingMode(mode: RecordingMode) {
        _recordingMode.value = mode
        Log.i("SessionManager", "Recording mode set to: ${mode.name}")
    }

    /**
     * Start a new session.
     *
     * @param curriculumId Optional curriculum to load
     * @param topicId Optional specific topic to start with
     * @param recordingMode Recording mode to use (default: VAD)
     */
    suspend fun startSession(
        curriculumId: String? = null,
        topicId: String? = null,
        recordingMode: RecordingMode = RecordingMode.VAD,
    ) {
        // DEBUG: Log at ERROR level to ensure visibility
        Log.e("SessionManager", "SESSION_DEBUG: startSession called, mode=$recordingMode, curriculum=$curriculumId")

        _recordingMode.value = recordingMode
        // Block if state is not IDLE or if there's already an active session
        if (_sessionState.value != SessionState.IDLE || _currentSession.value != null) {
            Log.w("SessionManager", "Cannot start session: already active")
            return
        }

        Log.i("SessionManager", "Starting new session")

        // Create new session
        val session =
            Session(
                id = UUID.randomUUID().toString(),
                curriculumId = curriculumId,
                topicId = topicId,
                startTime = System.currentTimeMillis(),
                endTime = null,
                turnCount = 0,
            )
        _currentSession.value = session

        // Load curriculum if specified
        curriculumId?.let {
            curriculumEngine.loadCurriculum(it, topicId)
        }

        // Initialize system prompt
        val systemPrompt = buildSystemPrompt()
        conversationHistory.clear()
        conversationHistory.add(LLMMessage(role = "system", content = systemPrompt))

        // Start audio capture
        val captureStarted =
            audioEngine.startCapture { audioSamples ->
                scope.launch {
                    processAudioFrame(audioSamples)
                }
            }

        if (!captureStarted) {
            _sessionState.value = SessionState.ERROR
            Log.e("SessionManager", "Failed to start audio capture")
            return
        }

        _sessionState.value = SessionState.IDLE
        currentTurnStartTime = System.currentTimeMillis()

        Log.i("SessionManager", "Session started: ${session.id}")
    }

    /**
     * Pause the current session.
     */
    suspend fun pauseSession() {
        if (_sessionState.value == SessionState.PAUSED) return

        Log.i("SessionManager", "Pausing session")

        // Cancel all active operations
        cancelActiveOperations()

        // Stop audio
        audioEngine.stopCapture()
        audioEngine.stopPlayback()

        _sessionState.value = SessionState.PAUSED
    }

    /**
     * Resume a paused session.
     */
    suspend fun resumeSession() {
        if (_sessionState.value != SessionState.PAUSED) return

        Log.i("SessionManager", "Resuming session")

        // Restart audio capture
        audioEngine.startCapture { audioSamples ->
            scope.launch {
                processAudioFrame(audioSamples)
            }
        }

        _sessionState.value = SessionState.IDLE
    }

    /**
     * Stop the current session.
     */
    suspend fun stopSession() {
        Log.i("SessionManager", "Stopping session")

        // Cancel all active operations
        cancelActiveOperations()

        // Stop audio
        audioEngine.stopCapture()
        audioEngine.stopPlayback()

        // Finalize session
        _currentSession.value?.let { session ->
            val finalSession =
                session.copy(
                    endTime = System.currentTimeMillis(),
                )
            _currentSession.value = finalSession

            // TODO: Save to database via SessionRepository
        }

        _sessionState.value = SessionState.IDLE
        _currentSession.value = null

        Log.i("SessionManager", "Session stopped")
    }

    /**
     * Start manual recording (for PUSH_TO_TALK and TOGGLE modes).
     *
     * In PUSH_TO_TALK mode: Call this when user presses the mic button.
     * In TOGGLE mode: Call this when user taps to start recording.
     */
    suspend fun startManualRecording() {
        if (_recordingMode.value == RecordingMode.VAD) {
            Log.w("SessionManager", "startManualRecording called but mode is VAD")
            return
        }

        if (_sessionState.value != SessionState.IDLE) {
            Log.w("SessionManager", "Cannot start manual recording: session not idle")
            return
        }

        Log.i("SessionManager", "Starting manual recording")
        _isManuallyRecording.value = true
        userSpeechStartTime = System.currentTimeMillis()
        _sessionState.value = SessionState.USER_SPEAKING
        startSTTStreaming()
    }

    /**
     * Stop manual recording (for PUSH_TO_TALK and TOGGLE modes).
     *
     * In PUSH_TO_TALK mode: Call this when user releases the mic button.
     * In TOGGLE mode: Call this when user taps to stop recording.
     */
    suspend fun stopManualRecording() {
        if (_recordingMode.value == RecordingMode.VAD) {
            Log.w("SessionManager", "stopManualRecording called but mode is VAD")
            return
        }

        if (_sessionState.value != SessionState.USER_SPEAKING) {
            Log.w("SessionManager", "Cannot stop manual recording: not currently recording")
            return
        }

        Log.i("SessionManager", "Stopping manual recording")
        _isManuallyRecording.value = false
        _sessionState.value = SessionState.PROCESSING_UTTERANCE
        finalizeSTT()
    }

    /**
     * Toggle manual recording (convenience method for TOGGLE mode).
     */
    suspend fun toggleManualRecording() {
        if (_isManuallyRecording.value) {
            stopManualRecording()
        } else {
            startManualRecording()
        }
    }

    // ==========================================================================
    // CURRICULUM PLAYBACK CONTROLS
    // ==========================================================================

    /**
     * Go back one segment in curriculum playback.
     *
     * @return Result.success if segment was changed, Result.failure if at first segment or no topic loaded
     */
    suspend fun goBackSegment(): Result<Unit> {
        Log.i("SessionManager", "Go back segment requested")

        if (curriculumEngine.currentTopic.value == null) {
            Log.w("SessionManager", "Cannot go back: no topic loaded")
            return Result.failure(IllegalStateException("No topic loaded"))
        }

        val currentIndex = curriculumEngine.currentSegmentIndex.value
        if (currentIndex <= 0) {
            Log.w("SessionManager", "Cannot go back: already at first segment")
            return Result.failure(IllegalStateException("Already at first segment"))
        }

        curriculumEngine.previousSegment()
        Log.i("SessionManager", "Moved back to segment ${curriculumEngine.currentSegmentIndex.value}")
        return Result.success(Unit)
    }

    /**
     * Replay the current topic from the beginning.
     *
     * @return Result.success if topic was reset to beginning, Result.failure if no topic loaded
     */
    suspend fun replayTopic(): Result<Unit> {
        Log.i("SessionManager", "Replay topic requested")

        if (curriculumEngine.currentTopic.value == null) {
            Log.w("SessionManager", "Cannot replay: no topic loaded")
            return Result.failure(IllegalStateException("No topic loaded"))
        }

        curriculumEngine.goToSegment(0)
        Log.i("SessionManager", "Replaying topic from beginning")
        return Result.success(Unit)
    }

    /**
     * Skip to the next topic in the curriculum.
     *
     * @return Result.success if advanced to next topic, Result.failure if at last topic or no curriculum loaded
     */
    @Suppress("ReturnCount")
    suspend fun nextTopic(): Result<Unit> {
        Log.i("SessionManager", "Next topic requested")

        val curriculum = curriculumEngine.currentCurriculum.value
        val currentTopic = curriculumEngine.currentTopic.value

        // Validate prerequisites
        val error =
            when {
                curriculum == null -> "No curriculum loaded"
                currentTopic == null -> "No topic loaded"
                else -> null
            }
        if (error != null) {
            Log.w("SessionManager", "Cannot skip to next topic: $error")
            return Result.failure(IllegalStateException(error))
        }

        // Find current position and validate
        val currentIndex = curriculum!!.topics.indexOfFirst { it.id == currentTopic!!.id }
        val nextIndex = currentIndex + 1
        val indexError =
            when {
                currentIndex == -1 -> "Current topic not found in curriculum"
                nextIndex >= curriculum.topics.size -> "Already at last topic"
                else -> null
            }
        if (indexError != null) {
            Log.w("SessionManager", "Cannot skip to next topic: $indexError")
            return Result.failure(IllegalStateException(indexError))
        }

        // Navigate to next topic
        val nextTopicData = curriculum.topics[nextIndex]
        curriculumEngine.loadTopic(nextTopicData.id)
        Log.i("SessionManager", "Skipped to next topic: ${nextTopicData.title}")
        return Result.success(Unit)
    }

    /**
     * Send a text message (for testing or text-based interaction).
     */
    suspend fun sendTextMessage(text: String) {
        if (_sessionState.value != SessionState.IDLE) {
            Log.w("SessionManager", "Cannot send text: session busy")
            return
        }

        Log.i("SessionManager", "Processing text message: $text")

        // Add to transcript
        val userEntry =
            TranscriptEntry(
                id = UUID.randomUUID().toString(),
                sessionId = _currentSession.value?.id ?: "",
                role = "user",
                text = text,
                timestamp = System.currentTimeMillis(),
            )
        addToTranscript(userEntry)

        // Add to conversation history
        conversationHistory.add(LLMMessage(role = "user", content = text))

        // Get AI response
        processLLMResponse()
    }

    // Counter for periodic logging
    private val audioFrameCount = AtomicInteger(0)

    // Audio gain for amplifying weak microphone signal
    // Oboe with VoiceRecognition preset often produces very quiet audio (~0.001 RMS)
    // We need to amplify to typical speech levels (~0.1 RMS) for VAD to work correctly
    private val audioGain = 100f

    /**
     * Process audio frame from AudioEngine.
     *
     * Contains debug logging for diagnosing VAD issues.
     */
    @Suppress("CyclomaticComplexMethod", "LongMethod")
    private suspend fun processAudioFrame(audioSamples: FloatArray) {
        val frameNum = audioFrameCount.incrementAndGet()

        // Skip all audio processing when muted - user can still hear AI responses
        if (_isMuted.value) {
            return
        }

        // DEBUG: Log every 500 frames at ERROR level to ensure visibility
        if (frameNum % 500 == 1) {
            Log.e(
                "SessionManager",
                "AUDIO_DEBUG: Frame #$frameNum, " +
                    "samples=${audioSamples.size}, state=${_sessionState.value}",
            )
        }

        // Apply gain to amplify weak microphone signal for VAD
        // The amplified samples are used for VAD only; UI level meters use original samples
        val amplifiedSamples =
            FloatArray(audioSamples.size) { i ->
                (audioSamples[i] * audioGain).coerceIn(-1f, 1f)
            }

        // Run VAD on amplified audio
        val vadResult = vadService.processAudio(amplifiedSamples)

        // DEBUG: Log VAD result at ERROR level periodically
        if (frameNum % 500 == 1) {
            var sumSquares = 0f
            for (sample in amplifiedSamples) {
                sumSquares += sample * sample
            }
            val rms = kotlin.math.sqrt(sumSquares / amplifiedSamples.size)
            Log.e(
                "SessionManager",
                "VAD_DEBUG: isSpeech=${vadResult.isSpeech}, " +
                    "conf=${vadResult.confidence}, rms=${"%.4f".format(rms)}",
            )
        }

        // Log periodically to avoid spam (every 100 frames = ~1.2 seconds at 12ms frames)
        if (frameNum % 100 == 0) {
            // Calculate RMS amplitude for debugging (on amplified samples)
            var sumSquares = 0f
            var maxAbs = 0f
            for (sample in amplifiedSamples) {
                sumSquares += sample * sample
                val abs = kotlin.math.abs(sample)
                if (abs > maxAbs) maxAbs = abs
            }
            val rms = kotlin.math.sqrt(sumSquares / amplifiedSamples.size)

            Log.d(
                "SessionManager",
                "Audio frame #$frameNum: state=${_sessionState.value}, " +
                    "mode=${_recordingMode.value}, VAD(isSpeech=${vadResult.isSpeech}, " +
                    "confidence=${"%.3f".format(vadResult.confidence)}), " +
                    "amplified(rms=${"%.4f".format(rms)}, peak=${"%.4f".format(maxAbs)}, gain=$audioGain)",
            )
        }

        // Log when speech is first detected
        if (vadResult.isSpeech && _sessionState.value == SessionState.IDLE) {
            Log.i("SessionManager", "VAD detected speech! confidence=${vadResult.confidence}")
        }

        when (_sessionState.value) {
            SessionState.IDLE, SessionState.USER_SPEAKING -> {
                when (_recordingMode.value) {
                    RecordingMode.VAD -> {
                        // Automatic voice detection mode
                        if (vadResult.isSpeech) {
                            handleSpeechDetected()
                        } else {
                            handleSilenceDetected()
                        }
                    }
                    RecordingMode.PUSH_TO_TALK, RecordingMode.TOGGLE -> {
                        // Manual modes: only process if manually recording
                        // Speech detection is handled by startManualRecording/stopManualRecording
                        // Just track speech activity for UI feedback
                        if (_isManuallyRecording.value) {
                            lastSpeechDetectedTime = System.currentTimeMillis()
                        }
                    }
                }
            }
            SessionState.AI_SPEAKING -> {
                // Check for barge-in (works in all modes, but NOT when muted)
                if (!_isMuted.value && vadResult.isSpeech && canBargeIn()) {
                    handleBargeIn()
                }
            }
            else -> {
                // Other states: ignore audio
            }
        }
    }

    /**
     * Handle speech detected by VAD.
     */
    private suspend fun handleSpeechDetected() {
        lastSpeechDetectedTime = System.currentTimeMillis()
        val frameCount = consecutiveSpeechFrames.incrementAndGet()

        if (_sessionState.value == SessionState.IDLE) {
            // Require minimum consecutive speech frames before triggering STT
            // This prevents false triggers from brief noise spikes
            if (frameCount < minimumSpeechFrames) {
                Log.d(
                    "SessionManager",
                    "Speech below threshold: $frameCount/$minimumSpeechFrames frames",
                )
                return
            }

            // Atomically transition from IDLE to USER_SPEAKING to prevent race conditions
            // when multiple coroutines on Dispatchers.Default see IDLE simultaneously
            if (!_sessionState.compareAndSet(SessionState.IDLE, SessionState.USER_SPEAKING)) {
                return
            }

            // Start of user utterance
            userSpeechStartTime = System.currentTimeMillis()
            Log.i(
                "SessionManager",
                "Transitioning from IDLE to USER_SPEAKING (after $frameCount frames)",
            )

            // Start STT streaming
            Log.i("SessionManager", "Starting STT streaming...")
            startSTTStreaming()

            Log.i("SessionManager", "User started speaking - STT streaming started")
        }
    }

    /**
     * Handle silence detected by VAD.
     */
    private suspend fun handleSilenceDetected() {
        // Reset consecutive speech counter on silence
        consecutiveSpeechFrames.set(0)

        if (_sessionState.value == SessionState.USER_SPEAKING) {
            val silenceDuration = System.currentTimeMillis() - lastSpeechDetectedTime

            if (silenceDuration >= silenceThresholdMs) {
                // User finished speaking
                _sessionState.value = SessionState.PROCESSING_UTTERANCE

                // Stop STT and finalize
                finalizeSTT()

                Log.i("SessionManager", "User stopped speaking (${silenceDuration}ms silence)")
            }
        }
    }

    /**
     * Check if barge-in is allowed.
     */
    private fun canBargeIn(): Boolean {
        val aiSpeechDuration = System.currentTimeMillis() - aiSpeechStartTime
        return aiSpeechDuration >= bargeInConfirmationWindowMs
    }

    /**
     * Handle user barge-in during AI speech.
     */
    private suspend fun handleBargeIn() {
        Log.i("SessionManager", "Barge-in detected")

        // Cancel AI operations
        llmJob?.cancel()

        // Cancel ALL TTS jobs (not just one)
        synchronized(ttsJobs) {
            ttsJobs.forEach { it.cancel() }
            ttsJobs.clear()
        }

        audioEngine.stopPlayback()

        _sessionState.value = SessionState.INTERRUPTED

        // Immediately transition to listening
        userSpeechStartTime = System.currentTimeMillis()
        _sessionState.value = SessionState.USER_SPEAKING

        // Start new STT stream
        startSTTStreaming()
    }

    /**
     * Start STT streaming.
     */
    private fun startSTTStreaming() {
        sttJob?.cancel()

        // IMPORTANT: Stop AudioEngine capture to release the microphone for Android's SpeechRecognizer
        // Android's SpeechRecognizer needs exclusive microphone access
        Log.i("SessionManager", "Pausing AudioEngine capture for STT")
        audioEngine.stopCapture()

        sttJob =
            scope.launch {
                try {
                    sttService.startStreaming().collect { result ->
                        Log.d("SessionManager", "STT: ${result.text} (final=${result.isFinal})")

                        if (result.isFinal) {
                            // Final transcription received.
                            // IMPORTANT: hand off to a NEW coroutine. handleFinalTranscription
                            // cancels sttJob (this collector) and then suspends in
                            // sttService.stopStreaming() - if it ran inside this job, the
                            // self-cancellation would surface as CancellationException at the
                            // first suspension point and the LLM would never be called.
                            scope.launch {
                                handleFinalTranscription(result.text)
                            }
                        }
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.e("SessionManager", "STT error", e)
                    // Resume audio capture on error
                    resumeAudioCapture()
                    _sessionState.value = SessionState.ERROR
                }
            }
    }

    /**
     * Resume audio capture after STT completes.
     *
     * This method attempts to restart audio capture with retries to handle
     * timing issues when STT releases the microphone.
     *
     * @return true if audio capture was successfully resumed (or already active), false otherwise
     */
    @Suppress("ReturnCount") // Multiple early returns improve readability in retry logic
    private suspend fun resumeAudioCapture(): Boolean {
        Log.i("SessionManager", "Resuming AudioEngine capture after STT")

        // Check if already capturing - if so, audio is already flowing, which is success
        if (audioEngine.isCapturing.value) {
            Log.i("SessionManager", "Audio capture already active, no need to resume")
            return true
        }

        // Small delay to allow STT to fully release the microphone
        // Android's SpeechRecognizer needs time to clean up
        delay(audioResumeDelayMs)

        var retries = 0
        var success = false

        while (retries < maxAudioResumeRetries && !success) {
            // Re-check capturing state in case it changed during delay
            if (audioEngine.isCapturing.value) {
                Log.i("SessionManager", "Audio capture became active during delay")
                return true
            }

            success =
                audioEngine.startCapture { audioSamples ->
                    scope.launch {
                        processAudioFrame(audioSamples)
                    }
                }

            if (!success) {
                retries++
                Log.w(
                    "SessionManager",
                    "Failed to resume audio capture (attempt $retries/$maxAudioResumeRetries)",
                )
                if (retries < maxAudioResumeRetries) {
                    delay(audioResumeRetryDelayMs)
                }
            }
        }

        if (success) {
            Log.i("SessionManager", "Audio capture resumed successfully")
        } else {
            Log.e(
                "SessionManager",
                "Failed to resume audio capture after $maxAudioResumeRetries attempts",
            )
        }

        return success
    }

    /**
     * Finalize STT (called when silence threshold reached).
     */
    private suspend fun finalizeSTT() {
        // STT service will emit final result
        // Wait briefly for final transcription
        delay(200)
    }

    /**
     * Handle final transcription from STT.
     */
    private suspend fun handleFinalTranscription(text: String) {
        // Stop the STT collector job. This runs in its own coroutine (see startSTTStreaming),
        // so cancelling sttJob here is safe and won't cancel the processing below.
        sttJob?.cancel()
        try {
            sttService.stopStreaming()
        } catch (e: CancellationException) {
            // Concurrent cancellation of the collector; the recognizer is torn down by
            // callbackFlow's awaitClose anyway.
            Log.w("SessionManager", "STT stop cancelled (expected during hand-off)")
        } catch (e: Exception) {
            Log.e("SessionManager", "STT stop error", e)
        }

        val trimmedText = text.trim()
        val speechDuration = System.currentTimeMillis() - userSpeechStartTime

        // Filter out empty or very short transcriptions (likely noise)
        if (trimmedText.isBlank() || trimmedText.length < 2) {
            Log.d(
                "SessionManager",
                "Empty/short transcription after ${speechDuration}ms of speech, returning to IDLE",
            )

            // Brief delay to prevent rapid state cycling visible in UI
            delay(100)

            val audioResumed = resumeAudioCapture()
            _sessionState.value = if (audioResumed) SessionState.IDLE else SessionState.ERROR

            // Reset turn timing for next attempt
            currentTurnStartTime = System.currentTimeMillis()
            return
        }

        Log.d("SessionManager", "Final transcription received (${speechDuration}ms speech)")

        // Add to transcript
        val userEntry =
            TranscriptEntry(
                id = UUID.randomUUID().toString(),
                sessionId = _currentSession.value?.id ?: "",
                role = "user",
                text = trimmedText,
                timestamp = System.currentTimeMillis(),
            )
        addToTranscript(userEntry)

        // Add to conversation history
        conversationHistory.add(LLMMessage(role = "user", content = trimmedText))

        // Process with LLM
        processLLMResponse()
    }

    /**
     * Process LLM response.
     */
    @Suppress("LongMethod") // Contains streaming logic that's clearer as one method
    private fun processLLMResponse() {
        _sessionState.value = SessionState.AI_THINKING

        val llmStartTime = System.currentTimeMillis()
        var firstTokenTime = 0L
        val responseBuffer = StringBuilder() // Buffer for TTS chunks (cleared after each chunk)
        val fullResponse = StringBuilder() // Complete response for conversation history

        llmJob?.cancel()
        llmJob =
            scope.launch {
                try {
                    llmService.streamCompletion(
                        messages = conversationHistory,
                        temperature = 0.7f,
                        maxTokens = 500,
                    ).collect { token ->
                        if (firstTokenTime == 0L && token.content.isNotEmpty()) {
                            firstTokenTime = System.currentTimeMillis()
                            val ttft = firstTokenTime - llmStartTime
                            Log.i("SessionManager", "LLM TTFT: ${ttft}ms")

                            // Track TTFT metric
                            // Running average
                            _metrics.value =
                                _metrics.value.copy(
                                    llmTTFT = (_metrics.value.llmTTFT + ttft) / 2,
                                )

                            // Transition to AI_SPEAKING
                            _sessionState.value = SessionState.AI_SPEAKING
                            aiSpeechStartTime = System.currentTimeMillis()
                        }

                        if (!token.isDone) {
                            responseBuffer.append(token.content)
                            fullResponse.append(token.content) // Track complete response

                            // Stream to TTS in chunks (every ~20 tokens or sentence boundary)
                            if (shouldSendToTTS(responseBuffer.toString())) {
                                val chunk = responseBuffer.toString()
                                synthesizeAndPlay(chunk)
                                responseBuffer.clear()
                            }
                        } else {
                            // LLM complete
                            if (responseBuffer.isNotEmpty()) {
                                val finalChunk = responseBuffer.toString()
                                synthesizeAndPlay(finalChunk)
                                responseBuffer.clear()
                            }

                            // Add assistant response to conversation history
                            val assistantResponse = fullResponse.toString()
                            if (assistantResponse.isNotEmpty()) {
                                conversationHistory.add(
                                    LLMMessage(role = "assistant", content = assistantResponse),
                                )
                                Log.d("SessionManager", "Added assistant response to history: $assistantResponse")
                            }

                            // Finalize response
                            finalizeLLMResponse()
                        }
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.e("SessionManager", "LLM error: ${e.message}", e)

                    // Show ERROR state briefly so UI can display error feedback
                    _sessionState.value = SessionState.ERROR
                    delay(1500) // Give UI time to show error state

                    // Resume audio capture so user can continue speaking
                    val audioResumed = resumeAudioCapture()
                    if (audioResumed) {
                        _sessionState.value = SessionState.IDLE
                        currentTurnStartTime = System.currentTimeMillis()
                        Log.i("SessionManager", "Recovered from LLM error, returning to IDLE")
                    } else {
                        // Stay in ERROR if can't recover audio
                        Log.e("SessionManager", "Failed to recover from LLM error - audio capture failed")
                    }
                }
            }
    }

    /**
     * Determine if accumulated text should be sent to TTS.
     */
    private fun shouldSendToTTS(text: String): Boolean {
        // Send on sentence boundaries or every 100 characters
        return text.endsWith(".") || text.endsWith("?") || text.endsWith("!") || text.length >= 100
    }

    /**
     * Synthesize text and play audio.
     *
     * IMPORTANT: This method adds TTS jobs to a list instead of overwriting a single job.
     * This prevents race conditions when multiple LLM chunks trigger TTS synthesis.
     * All jobs are awaited in finalizeLLMResponse() before transitioning state.
     */
    private fun synthesizeAndPlay(text: String) {
        val job =
            scope.launch {
                try {
                    val audioChunks = mutableListOf<ByteArray>()
                    var firstChunkTime = 0L
                    val ttsStartTime = System.currentTimeMillis()

                    ttsService.synthesize(text).collect { chunk ->
                        if (firstChunkTime == 0L && chunk.audioData.isNotEmpty()) {
                            firstChunkTime = System.currentTimeMillis()
                            val ttfb = firstChunkTime - ttsStartTime
                            Log.i("SessionManager", "TTS TTFB: ${ttfb}ms")

                            // Track TTFB metric
                            _metrics.value =
                                _metrics.value.copy(
                                    ttsTTFB = (_metrics.value.ttsTTFB + ttfb) / 2,
                                )
                        }

                        audioChunks.add(chunk.audioData)

                        // Queue audio for playback
                        if (chunk.audioData.isNotEmpty()) {
                            // Convert to float samples and queue
                            val floatSamples = convertBytesToFloat(chunk.audioData)
                            audioEngine.queuePlayback(floatSamples)
                        }
                    }

                    Log.d("SessionManager", "TTS synthesis complete: $text")
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.e("SessionManager", "TTS error", e)
                }
            }

        // Add to the list of TTS jobs (instead of overwriting single ttsJob)
        synchronized(ttsJobs) {
            ttsJobs.add(job)
        }
    }

    /**
     * Finalize LLM response and add to transcript.
     */
    private suspend fun finalizeLLMResponse() {
        // Wait for ALL TTS jobs to complete (fixes race condition)
        val jobsToAwait: List<Job>
        synchronized(ttsJobs) {
            jobsToAwait = ttsJobs.toList()
        }

        Log.i("SessionManager", "Waiting for ${jobsToAwait.size} TTS jobs to complete")
        jobsToAwait.forEach { it.join() }

        // Clear the TTS jobs list
        synchronized(ttsJobs) {
            ttsJobs.clear()
        }
        Log.i("SessionManager", "All TTS jobs completed")

        // Get full response from conversation history
        val fullResponse = conversationHistory.lastOrNull { it.role == "assistant" }?.content ?: ""

        // Add to transcript
        val aiEntry =
            TranscriptEntry(
                id = UUID.randomUUID().toString(),
                sessionId = _currentSession.value?.id ?: "",
                role = "assistant",
                text = fullResponse,
                timestamp = System.currentTimeMillis(),
            )
        addToTranscript(aiEntry)

        // Update session metrics
        _currentSession.value?.let { session ->
            _currentSession.value =
                session.copy(
                    turnCount = session.turnCount + 1,
                )
        }

        // Calculate E2E latency
        val e2eLatency = System.currentTimeMillis() - currentTurnStartTime
        Log.i("SessionManager", "E2E turn latency: ${e2eLatency}ms")

        _metrics.value =
            _metrics.value.copy(
                e2eLatency = (_metrics.value.e2eLatency + e2eLatency) / 2,
            )

        // Resume audio capture and return to listening
        val audioResumed = resumeAudioCapture()
        if (audioResumed) {
            _sessionState.value = SessionState.IDLE
            currentTurnStartTime = System.currentTimeMillis()
        } else {
            _sessionState.value = SessionState.ERROR
            Log.e("SessionManager", "Failed to resume audio capture after LLM response")
        }
    }

    /**
     * Build system prompt for LLM.
     */
    private fun buildSystemPrompt(): String {
        val curriculumContext = curriculumEngine.getCurrentContext()

        return buildString {
            appendLine("You are an AI tutor helping a student learn through voice conversation.")
            appendLine("Keep responses concise and conversational (2-3 sentences).")
            appendLine("Encourage questions and check for understanding.")

            if (curriculumContext != null) {
                appendLine()
                appendLine("Current Topic: ${curriculumContext.topicTitle}")
                appendLine("Learning Objectives: ${curriculumContext.learningObjectives.joinToString(", ")}")
            }
        }
    }

    /**
     * Add entry to transcript.
     */
    private fun addToTranscript(entry: TranscriptEntry) {
        _transcript.value = _transcript.value + entry
    }

    /**
     * Cancel all active operations.
     */
    private fun cancelActiveOperations() {
        sttJob?.cancel()
        llmJob?.cancel()
        vadJob?.cancel()

        // Cancel ALL TTS jobs (not just one)
        synchronized(ttsJobs) {
            ttsJobs.forEach { it.cancel() }
            ttsJobs.clear()
        }

        // Launch coroutine to stop services asynchronously
        scope.launch {
            runCatching { sttService.stopStreaming() }
            runCatching { llmService.stop() }
            runCatching { ttsService.stop() }
        }
    }

    /**
     * Convert byte array to float samples.
     */
    private fun convertBytesToFloat(bytes: ByteArray): FloatArray {
        // Assuming 16-bit PCM
        val floats = FloatArray(bytes.size / 2)
        for (i in floats.indices) {
            val sample = (bytes[i * 2 + 1].toInt() shl 8) or (bytes[i * 2].toInt() and 0xFF)
            floats[i] = sample / 32768.0f
        }
        return floats
    }

    /**
     * Release resources.
     */
    fun release() {
        scope.cancel()
        audioEngine.release()
    }
}

/**
 * Session metrics tracked in real-time.
 */
data class SessionMetrics(
    /** Time to first token (ms) */
    val llmTTFT: Long = 0,
    /** Time to first byte (ms) */
    val ttsTTFB: Long = 0,
    /** End-to-end turn latency (ms) */
    val e2eLatency: Long = 0,
)
