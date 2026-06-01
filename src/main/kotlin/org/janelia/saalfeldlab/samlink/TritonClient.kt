package org.janelia.saalfeldlab.samlink

import com.google.protobuf.ByteString
import inference.GRPCInferenceServiceGrpcKt
import inference.GrpcService
import inference.ModelInferRequestKt
import inference.ModelInferRequestKt.inferInputTensor
import inference.inferParameter
import inference.modelInferRequest
import inference.modelReadyRequest
import io.github.oshai.kotlinlogging.KotlinLogging
import io.grpc.ManagedChannel
import io.grpc.Status
import io.grpc.StatusRuntimeException
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder
import io.grpc.netty.shaded.io.netty.channel.nio.NioEventLoopGroup
import io.grpc.netty.shaded.io.netty.channel.socket.nio.NioSocketChannel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineName
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.consumeAsFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout
import java.net.SocketException
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.time.Duration.Companion.milliseconds

/**
 * Multiplexed streaming Triton client with a per-lane stream + channel pool.
 *
 * Keeps [streamCount] lanes open. Each lane is self-contained:
 *   - [ManagedChannel]
 *   - single-thread [NioEventLoopGroup]
 *   - long-lived `ModelStreamInfer` bidi RPC
 */
class TritonClient(
    val host: String,
    val port: Int,
    var timeoutMs: Long = 0,
    val useTls: Boolean = port == 443,
    val compression: String? = "gzip",
    val maxStreamRetries: Int = 3,
    val streamCount: Int = 4,
) : AutoCloseable {

    private val lock = Any()

    @Volatile
    private var closed = false
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO + CoroutineName("TritonClient"))

    private val controlChannel: Lazy<Pair<ManagedChannel, NioEventLoopGroup>> = lazy { buildChannel() }

    init {
        require(streamCount >= 1) { "streamCount must be >= 1, got $streamCount" }
        /* hint to init the ort env during the first request, instead of waiting until it's actually used */
        scope.launch { hintOrtEnvInit() }
    }

    private val pending = ConcurrentHashMap<String, PendingCall>()
    private val pool: Array<Lane?> = arrayOfNulls(streamCount)
    private val nextSlot = AtomicInteger()

    private fun pickSlot(): Int = (nextSlot.getAndIncrement() and Int.MAX_VALUE) % streamCount

    private fun buildChannel(): Pair<ManagedChannel, NioEventLoopGroup> {
        val eventLoop = NioEventLoopGroup(1)
        val channel = NettyChannelBuilder.forAddress(host, port)
            .eventLoopGroup(eventLoop)
            .channelType(NioSocketChannel::class.java)
            .maxInboundMessageSize(32 * BYTES_MB)
            .apply { if (!useTls) usePlaintext() }
            .build()
        return channel to eventLoop
    }

    @OptIn(DelicateCoroutinesApi::class)
    private fun ensureLaneLocked(slot: Int): Lane {
        /* lock must already be held */
        check(!closed) { "TritonClient is closed" }
        val existing = pool[slot]
        if (existing != null && !existing.dead && !existing.requests.isClosedForSend) {
            return existing
        }
        /* Reuse the underlying channel and event loop if still alive.  */
        val channel: ManagedChannel
        val eventLoop: NioEventLoopGroup
        if (existing != null && !existing.channel.isShutdown && !existing.channel.isTerminated) {
            channel = existing.channel
            eventLoop = existing.eventLoop
        } else {
            existing?.channel?.shutdown()
            existing?.eventLoop?.shutdownGracefully()
            val (ch, elg) = buildChannel()
            channel = ch
            eventLoop = elg
        }
        val lane = Lane(slot, channel, eventLoop, Channel(Channel.UNLIMITED))
        lane.collector = scope.launch {
            try {
                streamingStub(channel)
                    .modelStreamInfer(lane.requests.consumeAsFlow())
                    .collect { response -> dispatch(response, lane) }
            } catch (cause: Throwable) {
                if (cause is CancellationException) throw cause

                if (closed) {
                    LOG.debug(cause) { "modelStreamInfer collector (slot $slot) exited (closed)" }
                    return@launch
                }

                val retry = synchronized(lock) {
                    lane.dead = true
                    lane.ownedIds.toList()
                }

                if (retry.isEmpty() &&
                    Status.fromThrowable(cause).code == Status.Code.UNAVAILABLE &&
                    generateSequence(cause) { it.cause }.any { it is SocketException }
                ) {
                    LOG.debug(cause) { "modelStreamInfer collector (slot $slot) reset while idle; lane will rebuild on next infer()" }
                    return@launch
                }

                LOG.warn(cause) { "Infer failed. will retry ${retry.size} in-flight request(s)" }
                if (retry.isNotEmpty()) scope.launch { retryOwnedPending(lane, retry, cause) }
            }
        }
        pool[slot] = lane
        return lane
    }

    private fun submitLocked(call: PendingCall, slot: Int): Lane = synchronized(lock) {
        check(!closed) { "TritonClient is closed" }
        ensureLaneLocked(slot).apply {
            ownedIds.add(call.request.id)
            check(requests.trySend(call.request).isSuccess) {
                "fresh lane's request channel unexpectedly closed"
            }
        }
    }

    private fun dispatch(response: GrpcService.ModelStreamInferResponse, lane: Lane) {
        val id = response.inferResponse.id
        lane.ownedIds.remove(id)
        val call = pending.remove(id)
        if (call == null) {
            LOG.debug { "received response for unknown or already-cancelled request id='$id'" }
            return
        }
        if (response.errorMessage.isNotEmpty()) {
            call.deferred.completeExceptionally(
                StatusRuntimeException(Status.INTERNAL.withDescription(response.errorMessage))
            )
        } else {
            call.deferred.complete(response.inferResponse)
        }
    }

    private fun retryOwnedPending(failedLane: Lane, ownedSnapshot: List<String>, cause: Throwable) {
        val callsToRetry = ownedSnapshot.mapNotNull { id -> pending[id] }
        for (call in callsToRetry) {
            call.attempts++
            if (call.attempts > maxStreamRetries) {
                pending.remove(call.request.id)?.deferred?.completeExceptionally(
                    StatusRuntimeException(
                        Status.UNAVAILABLE
                            .withDescription("stream failed after $maxStreamRetries retries")
                            .withCause(cause)
                    )
                )
                continue
            }
            failedLane.ownedIds.remove(call.request.id)
            try {
                submitLocked(call, failedLane.slot)
            } catch (e: Exception) {
                pending.remove(call.request.id)?.deferred?.completeExceptionally(e)
            }
        }
    }

    private fun failAllPending(cause: Throwable) {
        for (id in pending.keys.toList()) {
            pending.remove(id)?.deferred?.completeExceptionally(cause)
        }
    }

    /**
     * Health check over the shared control channel; safe to poll without paying
     * per-call channel construction.
     */
    suspend fun isModelReady(modelName: String, modelVersion: String = ""): Boolean {
        val (channel, _) = controlChannel.value
        return try {
            val request = modelReadyRequest {
                name = modelName
                version = modelVersion
            }
            unaryStub(channel).modelReady(request).ready
        } catch (e: Exception) {
            LOG.warn { "modelReady failed for model [$modelName] at $host:$port: ${e.message}" }
            false
        }
    }

    private fun ModelInferRequestKt.Dsl.priority(priority: Long) {
        parameters["priority"] = inferParameter { int64Param = priority }
    }

    /**
     * Submit one inference. Routes to one of the [streamCount] lanes round-robin;
     * transparently retries on stream failure (up to [maxStreamRetries]). When
     * [timeoutMs] > 0 it bounds the per-call wait via [withTimeout] (the server-side
     * work isn't cancelled; a late response is simply discarded). Caller cancellation
     * unregisters the pending entry; a later-arriving response for that id is
     * silently discarded.
     */
    suspend fun infer(
        model: String,
        inferInputs: List<InferenceInput>,
        params: Map<String, GrpcService.InferParameter> = emptyMap()
    ): GrpcService.ModelInferResponse {
        val requestId = UUID.randomUUID().toString()
        val request = modelInferRequest {
            modelName = model
            id = requestId
            priority(5)
            parameters.putAll(params)
            for (input in inferInputs) {
                inputs += inferInputTensor {
                    name = input.name
                    datatype = input.datatype
                    shape += input.shape.toList()
                }
                rawInputContents += input.data
            }
        }
        val call = PendingCall(request, CompletableDeferred())
        pending[requestId] = call
        val lane = try {
            submitLocked(call, pickSlot())
        } catch (e: Throwable) {
            pending.remove(requestId)
            throw e
        }
        try {
            return if (timeoutMs > 0) {
                withTimeout(timeoutMs.milliseconds) { call.deferred.await() }
            } else {
                call.deferred.await()
            }
        } catch (e: TimeoutCancellationException) {
            throw StatusRuntimeException(
                Status.DEADLINE_EXCEEDED
                    .withDescription("infer($model) timed out after ${timeoutMs.milliseconds}")
                    .withCause(e)
            )
        } finally {
            pending.remove(requestId)
            lane.ownedIds.remove(requestId)
        }
    }

    /** Streaming stub: never carries a stream-wide deadline (per-call timing is in [infer]). */
    private fun streamingStub(
        channel: ManagedChannel,
    ): GRPCInferenceServiceGrpcKt.GRPCInferenceServiceCoroutineStub {
        var stub = GRPCInferenceServiceGrpcKt.GRPCInferenceServiceCoroutineStub(channel)
        if (compression != null) stub = stub.withCompression(compression)
        return stub
    }

    /** Unary stub: per-RPC deadline is meaningful here (single request/response). */
    private fun unaryStub(
        channel: ManagedChannel,
    ): GRPCInferenceServiceGrpcKt.GRPCInferenceServiceCoroutineStub {
        var stub = GRPCInferenceServiceGrpcKt.GRPCInferenceServiceCoroutineStub(channel)
        if (timeoutMs > 0) stub = stub.withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS)
        if (compression != null) stub = stub.withCompression(compression)
        return stub
    }

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            for (i in pool.indices) {
                val lane = pool[i] ?: continue
                lane.requests.close()
                lane.channel.shutdown()
                lane.eventLoop.shutdownGracefully()
                pool[i] = null
            }
            if (controlChannel.isInitialized()) {
                val (ch, elg) = controlChannel.value
                ch.shutdown()
                elg.shutdownGracefully()
            }
        }
        scope.cancel()
        failAllPending(IllegalStateException("TritonClient is closed"))
    }

    /**
     * Input tensor for inference request.
     *
     * @property name tensor name
     * @property datatype data type string (e.g., "FP32")
     * @property shape tensor shape as list of dimensions
     * @property data tensor data as ByteString
     */
    class InferenceInput(
        val name: String,
        val shape: LongArray,
        val datatype: String,
        val data: ByteString,
    )

    companion object {
        private const val BYTES_MB = 1024 * 1024
        private val LOG = KotlinLogging.logger { }
    }
}

private suspend fun hintOrtEnvInit() {
    if (ORT_ENV_LAZY.isInitialized())
        coroutineScope {
            ORT_ENV //It's a lazy delegate, this should load it
        }
}

/**
 * A self-contained inference lane: dedicated channel, dedicated Netty event loop,
 * and a single bidi-streaming collector coroutine.
 *
 * The [channel] + [eventLoop] live across stream-level failures; only the
 * [requests] channel and [collector] coroutine are rebuilt when the RPC fails.
 */
private class Lane(
    val slot: Int,
    val channel: ManagedChannel,
    val eventLoop: NioEventLoopGroup,
    val requests: Channel<GrpcService.ModelInferRequest>,
) {
    /** Request ids currently in flight on this lane — used to scope retries. */
    val ownedIds: MutableSet<String> = ConcurrentHashMap.newKeySet()
    lateinit var collector: Job

    /**
     * Set by the collector's catch path (under TritonClient.lock) once the stream
     * has failed. Submitters check it inside the same lock before adding to the
     * lane, which closes the race between "lane just died" and "new submission".
     */
    @Volatile
    var dead: Boolean = false
}

private class PendingCall(
    val request: GrpcService.ModelInferRequest,
    val deferred: CompletableDeferred<GrpcService.ModelInferResponse>,
) {
    @Volatile
    var attempts: Int = 0
}
