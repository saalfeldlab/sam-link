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
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.consumeAsFlow
import kotlinx.coroutines.launch
import java.lang.Float.float16ToFloat
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Multiplexed streaming Triton client with a per-lane stream + channel pool.
 *
 * Keeps [streamCount] lanes open. Each lane is self-contained:
 *   - its own [ManagedChannel]
 *   - its own single-thread [NioEventLoopGroup]
 *   - its own long-lived `ModelStreamInfer` bidi RPC
 */
class TritonClient(
    private val host: String,
    private val port: Int,
    private val timeoutMs: Long = 0,
    private val useTls: Boolean = port == 443,
    private val compression: String? = "gzip",
    private val maxStreamRetries: Int = 3,
    private val streamCount: Int = 4,
) : AutoCloseable {

    private val lock = Any()

    private var closed = false
    private val scope = CoroutineScope(
        SupervisorJob() + Dispatchers.IO + CoroutineName("TritonClient")
    )

    init {
        require(streamCount >= 1) { "streamCount must be >= 1, got $streamCount" }
        /* hint to init the ort env during the first request, instead of waiting until it's actually used */
        scope.launch { hintOrtEnvInit() }
    }

    private class PendingCall(
        val request: GrpcService.ModelInferRequest,
        val deferred: CompletableDeferred<GrpcService.ModelInferResponse>,
    ) {
        @Volatile
        var attempts: Int = 0
    }

    private val pending = ConcurrentHashMap<String, PendingCall>()
    private val pool: Array<Lane?> = arrayOfNulls(streamCount)
    private val nextSlot = AtomicInteger()

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
    }

    private fun pickSlot(): Int = (nextSlot.getAndIncrement() and Int.MAX_VALUE) % streamCount

    private fun buildChannel(): Pair<ManagedChannel, NioEventLoopGroup> {
        val eventLoop = NioEventLoopGroup(1)
        val channel = NettyChannelBuilder.forAddress(host, port)
            .eventLoopGroup(eventLoop)
            .channelType(NioSocketChannel::class.java)
            .maxInboundMessageSize(32 * MEGABYTES)
            .apply { if (!useTls) usePlaintext() }
            .build()
        return channel to eventLoop
    }

    @OptIn(DelicateCoroutinesApi::class)
    private fun ensureLane(slot: Int): Lane = synchronized(lock) {
        check(!closed) { "TritonClient is closed" }
        val existing = pool[slot]
        if (existing != null && existing.collector.isActive && !existing.requests.isClosedForSend) {
            return existing
        }
        /* Reuse the underlying channel + event loop if still alive; ManagedChannel
         * auto-reconnects if the TCP layer dropped. Only rebuild the RPC on top. */
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
                coroutineStub(channel)
                    .modelStreamInfer(lane.requests.consumeAsFlow())
                    .collect { response -> dispatch(response, lane) }
            } catch (t: Throwable) {
                when {
                    /* Normal cleanup: client is closing or our own scope was cancelled. */
                    closed || t is CancellationException -> {
                        LOG.debug(t) { "modelStreamInfer collector (slot $slot) exited (closed=$closed)" }
                        if (t is CancellationException) throw t
                    }
                    /* Expected under server overload: the per-RPC deadline fired. The channel is
                     * still healthy; any in-flight requests are retried on a fresh stream, and
                     * the next infer() call rebuilds the lane via ensureLane. */
                    Status.fromThrowable(t).code == Status.Code.DEADLINE_EXCEEDED -> {
                        LOG.debug { "modelStreamInfer collector (slot $slot) deadline exceeded; ${lane.ownedIds.size} in-flight" }
                        if (lane.ownedIds.isNotEmpty()) scope.launch { retryOwnedPending(lane, t) }
                    }
                    /* Real failure: resurrect the stream on this lane and re-send its in-flight requests. */
                    else -> {
                        LOG.warn(t) { "modelStreamInfer collector (slot $slot) failed; will retry ${lane.ownedIds.size} in-flight request(s)" }
                        scope.launch { retryOwnedPending(lane, t) }
                    }
                }
            }
        }
        pool[slot] = lane
        lane
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

    private suspend fun retryOwnedPending(failedLane: Lane, cause: Throwable) {
        val ownedSnapshot = failedLane.ownedIds.toList()
        if (ownedSnapshot.isEmpty()) return
        val callsToRetry = ownedSnapshot.mapNotNull { id -> pending[id] }
        /* Rebuild the lane's RPC (channel + event loop are reused). */
        val newLane = try {
            ensureLane(failedLane.slot)
        } catch (_: IllegalStateException) {
            /* Client was closed between collector death and retry; fail this lane's pending. */
            for (call in callsToRetry) {
                pending.remove(call.request.id)?.deferred?.completeExceptionally(cause)
            }
            return
        }
        for (call in callsToRetry) {
            call.attempts++
            if (call.attempts > maxStreamRetries) {
                pending.remove(call.request.id)
                call.deferred.completeExceptionally(
                    StatusRuntimeException(
                        Status.UNAVAILABLE
                            .withDescription("stream failed after $maxStreamRetries retries")
                            .withCause(cause)
                    )
                )
                continue
            }
            failedLane.ownedIds.remove(call.request.id)
            newLane.ownedIds.add(call.request.id)
            try {
                newLane.requests.send(call.request)
            } catch (e: Exception) {
                pending.remove(call.request.id)
                call.deferred.completeExceptionally(e)
            }
        }
    }

    private fun failAllPending(cause: Throwable) {
        for (id in pending.keys.toList()) {
            pending.remove(id)?.deferred?.completeExceptionally(cause)
        }
    }

    /**
     * Health check on a one-shot channel. Deliberately does NOT use the lane pool.
     */
    suspend fun isModelReady(modelName: String, modelVersion: String = ""): Boolean {
        val (channel, eventLoop) = buildChannel()
        try {
            val request = modelReadyRequest {
                name = modelName
                version = modelVersion
            }
            return coroutineStub(channel).modelReady(request).ready
        } catch (e: Exception) {
            LOG.warn { "modelReady failed for model [$modelName] at $host:$port: ${e.message}" }
            return false
        } finally {
            channel.shutdownNow()
            eventLoop.shutdownGracefully()
        }
    }

    private fun ModelInferRequestKt.Dsl.priority(priority: Long) {
        parameters["priority"] = inferParameter { int64Param = priority }
    }

    /**
     * Submit one inference. Routes to one of the [streamCount] lanes round-robin;
     * transparently retries on stream failure (up to [maxStreamRetries]). Caller
     * cancellation unregisters the pending entry; a later-arriving response for that
     * id is silently discarded.
     */
    suspend fun infer(modelName: String, inputs: List<InferInput>, params: Map<String, GrpcService.InferParameter> = emptyMap()): InferResponse {
        val requestId = UUID.randomUUID().toString()
        val request = modelInferRequest {
            this.modelName = modelName
            id = requestId
            priority(5)
            parameters.putAll(params)
            for (input in inputs) {
                this.inputs += inferInputTensor {
                    name = input.name
                    datatype = input.datatype
                    shape += input.shape
                }
                rawInputContents += input.data!!
            }
        }
        val call = PendingCall(request, CompletableDeferred())
        val lane = ensureLane(pickSlot())
        pending[requestId] = call
        lane.ownedIds.add(requestId)
        try {
            lane.requests.send(request)
            return InferResponse(call.deferred.await())
        } finally {
            pending.remove(requestId)
            lane.ownedIds.remove(requestId)
        }
    }

    private fun coroutineStub(
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
        }
        scope.cancel()
        failAllPending(IllegalStateException("TritonClient is closed"))
    }

    /**
     * Input tensor for inference request.
     *
     * @property name tensor name
     * @property datatype data type string (e.g., "FP32", "FP16")
     * @property shape tensor shape as list of dimensions
     * @property data tensor data as ByteString
     */
    data class InferInput(
        val name: String,
        val datatype: String,
        val shape: List<Long>,
        val data: ByteString? = null,
    )

    /**
     * Wrapper for inference response with convenient output parsing.
     */
    class InferResponse(private val response: GrpcService.ModelInferResponse) {

        /**
         * Get output tensor by name as a float array.
         * Automatically handles FP16 to FP32 conversion if needed.
         */
        fun getFloatArray(name: String): FloatArray? {
            val index = response.outputsList.indexOfFirst { it.name == name }
            if (index < 0) return null

            val output = response.outputsList[index]
            val rawBytes = response.getRawOutputContents(index)
            val buffer = rawBytes.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN)

            return if (output.datatype == "FP16") {
                fp16BytesToFloatArray(buffer)
            } else {
                FloatArray(buffer.remaining() / 4).also { buffer.asFloatBuffer().get(it) }
            }
        }
    }


    companion object {

        private const val MEGABYTES = 1024 * 1024

        private val LOG = KotlinLogging.logger { }

        /** Convert FP16 bytes to float array */
        fun fp16BytesToFloatArray(buffer: ByteBuffer): FloatArray {
            val count = buffer.remaining() / 2
            return FloatArray(count) { float16ToFloat(buffer.short) }
        }

        private suspend fun hintOrtEnvInit() {
            if (ORT_ENV_LAZY.isInitialized())
                coroutineScope {
                    ORT_ENV //It's a lazy delegate, this should load it
                }
        }
    }
}
