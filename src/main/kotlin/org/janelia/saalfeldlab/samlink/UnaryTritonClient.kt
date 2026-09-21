package org.janelia.saalfeldlab.samlink

import inference.GRPCInferenceServiceGrpcKt
import inference.GrpcService
import inference.ModelInferRequestKt
import inference.ModelInferRequestKt.inferInputTensor
import inference.inferParameter
import inference.modelInferRequest
import inference.modelReadyRequest
import io.github.oshai.kotlinlogging.KotlinLogging
import io.grpc.ManagedChannel
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder
import io.grpc.netty.shaded.io.netty.channel.nio.NioEventLoopGroup
import io.grpc.netty.shaded.io.netty.channel.socket.nio.NioSocketChannel
import kotlinx.coroutines.CoroutineName
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import java.util.UUID
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Triton client that issues one unary `ModelInfer` RPC per request.
 *
 * At most [callsPerConnection] responses are in flight on one socket, and excess callers wait here
 * rather than on the wire. Large responses sharing a socket interleave and land together at the end
 * of a batch; one at a time keeps delivery first-come-first-served.
 *
 * A failed call fails its own caller and nothing else, and [timeoutMs] is a real per-call deadline.
 * Cancelling the caller aborts the RPC, and the server discards whatever is still queued; a request
 * already executing runs to completion regardless.
 */
class UnaryTritonClient(
    val host: String,
    val port: Int,
    override var timeoutMs: Long = 0,
    val useTls: Boolean = port == 443,
    val compression: String? = "gzip",
    val connectionCount: Int = 4,
    val callsPerConnection: Int = 1,
) : TritonClient {

    private val lock = Any()

    @Volatile
    private var closed = false
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO + CoroutineName("UnaryTritonClient"))

    private val connections: List<Connection>
    private val nextSlot = AtomicInteger()

    init {
        require(connectionCount >= 1) { "connectionCount must be >= 1, got $connectionCount" }
        require(callsPerConnection >= 1) { "callsPerConnection must be >= 1, got $callsPerConnection" }
        connections = List(connectionCount) { Connection(buildChannel(), callsPerConnection) }
        scope.launch { hintOrtEnvInit() }
    }

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

    private fun pickSlot() = (nextSlot.getAndIncrement() and Int.MAX_VALUE) % connections.size

    /** prefer a connection with nothing in flight; otherwise wait on the next one round-robin */
    private suspend fun <T> onConnection(block: suspend (ManagedChannel) -> T): T {
        check(!closed) { "UnaryTritonClient is closed" }
        for (connection in connections) {
            if (connection.permits.tryAcquire()) {
                try {
                    return block(connection.channel)
                } finally {
                    connection.permits.release()
                }
            }
        }
        val connection = connections[pickSlot()]
        return connection.permits.withPermit { block(connection.channel) }
    }

    override suspend fun isModelReady(modelName: String, modelVersion: String): Boolean {
        return try {
            val request = modelReadyRequest {
                name = modelName
                version = modelVersion
            }
            stub(connections[0].channel).modelReady(request).ready
        } catch (e: Exception) {
            LOG.warn { "modelReady failed for model [$modelName] at $host:$port: ${e.message}" }
            false
        }
    }

    override suspend fun infer(
        model: String,
        inferInputs: List<InferenceInput>,
        params: Map<String, GrpcService.InferParameter>,
    ): GrpcService.ModelInferResponse {
        val request = modelInferRequest {
            modelName = model
            id = UUID.randomUUID().toString()
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
        return onConnection { channel -> stub(channel).modelInfer(request) }
    }

    private fun ModelInferRequestKt.Dsl.priority(priority: Long) {
        parameters["priority"] = inferParameter { int64Param = priority }
    }

    private fun stub(channel: ManagedChannel): GRPCInferenceServiceGrpcKt.GRPCInferenceServiceCoroutineStub {
        var stub = GRPCInferenceServiceGrpcKt.GRPCInferenceServiceCoroutineStub(channel)
        if (timeoutMs > 0) stub = stub.withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS)
        if (compression != null) stub = stub.withCompression(compression)
        return stub
    }

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            for (connection in connections) {
                connection.channel.shutdown()
                connection.eventLoop.shutdownGracefully()
            }
        }
        scope.cancel()
    }

    private class Connection(channelAndLoop: Pair<ManagedChannel, NioEventLoopGroup>, callsPerConnection: Int) {
        val channel = channelAndLoop.first
        val eventLoop = channelAndLoop.second
        val permits = Semaphore(callsPerConnection)
    }

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
