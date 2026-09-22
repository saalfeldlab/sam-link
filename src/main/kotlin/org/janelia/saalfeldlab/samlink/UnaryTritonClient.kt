package org.janelia.saalfeldlab.samlink

import inference.GRPCInferenceServiceGrpcKt
import inference.GrpcService
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
 * At most [callsPerConnection] responses are in flight on one socket. Default of `1` since our
 * packet sizes are expected to be quite large; this stops messages from interleaving and blocking
 * each other.
 *
 * Number of [reservedConnections] that wait for priority 1 requests.
 *
 * [timeoutMs] is a per-call timeout.
 *
 * Requests are cancellable
 */
class UnaryTritonClient(
    val host: String,
    val port: Int,
    override var timeoutMs: Long = 0,
    val useTls: Boolean = port == 443,
    val compression: String? = "gzip",
    val connectionCount: Int = 5,
    val callsPerConnection: Int = 1,
    val reservedConnections: Int = 1,
) : TritonClient {

    private val lock = Any()

    @Volatile
    private var closed = false
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO + CoroutineName("UnaryTritonClient"))

    private val connections: List<Connection>

    /** the reserved connections for priority 1*/
    private val reserved: List<Connection>
    private val shared: List<Connection>

    private val nextSlot = AtomicInteger()

    init {
        require(connectionCount >= 1) { "connectionCount must be >= 1, got $connectionCount" }
        require(callsPerConnection >= 1) { "callsPerConnection must be >= 1, got $callsPerConnection" }
        require(reservedConnections in 0 until connectionCount) {
            "reservedConnections must leave at least one shared connection, got $reservedConnections of $connectionCount"
        }
        connections = List(connectionCount) { Connection(buildChannel(), callsPerConnection) }
        reserved = connections.take(reservedConnections)
        shared = connections.drop(reservedConnections)
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

    private fun pickSlot(size: Int) = (nextSlot.getAndIncrement() and Int.MAX_VALUE) % size

    /**
     * Prefer a connection with nothing in flight; otherwise wait on the next one round-robin.
     *
     * Priority 1 uses reserved connections first and falls back to the shared pool
     */
    private suspend fun <T> onConnection(priority: Long, block: suspend (ManagedChannel) -> T): T {
        check(!closed) { "UnaryTritonClient is closed" }
        val connections = if (priority <= HIGH_PRIORITY) reserved + shared else shared
        for (connection in connections) {
            if (connection.permits.tryAcquire()) {
                try {
                    return block(connection.channel)
                } finally {
                    connection.permits.release()
                }
            }
        }
        val connection = connections[pickSlot(connections.size)]
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
        priority: Long,
        params: Map<String, GrpcService.InferParameter>,
    ): GrpcService.ModelInferResponse {
        val request = modelInferRequest {
            modelName = model
            id = UUID.randomUUID().toString()
            parameters["priority"] = inferParameter { int64Param = priority }
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
        return onConnection(priority) { channel -> stub(channel).modelInfer(request) }
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
        private const val HIGH_PRIORITY = 1L
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
