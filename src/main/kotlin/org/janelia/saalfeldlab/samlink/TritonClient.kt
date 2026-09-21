package org.janelia.saalfeldlab.samlink

import com.google.protobuf.ByteString
import inference.GrpcService

/** A Triton gRPC client. */
interface TritonClient : AutoCloseable {

    /** per-call timeout in milliseconds; 0 waits indefinitely */
    var timeoutMs: Long

    suspend fun isModelReady(modelName: String, modelVersion: String = ""): Boolean

    /** Submit one inference; the server returns every output the model declares */
    suspend fun infer(
        model: String,
        inferInputs: List<InferenceInput>,
        params: Map<String, GrpcService.InferParameter> = emptyMap(),
    ): GrpcService.ModelInferResponse
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
