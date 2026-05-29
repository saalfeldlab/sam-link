package org.janelia.saalfeldlab.samlink.encode.triton

import org.janelia.saalfeldlab.samlink.TritonClient
import org.janelia.saalfeldlab.samlink.encode.EncoderResult
import org.janelia.saalfeldlab.samlink.encode.SamEncoder
import org.janelia.saalfeldlab.samlink.encode.TritonEncodeOptions

abstract class SamTritonEncoder<R : EncoderResult, O : TritonEncodeOptions> : SamEncoder<R, O> {

    protected val client: TritonClient
    protected val model: String

    var responseTimeout: Long
        get() = client.timeoutMs
        set(value) {
            client.timeoutMs = value
        }

    constructor(client: TritonClient, model: String) {
        this.client = client
        this.model = model
    }

    constructor(
        host: String,
        port: Int = 8001,
        model: String,
        responseTimeout: Long = 30_000
    ) : this(TritonClient(host, port), model) {
        this.responseTimeout = responseTimeout
    }

    override suspend fun isReady() = client.isModelReady(model)

    override fun close() {
        client.close()
    }
}