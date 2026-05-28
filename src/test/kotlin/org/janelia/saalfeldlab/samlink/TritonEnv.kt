package org.janelia.saalfeldlab.samlink

import org.janelia.saalfeldlab.samlink.encode.Sam1TritonEncoder
import org.janelia.saalfeldlab.samlink.encode.Sam2TritonEncoder
import org.janelia.saalfeldlab.samlink.encode.Sam3TrackerTritonEncoder
import java.net.URI

/**
 * Reads Triton configuration from environment variables for tests.
 */
object TritonEnv {

    private fun hostFromUrl(url: String?) = runCatching { url?.let { URI.create(url).host } ?: host }.getOrNull()
    private fun portFromUrl(url: String?) = runCatching { if (url?.startsWith("https") == true) 443 else 8001 }.getOrNull()

    val url: String? = System.getenv("TRITON_URL")
    val host: String = requireNotNull(System.getenv("TRITON_HOST") ?: hostFromUrl(url)) { "Environment variable TRITON_HOST not specified "}
    val port: Int = requireNotNull(System.getenv("TRITON_PORT")?.toIntOrNull() ?: portFromUrl(url)) { "Environment variable TRITON_PORT not specified "}

    val sam1Model: String = System.getenv("SAM1_MODEL") ?: "sam1_encoder"
    val sam2Model: String = System.getenv("SAM2_MODEL") ?: "sam2.1_large_encoder"
    val sam3TrackerModel: String = System.getenv("SAM3_TRACKER_MODEL") ?: "sam3_tracker_encoder_fp16"

    fun newClient(): TritonClient {
        val name = runCatching {
            URI.create(host).host ?: host
        }.getOrElse { host }
        return TritonClient(host = name, port = port)
    }

    fun newSam1Encoder(): Sam1TritonEncoder = Sam1TritonEncoder(host, sam1Model, grpcPort = port)
    fun newSam2Encoder(): Sam2TritonEncoder = Sam2TritonEncoder(host, sam2Model, grpcPort = port)
    fun newSam3TrackerEncoder(): Sam3TrackerTritonEncoder =
        Sam3TrackerTritonEncoder(host, sam3TrackerModel, grpcPort = port)
}