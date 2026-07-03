package org.janelia.saalfeldlab.samlink

import org.janelia.saalfeldlab.samlink.encode.ImageEncoding
import org.janelia.saalfeldlab.samlink.encode.triton.Sam1TritonEncoder
import org.janelia.saalfeldlab.samlink.encode.triton.Sam2TritonEncoder
import org.janelia.saalfeldlab.samlink.encode.triton.Sam3TrackerTritonEncoder
import java.net.URI

/**
 * Reads Triton configuration from environment variables for tests.
 */
object TritonEnv {

    fun errorTritonInfo() =
        IllegalStateException("Triton environment variables are required (TRITON_URL or TRITON_HOST and TRITON_PORT)")

    private fun hostFromUrl(url: String?) = runCatching { url?.let { URI.create(url).host } ?: url }.getOrNull()
    private fun portFromUrl(url: String?) =
        runCatching { if (url?.startsWith("https") == true) 443 else 8001 }.getOrNull()

    private val url = System.getenv("TRITON_URL")
    private val host = System.getenv("TRITON_HOST")
    private val port = System.getenv("TRITON_PORT")
    private val sam1Model = System.getenv("SAM1_MODEL")
    private val sam2Model = System.getenv("SAM2_MODEL")
    private val sam2JpegModel = System.getenv("SAM2_JPEG_MODEL")
    private val sam3TrackerModel = System.getenv("SAM3_TRACKER_MODEL")

    /** per-call gRPC timeout; raise via RESPONSE_TIMEOUT_MS for slow/throttled links */
    private val responseTimeoutMs = System.getenv("RESPONSE_TIMEOUT_MS")?.toLongOrNull() ?: 30_000L

    fun url(): String? = url
    fun host() = host ?: hostFromUrl(url()) ?: throw errorTritonInfo()
    fun port() = port?.toIntOrNull() ?: portFromUrl(url()) ?: throw errorTritonInfo()
    fun sam1Model() = sam1Model ?: "sam1_encoder"
    fun sam2Model(encoding: ImageEncoding) = when (encoding) {
        ImageEncoding.RAW -> sam2Model ?: "sam2.1_large_encoder"
        ImageEncoding.JPEG -> sam2JpegModel ?: "sam2_encoder_jpeg"
    }
    fun sam2JpegModel() = sam2JpegModel ?: "sam2_encoder_jpeg"
    fun sam3TrackerModel() = sam3TrackerModel ?: "sam3_tracker_encoder_fp16"

    fun newClient(): TritonClient {
        val name = runCatching {
            val hostAsString = host()
            URI.create(hostAsString).host ?: hostAsString
        }.getOrElse { host() }
        return TritonClient(host = name, port = port())
    }

    fun newSam1Encoder() = Sam1TritonEncoder(host(), port(), sam1Model(), responseTimeoutMs)

    fun newSam2Encoder(encoding: ImageEncoding) = Sam2TritonEncoder(host(), port(), sam2Model(encoding), responseTimeoutMs)

    fun newSam3TrackerEncoder() = Sam3TrackerTritonEncoder(host(), port(), sam3TrackerModel(), responseTimeoutMs)
}