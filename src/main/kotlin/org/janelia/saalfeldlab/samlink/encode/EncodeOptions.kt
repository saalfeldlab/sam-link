package org.janelia.saalfeldlab.samlink.encode

/**
 * Base interface for encode options.
 * Each encoder type defines its own options class.
 */
sealed interface EncodeOptions

/**
 * How the source image is delivered to the encoder over the wire.
 *
 * [RAW] sends the FP32 tensor exactly as the encoder expects it
 * [JPEG] sends the image as a JPEG, scaled and padded to the target input size.
 */
enum class ImageEncoding { RAW, JPEG }

/**
 * @property imageEncoding how the image is sent to the server; see [ImageEncoding]
 * @property quality JPEG quality in [0, 1]; ignored by [ImageEncoding.RAW]
 * @property requestFp16 ask the endpoint for half precision.
 */
abstract class TritonEncodeOptions(
    var priority: Long,
    val imageEncoding: ImageEncoding,
    val quality: Float,
    val requestFp16: Boolean,
) : EncodeOptions

class Sam1TritonOptions(
    priority: Long = 5,
    imageEncoding: ImageEncoding = ImageEncoding.RAW,
    quality: Float = 0.75f,
    requestFp16: Boolean = false,
) : TritonEncodeOptions(priority, imageEncoding, quality, requestFp16)

class Sam2TritonOptions(
    priority: Long = 5,
    imageEncoding: ImageEncoding = ImageEncoding.RAW,
    quality: Float = 0.75f,
    requestFp16: Boolean = false,
) : TritonEncodeOptions(priority, imageEncoding, quality, requestFp16)

class Sam3TrackerTritonOptions(
    priority: Long = 5,
    imageEncoding: ImageEncoding = ImageEncoding.RAW,
    quality: Float = 0.75f,
    requestFp16: Boolean = false,
) : TritonEncodeOptions(priority, imageEncoding, quality, requestFp16)
