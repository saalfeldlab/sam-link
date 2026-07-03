package org.janelia.saalfeldlab.samlink.encode

/**
 * Base interface for encode options.
 * Each encoder type defines its own options class.
 */
sealed interface EncodeOptions

abstract class TritonEncodeOptions(var priority: Long) : EncodeOptions

/**
 * How the source image is delivered to the encoder over the wire.
 *
 * [RAW] sends the preprocessed FP32 CHW tensor directly (~12.6 MB uncompressed).
 * [JPEG] sends the padded input as a JPEG.
 */
enum class ImageEncoding { RAW, JPEG }

/**
 * Options for SAM2 Triton encoder.
 */
class Sam1TritonOptions(priority: Long = 5) : TritonEncodeOptions(priority)

/**
 * Options for SAM2 Triton encoder.
 *
 * @property imageEncoding how the image is sent to the server; see [ImageEncoding].
 * @property quality supported dependent on the ImageEncoding. (e.g. in range [0,1] for jpeg)
 */
class Sam2TritonOptions(
    val imageEncoding: ImageEncoding,
    val quality: Float = 0.75f,
    priority: Long = 5,
) : TritonEncodeOptions(priority)

/**
 * Options for SAM3 Tracker Triton encoder.
 */
class Sam3TrackerTritonOptions(priority: Long = 5) : TritonEncodeOptions(priority)