package org.janelia.saalfeldlab.samlink.encode

/**
 * Per-channel `(x / 255 - mean) / std` over RGB, in that order.
 *
 * None of the encoder graphs normalize internally, so [ImageEncoding.RAW] applies this before sending
 * and [ImageEncoding.JPEG] leaves it to the server. The two only agree if they use the same constants.
 */
enum class Normalization(val mean: FloatArray, val std: FloatArray) {

    /** the torchvision ImageNet constants, used by SAM1 and SAM2 */
    IMAGENET(floatArrayOf(0.485f, 0.456f, 0.406f), floatArrayOf(0.229f, 0.224f, 0.225f)),

    /** maps 0-255 onto -1..1; used by SAM3 */
    SYMMETRIC(floatArrayOf(0.5f, 0.5f, 0.5f), floatArrayOf(0.5f, 0.5f, 0.5f))
}
