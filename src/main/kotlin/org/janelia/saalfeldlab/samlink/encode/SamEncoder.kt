package org.janelia.saalfeldlab.samlink.encode

import ai.onnxruntime.OnnxTensor
import org.janelia.saalfeldlab.samlink.ORT_ENV
import java.awt.image.BufferedImage
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Interface for SAM image encoders.
 *
 * @param O the options type for this encoder
 */
sealed interface SamEncoder<R: EncoderResult, O : EncodeOptions> : AutoCloseable {

    /**
     * Returns a new options instance with default values.
     */
    fun options(): O

    /**
     * Encode an image and return the embeddings.
     *
     * This is a suspend function that supports coroutine cancellation. When cancelled,
     * Triton encoders will signal the server to stop processing the request.
     *
     * For blocking usage, wrap in `runBlocking { encode(image) }`.
     *
     * @param image the image to encode
     * @param options encoding options
     * @return the encoder result containing embeddings/features
     */
    suspend fun encode(image: BufferedImage, options: O = options()): R

    /**
     * The input size expected by this encoder.
     */
    val inputSize: Int

    /**
     * Check if the encoder service is ready for inference.
     */
    suspend fun isReady(): Boolean
}

/**
 * Base class for encoder results.
 * Results hold OnnxTensors and must be closed when no longer needed.
 *
 * [imageWidth] and [imageHeight] are the dimensions of the image content
 * within the [inputSize]×[inputSize] padded square, after resizing.
 * The content occupies the top-left [imageWidth]×[imageHeight] region;
 * the remainder is black padding.
 */
sealed interface EncoderResult : AutoCloseable {

    val inputSize: Int
    val imageWidth: Int
    val imageHeight: Int
}

/**
 * Base interface for encoder options.
 * Each encoder type defines its own options class.
 */
sealed interface EncodeOptions

abstract class TritonEncodeOptions(var priority: Long) : EncodeOptions

/**
 * Options for SAM2 Triton encoder.
 */
class Sam1TritonOptions(priority: Long = 5) : TritonEncodeOptions(priority)

/**
 * Options for SAM2 Triton encoder.
 */
class Sam2TritonOptions(priority: Long = 5) : TritonEncodeOptions(priority)

/**
 * Options for SAM3 Triton encoder.
 */
class Sam3TritonOptions(priority: Long = 5) : TritonEncodeOptions(priority)

/**
 * Options for SAM3 Tracker Triton encoder.
 */
class Sam3TrackerTritonOptions(priority: Long = 5) : TritonEncodeOptions(priority)

/**
 * SAM1 encoder result - only image embeddings.
 *
 * @property imageEmbedding image embeddings tensor, shape (1, 256, 64, 64)
 * @property sessionId the session ID used for this encoding request
 */
data class Sam1EncoderResult(
    val imageEmbedding: OnnxTensor,
    val sessionId: String,
    override val imageWidth: Int = 1024,
    override val imageHeight: Int = 1024
) : EncoderResult {
    override val inputSize: Int get() = 1024
    override fun close() = imageEmbedding.close()
}

/**
 * SAM2/2.1 encoder result - image embeddings + high resolution features.
 *
 * @property imageEmbedding image embeddings tensor, shape (1, 256, 64, 64)
 * @property highResFeats0 high resolution features level 0, shape (1, 32, 256, 256)
 * @property highResFeats1 high resolution features level 1, shape (1, 64, 128, 128)
 */
data class Sam2EncoderResult(
    val imageEmbedding: OnnxTensor,
    val highResFeats0: OnnxTensor,
    val highResFeats1: OnnxTensor,
    override val imageWidth: Int = 1024,
    override val imageHeight: Int = 1024
) : EncoderResult {
    override val inputSize: Int get() = 1024
    override fun close() {
        imageEmbedding.close()
        highResFeats0.close()
        highResFeats1.close()
    }
}

/**
 * SAM3 Tracker encoder result - image embeddings at three scales.
 *
 * @property imageEmbeddings0 level 0 embeddings, shape (1, 32, 288, 288)
 * @property imageEmbeddings1 level 1 embeddings, shape (1, 64, 144, 144)
 * @property imageEmbeddings2 level 2 embeddings, shape (1, 256, 72, 72)
 */
data class Sam3TrackerEncoderResult(
    val imageEmbeddings0: OnnxTensor,
    val imageEmbeddings1: OnnxTensor,
    val imageEmbeddings2: OnnxTensor,
    override val imageWidth: Int = 1008,
    override val imageHeight: Int = 1008
) : EncoderResult {
    override val inputSize: Int get() = 1008
    override fun close() {
        imageEmbeddings0.close()
        imageEmbeddings1.close()
        imageEmbeddings2.close()
    }
}

/**
 * Create an OnnxTensor from a FloatArray using a direct buffer.
 */
fun createTensorDirect(
    data: FloatArray,
    shape: LongArray
): OnnxTensor {
    val byteSize = data.size * 4
    val directBuffer = ByteBuffer.allocateDirect(byteSize).order(ByteOrder.nativeOrder())
    directBuffer.asFloatBuffer().put(data)
    directBuffer.position(0)
    return OnnxTensor.createTensor(ORT_ENV, directBuffer.asFloatBuffer(), shape)
}

/**
 * Compute the content dimensions after [resizeImageWithPadding].
 * Only downscales if the image is larger than [targetSize]; otherwise returns original dimensions.
 */
fun contentDimensions(imageWidth: Int, imageHeight: Int, targetSize: Int): Pair<Int, Int> {
    val maxEdge = maxOf(imageWidth, imageHeight)
    return if (maxEdge > targetSize) {
        val scale = targetSize.toFloat() / maxEdge
        Pair((imageWidth * scale).toInt(), (imageHeight * scale).toInt())
    } else {
        Pair(imageWidth, imageHeight)
    }
}

/**
 * Place image into a [targetSize]×[targetSize] square, padding with black.
 * If the image is larger than [targetSize] on either side, it is downscaled
 * so the longest edge fits. If already smaller, it is placed at the top-left
 * without upscaling.
 */
fun resizeImageWithPadding(image: BufferedImage, targetSize: Int): BufferedImage {
    if (image.width == targetSize && image.height == targetSize)
        return image
    val srcW = image.width
    val srcH = image.height
    val maxEdge = maxOf(srcW, srcH)
    val drawW: Int
    val drawH: Int
    if (maxEdge > targetSize) {
        val scale = targetSize.toFloat() / maxEdge
        drawW = (srcW * scale).toInt()
        drawH = (srcH * scale).toInt()
    } else {
        drawW = srcW
        drawH = srcH
    }

    return BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB)
        .apply {
            createGraphics().apply {
                drawImage(image, 0, 0, drawW, drawH, null)
            }.dispose()
        }
}