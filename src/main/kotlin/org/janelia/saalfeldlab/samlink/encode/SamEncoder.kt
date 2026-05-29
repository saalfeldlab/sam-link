package org.janelia.saalfeldlab.samlink.encode

import ai.onnxruntime.OnnxTensor
import com.google.protobuf.ByteString
import com.google.protobuf.UnsafeByteOperations
import org.janelia.saalfeldlab.samlink.encode.triton.Sam1TritonEncoder
import org.janelia.saalfeldlab.samlink.encode.triton.Sam2TritonEncoder
import org.janelia.saalfeldlab.samlink.encode.triton.Sam3TrackerTritonEncoder
import java.awt.image.BufferedImage
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

/**
 * Interface for SAM image encoders.
 *
 * @param O the options type for this encoder
 */
interface SamEncoder<R : EncoderResult, O : EncodeOptions> : AutoCloseable {

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
     * Check if the encoder service is ready for inference.
     */
    suspend fun isReady(): Boolean
}

/**
 * Base class for encoder results.
 * EncoderResult may hold closeable references, and should be closed when no longer needed.
 *
 * [imageWidth] and [imageHeight] are the dimensions of the image content
 * within the [inputSize]×[inputSize] padded square, after resizing.
 * The content occupies the top-left [imageWidth]×[imageHeight] region;
 * the remainder is padding.
 */
sealed interface EncoderResult : AutoCloseable {

    /** required [inputSize]x[inputSize] padded input */
    val inputSize: Long

    /** width of the image content in the padded [inputSize]x[inputSize] encoded input */
    val imageWidth: Int

    /** height of the image content in the padded [inputSize]x[inputSize] encoded input */
    val imageHeight: Int
}

/**
 * SAM1 encoder result
 *
 * @property imageEmbedding image embeddings tensor, shape (1, 256, 64, 64)
 */
data class Sam1EncoderResult(
    val imageEmbedding: OnnxTensor,
    override val imageWidth: Int,
    override val imageHeight: Int
) : EncoderResult {
    override val inputSize = Sam1TritonEncoder.INPUT_SIZE
    override fun close() = imageEmbedding.close()
}

/**
 * SAM2 encoder result
 *
 * @property imageEmbedding image embeddings tensor, shape (1, 256, 64, 64)
 * @property highResFeats0 high resolution features level 0, shape (1, 32, 256, 256)
 * @property highResFeats1 high resolution features level 1, shape (1, 64, 128, 128)
 */
data class Sam2EncoderResult(
    val imageEmbedding: OnnxTensor,
    val highResFeats0: OnnxTensor,
    val highResFeats1: OnnxTensor,
    override val imageWidth: Int,
    override val imageHeight: Int
) : EncoderResult {

    override val inputSize = Sam2TritonEncoder.INPUT_SIZE
    override fun close() {
        imageEmbedding.close()
        highResFeats0.close()
        highResFeats1.close()
    }
}

/**
 * SAM3 Tracker encoder result.
 *
 * @property imageEmbeddings0 level 0 embeddings, shape (1, 32, 288, 288)
 * @property imageEmbeddings1 level 1 embeddings, shape (1, 64, 144, 144)
 * @property imageEmbeddings2 level 2 embeddings, shape (1, 256, 72, 72)
 */
data class Sam3TrackerEncoderResult(
    val imageEmbeddings0: OnnxTensor,
    val imageEmbeddings1: OnnxTensor,
    val imageEmbeddings2: OnnxTensor,
    override val imageWidth: Int,
    override val imageHeight: Int
) : EncoderResult {
    override val inputSize = Sam3TrackerTritonEncoder.INPUT_SIZE
    override fun close() {
        imageEmbeddings0.close()
        imageEmbeddings1.close()
        imageEmbeddings2.close()
    }
}

/**
 * Given an image with dimensions [imageWidth]x[imageHeight], calculate the size of the image data
 * after scaling such that the longest edge becomes [maxEdgeSize]. The resulting dimensions may be
 * larger or smaller than the original dimensions, but will maintain the aspect ratio.
 */
fun scaleToMaxEdgeSize(imageWidth: Int, imageHeight: Int, maxEdgeSize: Int): Pair<Int, Int> {
    val actualMaxEdge = maxOf(imageWidth, imageHeight)

    if (actualMaxEdge == maxEdgeSize)
        return imageWidth to imageHeight

    val scale = maxEdgeSize.toFloat() / actualMaxEdge.toFloat()

    return when {
        imageWidth == imageHeight -> maxEdgeSize to maxEdgeSize
        imageWidth > imageHeight -> maxEdgeSize to (imageHeight * scale).toInt()
        else  -> (imageWidth * scale).toInt() to maxEdgeSize
    }
}

/**
 * Scale [image] to a resulting [BufferedImage] of the [targetWidth]×[targetHeight].
 * Maintains the aspect ratio and pads with black to fit the target dimensions.
 * If [image] dimensions match the target dimensions this will return the original image.
 */
fun scaleWithPadding(image: BufferedImage, contentWidth: Int, contentHeight: Int, targetWidth: Int, targetHeight: Int): BufferedImage {
    if (image.width == targetWidth && image.height == targetHeight)
        return image

    return BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB).apply {
        val graphics = createGraphics()
        graphics.drawImage(image, 0, 0, contentWidth, contentHeight, null)
        graphics.dispose()
    }
}

/**
 * Convert a [BufferedImage] to a [ByteString] in channel-height-width format (CHW).
 * Assumes input is [BufferedImage.TYPE_INT_RGB].
 *
 * Maps RGB values from [0,255] to [outputRange]
 *
 * @param outputRange range to normalize pixel values to, default is [0f, 1f]
 */
fun BufferedImage.intRGBtoCHW(outputRange : ClosedFloatingPointRange<Float> = 0f..1f): ByteString {
    val dataBuffer = raster.dataBuffer
    val numPixels = width * height
    val rgbCount = 3 * numPixels
    val buffer = ByteBuffer.allocate(rgbCount * 4).order(ByteOrder.LITTLE_ENDIAN)
    val floatView = buffer.asFloatBuffer()

    val rgbRange = 255f
    val scale = (outputRange.endInclusive - outputRange.start) / rgbRange
    val offset = outputRange.start

    /* reorder to channel-height-width format and normalize to output range*/
    repeat(numPixels) { redIdx ->
        val greenIdx = redIdx + numPixels
        val blueIdx = redIdx + numPixels * 2

        val rgbInt = dataBuffer.getElem(redIdx)
        val red = ((rgbInt shr 16) and 0xFF) * scale + offset
        val green = ((rgbInt shr 8) and 0xFF) * scale + offset
        val blue = (rgbInt and 0xFF) * scale + offset

        floatView.put(redIdx, red)
        floatView.put(greenIdx, green)
        floatView.put(blueIdx, blue)
    }

    return UnsafeByteOperations.unsafeWrap(buffer)
}