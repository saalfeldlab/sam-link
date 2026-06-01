package org.janelia.saalfeldlab.samlink.encode

import ai.onnxruntime.OnnxTensor
import org.janelia.saalfeldlab.samlink.models.Sam1Model
import org.janelia.saalfeldlab.samlink.models.Sam2Model
import org.janelia.saalfeldlab.samlink.models.Sam3TrackerModel
import java.awt.image.BufferedImage

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
 * [scaledWidth] and [scaledHeight] are the dimensions of the image content
 * within the [inputSize]×[inputSize] padded square, after resizing.
 * The content occupies the top-left [scaledWidth]×[scaledHeight] region;
 * the remainder is padding.
 */
sealed interface EncoderResult : AutoCloseable {

    /** required [inputSize]x[inputSize] padded input */
    val inputSize: Long

    /** width of the image content in the scaled and padded [inputSize]x[inputSize] encoded input */
    val scaledWidth: Int

    /** height of the image content in the scaled and padded [inputSize]x[inputSize] encoded input */
    val scaledHeight: Int

    /** width of the source image the encoder was given (before any scaling/padding) */
    val sourceWidth: Int

    /** height of the source image the encoder was given (before any scaling/padding) */
    val sourceHeight: Int
}

/**
 * SAM1 encoder result
 *
 * @property imageEmbedding image embeddings tensor, shape (1, 256, 64, 64)
 */
data class Sam1EncoderResult(
    val imageEmbedding: OnnxTensor,
    override val scaledWidth: Int,
    override val scaledHeight: Int,
    override val sourceWidth: Int,
    override val sourceHeight: Int,
) : EncoderResult {
    override val inputSize = Sam1Model.Encoder.INPUT_EDGE_SIZE
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
    override val scaledWidth: Int,
    override val scaledHeight: Int,
    override val sourceWidth: Int,
    override val sourceHeight: Int,
) : EncoderResult {

    override val inputSize = Sam2Model.Encoder.INPUT_EDGE_SIZE
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
    override val scaledWidth: Int,
    override val scaledHeight: Int,
    override val sourceWidth: Int,
    override val sourceHeight: Int,
) : EncoderResult {
    override val inputSize = Sam3TrackerModel.Encoder.INPUT_EDGE_SIZE
    override fun close() {
        imageEmbeddings0.close()
        imageEmbeddings1.close()
        imageEmbeddings2.close()
    }
}

