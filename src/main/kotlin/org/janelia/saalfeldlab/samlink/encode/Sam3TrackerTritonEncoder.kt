package org.janelia.saalfeldlab.samlink.encode

import com.google.protobuf.ByteString
import com.google.protobuf.UnsafeByteOperations
import org.janelia.saalfeldlab.samlink.TritonClient
import java.awt.image.BufferedImage
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * SAM3 Tracker encoder using Triton Inference Server.
 *
 * Encodes images into multi-scale embeddings for the SAM3 tracker decoder.
 * Uses [-1, 1] normalization (value / 127.5 - 1.0).
 *
 * @property serviceHost Triton server hostname
 * @property encoderModel model name on the Triton server
 * @property grpcPort Triton server gRPC port
 * @property responseTimeout timeout in milliseconds for inference requests
 */
class Sam3TrackerTritonEncoder(
    private val serviceHost: String,
    private val encoderModel: String,
    private val grpcPort: Int = 8001,
    private val responseTimeout: Int = 30_000
) : SamEncoder<Sam3TrackerEncoderResult, Sam3TrackerTritonOptions> {

    private val client = TritonClient(serviceHost, grpcPort, responseTimeout.toLong())

    override val inputSize: Int = INPUT_SIZE

    override fun options(): Sam3TrackerTritonOptions = Sam3TrackerTritonOptions()

    override suspend fun encode(image: BufferedImage, options: Sam3TrackerTritonOptions): Sam3TrackerEncoderResult {
        val imageData = prepareImageData(image)

        val inputs = listOf(
            TritonClient.InferInput(
                name = INPUT_NAME,
                datatype = "FP32",
                shape = listOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
                data = imageData
            )
        )

        val response = client.infer(encoderModel, inputs)

        val emb0 = response.getFloatArray(OUTPUT_EMB_0)
            ?: throw IllegalStateException("Missing $OUTPUT_EMB_0 in response")
        val emb1 = response.getFloatArray(OUTPUT_EMB_1)
            ?: throw IllegalStateException("Missing $OUTPUT_EMB_1 in response")
        val emb2 = response.getFloatArray(OUTPUT_EMB_2)
            ?: throw IllegalStateException("Missing $OUTPUT_EMB_2 in response")

        val (contentWidth, contentHeight) = contentDimensions(image.width, image.height, INPUT_SIZE)
        return Sam3TrackerEncoderResult(
            imageEmbeddings0 = createTensorDirect(emb0, longArrayOf(1, 32, 288, 288)),
            imageEmbeddings1 = createTensorDirect(emb1, longArrayOf(1, 64, 144, 144)),
            imageEmbeddings2 = createTensorDirect(emb2, longArrayOf(1, 256, 72, 72)),
            imageWidth = contentWidth,
            imageHeight = contentHeight
        )
    }

    override fun close() {
        client.close()
    }

    override suspend fun isReady() = client.isModelReady(encoderModel)

    private fun prepareImageData(image: BufferedImage): ByteString {
        val resized = resizeImageWithPadding(image, INPUT_SIZE)
        val dataBuffer = resized.raster.dataBuffer
        val numPixels = INPUT_SIZE * INPUT_SIZE
        val rgbCount = 3 * numPixels
        val buffer = ByteBuffer.allocate(rgbCount * 4).order(ByteOrder.LITTLE_ENDIAN)
        val floatView = buffer.asFloatBuffer()

        /* reorder to channel-height-width format */
        repeat(numPixels) { redIdx ->
            val greenIdx = redIdx + numPixels
            val blueIdx = redIdx + numPixels * 2

            val rgbInt = dataBuffer.getElem(redIdx)
            val red = ((rgbInt shr 16) and 0xFF) / 127.5f - 1f
            val green = ((rgbInt shr 8) and 0xFF) / 127.5f - 1f
            val blue = (rgbInt and 0xFF) / 127.5f - 1f

            floatView.put(redIdx, red)
            floatView.put(greenIdx, green)
            floatView.put(blueIdx, blue)
        }

        return UnsafeByteOperations.unsafeWrap(buffer)
    }

    companion object {
        const val INPUT_SIZE = 1008
        const val INPUT_NAME = "pixel_values"
        const val OUTPUT_EMB_0 = "image_embeddings.0"
        const val OUTPUT_EMB_1 = "image_embeddings.1"
        const val OUTPUT_EMB_2 = "image_embeddings.2"
    }
}
