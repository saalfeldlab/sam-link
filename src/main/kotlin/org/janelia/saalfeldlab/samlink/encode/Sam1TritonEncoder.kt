package org.janelia.saalfeldlab.samlink.encode

import com.google.protobuf.ByteString
import com.google.protobuf.UnsafeByteOperations
import org.janelia.saalfeldlab.samlink.TritonClient
import java.awt.image.BufferedImage
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * SAM1 encoder using Triton Inference Server.
 *
 * Encodes images into embeddings for the SAM1 decoder.
 *
 * @property serviceHost Triton server hostname
 * @property encoderModel model name on the Triton server (e.g., "sam_vit_b_multi_encoder")
 * @property grpcPort Triton server gRPC port
 * @property responseTimeout timeout in milliseconds for inference requests
 */
class Sam1TritonEncoder(
    private val serviceHost: String,
    private val encoderModel: String,
    private val grpcPort: Int = 8001,
    private val responseTimeout: Int = 30_000
) : SamEncoder<Sam1EncoderResult, Sam1TritonOptions> {

    private val client = TritonClient(serviceHost, grpcPort, responseTimeout.toLong())

    override val inputSize: Int = INPUT_SIZE

    override fun options(): Sam1TritonOptions = Sam1TritonOptions()

    override suspend fun encode(image: BufferedImage, options: Sam1TritonOptions): Sam1EncoderResult {
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

        val embeddingData = response.getFloatArray(OUTPUT_NAME)
            ?: throw IllegalStateException("Missing $OUTPUT_NAME in response")

        val (contentWidth, contentHeight) = contentDimensions(image.width, image.height, INPUT_SIZE)
        return Sam1EncoderResult(
            imageEmbedding = createTensorDirect(embeddingData, longArrayOf(1, 256, 64, 64)),
            sessionId = "",
            imageWidth = contentWidth,
            imageHeight = contentHeight
        )
    }

    override fun close() {
        client.close()
    }

    override suspend fun isReady(): Boolean {
        return client.isModelReady(encoderModel)
    }

    private fun prepareImageData(image: BufferedImage): ByteString {
        val resized = resizeImageWithPadding(image, Sam2TritonEncoder.INPUT_SIZE)
        val dataBuffer = resized.raster.dataBuffer
        val numPixels = Sam2TritonEncoder.INPUT_SIZE * Sam2TritonEncoder.INPUT_SIZE
        val rgbCount = 3 * numPixels
        val buffer = ByteBuffer.allocate(rgbCount * 4).order(ByteOrder.LITTLE_ENDIAN)
        val floatView = buffer.asFloatBuffer()

        /* reorder to channel-height-width format */
        repeat(numPixels) { redIdx ->
            val greenIdx = redIdx + numPixels
            val blueIdx = redIdx + numPixels * 2

            val rgbInt = dataBuffer.getElem(redIdx)
            val red = ((rgbInt shr 16) and 0xFF) / 255f
            val green = ((rgbInt shr 8) and 0xFF) / 255f
            val blue = (rgbInt and 0xFF) / 255f

            floatView.put(redIdx, red)
            floatView.put(greenIdx, green)
            floatView.put(blueIdx, blue)
        }

        return UnsafeByteOperations.unsafeWrap(buffer)
    }

    private fun prepareImageData2(image: BufferedImage): ByteString {
        val resized = resizeImageWithPadding(image, INPUT_SIZE)

        val imageData = ByteArray(3 * INPUT_SIZE * INPUT_SIZE)
        for (c in 0 until 3) {
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    val rgb = resized.getRGB(x, y)
                    val value = when (c) {
                        0 -> (rgb shr 16) and 0xFF
                        1 -> (rgb shr 8) and 0xFF
                        else -> rgb and 0xFF
                    }
                    imageData[c * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] = value.toByte()
                }
            }
        }

        return UnsafeByteOperations.unsafeWrap(imageData)
    }

    companion object {
        const val INPUT_SIZE = 1024
        const val INPUT_NAME = "image"
        const val OUTPUT_NAME = "image_embeddings"
    }
}
