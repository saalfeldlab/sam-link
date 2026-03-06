package org.janelia.saalfeldlab.samlink.encode

import com.google.protobuf.ByteString
import com.google.protobuf.UnsafeByteOperations
import inference.inferParameter
import io.github.oshai.kotlinlogging.KotlinLogging
import org.janelia.saalfeldlab.samlink.TritonClient
import java.awt.image.BufferedImage
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * SAM2/2.1 encoder using Triton Inference Server.
 *
 * Encodes images into embeddings and high-resolution features for SAM2.1 decoder.
 *
 * @property serviceHost Triton server hostname
 * @property encoderModel model name on the Triton server (e.g., "sam2.1_large_encoder")
 * @property grpcPort Triton server gRPC port
 * @property responseTimeout timeout in milliseconds for inference requests
 */
class Sam2TritonEncoder(
    private val serviceHost: String,
    private val encoderModel: String,
    private val grpcPort: Int = 8001,
    private val responseTimeout: Int = 30_000,
) : SamEncoder<Sam2EncoderResult, Sam2TritonOptions> {

    private val client = TritonClient(
        host = serviceHost,
        port = grpcPort,
        timeoutMs = responseTimeout.toLong(),
    )

    override val inputSize: Int = INPUT_SIZE

    override fun options(): Sam2TritonOptions = Sam2TritonOptions()

    override suspend fun encode(image: BufferedImage, options: Sam2TritonOptions): Sam2EncoderResult {
        val imageData = prepareImageData(image)

        val inputs = listOf(
            TritonClient.InferInput(
                name = INPUT_NAME,
                datatype = "FP32",
                shape = listOf(1L, 3L, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
                data = imageData,
            )
        )

        val params = mapOf("priority" to inferParameter { int64Param = options.priority })

        val response = client.infer(encoderModel, inputs, params)

        val imageEmbedData = response.getFloatArray(OUTPUT_IMAGE_EMBED)
            ?: throw IllegalStateException("Missing $OUTPUT_IMAGE_EMBED in response")
        val highResFeats0Data = response.getFloatArray(OUTPUT_HIGH_RES_FEATS_0)
            ?: throw IllegalStateException("Missing $OUTPUT_HIGH_RES_FEATS_0 in response")
        val highResFeats1Data = response.getFloatArray(OUTPUT_HIGH_RES_FEATS_1)
            ?: throw IllegalStateException("Missing $OUTPUT_HIGH_RES_FEATS_1 in response")

        val (contentWidth, contentHeight) = contentDimensions(image.width, image.height, INPUT_SIZE)
        return Sam2EncoderResult(
            imageEmbedding = createTensorDirect(imageEmbedData, longArrayOf(1, 256, 64, 64)),
            highResFeats0 = createTensorDirect(highResFeats0Data, longArrayOf(1, 32, 256, 256)),
            highResFeats1 = createTensorDirect(highResFeats1Data, longArrayOf(1, 64, 128, 128)),
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
            val red = ((rgbInt shr 16) and 0xFF) / 255f
            val green = ((rgbInt shr 8) and 0xFF) / 255f
            val blue = (rgbInt and 0xFF) / 255f

            floatView.put(redIdx, red)
            floatView.put(greenIdx, green)
            floatView.put(blueIdx, blue)
        }

        return UnsafeByteOperations.unsafeWrap(buffer)
    }

    companion object {
        private val LOG = KotlinLogging.logger {}
        const val INPUT_SIZE = 1024
        const val INPUT_NAME = "image"
        const val OUTPUT_IMAGE_EMBED = "image_embed"
        const val OUTPUT_HIGH_RES_FEATS_0 = "high_res_feats_0"
        const val OUTPUT_HIGH_RES_FEATS_1 = "high_res_feats_1"
    }
}
