package org.janelia.saalfeldlab.samlink.encode.triton

import org.janelia.saalfeldlab.samlink.TritonClient
import org.janelia.saalfeldlab.samlink.encode.EncodeParameter
import org.janelia.saalfeldlab.samlink.encode.EncodeParameter.Companion.getAsTensor
import org.janelia.saalfeldlab.samlink.encode.Sam3TrackerEncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam3TrackerTritonOptions
import org.janelia.saalfeldlab.samlink.encode.intRGBtoCHW
import org.janelia.saalfeldlab.samlink.encode.scaleToMaxEdgeSize
import org.janelia.saalfeldlab.samlink.encode.scaleWithPadding
import java.awt.image.BufferedImage

/**
 * SAM3 Tracker encoder using Triton Inference Server.
 *
 * Encodes images into multi-scale embeddings for the SAM3 tracker decoder.
 * Uses [-1, 1] normalization (value / 127.5 - 1.0).
 *
 * @property org.janelia.saalfeldlab.samlink.encode.triton.SamTritonEncoder.client Triton Client
 * @property org.janelia.saalfeldlab.samlink.encode.triton.SamTritonEncoder.model model name on the Triton server
 */
class Sam3TrackerTritonEncoder : SamTritonEncoder<Sam3TrackerEncoderResult, Sam3TrackerTritonOptions> {

    constructor(client: TritonClient, model: String) : super(client, model)
    constructor(host: String, port: Int, model: String, responseTimeout: Long = 30_000) : super(host, port, model, responseTimeout)

    override fun options(): Sam3TrackerTritonOptions = Sam3TrackerTritonOptions()

    override suspend fun encode(image: BufferedImage, options: Sam3TrackerTritonOptions): Sam3TrackerEncoderResult {

        val maxEdgeSize = INPUT_SIZE.toInt()
        val (scaledWidth, scaledHeight) = scaleToMaxEdgeSize(image.width, image.height, maxEdgeSize)
        val scaledPaddedImg = scaleWithPadding(image, scaledWidth, scaledHeight, maxEdgeSize, maxEdgeSize)

        val imageData = scaledPaddedImg.intRGBtoCHW(outputRange = -1f..1f)

        val inputs = listOf(
            TritonClient.InferenceInput(
                name = Inputs.PIXEL_VALUES.parameter,
                shape = Inputs.PIXEL_VALUES.shape,
                datatype = "FP32",
                data = imageData
            ),
        )


        val response = client.infer(model, inputs)

        val (contentWidth, contentHeight) = scaleToMaxEdgeSize(image.width, image.height, INPUT_SIZE.toInt())
        return Sam3TrackerEncoderResult(
            imageEmbeddings0 = response.getAsTensor(Outputs.IMAGE_EMBED_0),
            imageEmbeddings1 = response.getAsTensor(Outputs.IMAGE_EMBED_1),
            imageEmbeddings2 = response.getAsTensor(Outputs.IMAGE_EMBED_2),
            imageWidth = contentWidth,
            imageHeight = contentHeight
        )
    }

    companion object {
        const val INPUT_SIZE = 1008L

        private enum class Inputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            PIXEL_VALUES("pixel_values", longArrayOf(1L, 3L, INPUT_SIZE, INPUT_SIZE)),
        }

        private enum class Outputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            IMAGE_EMBED_0("image_embeddings.0", longArrayOf(1, 32, 288, 288)),
            IMAGE_EMBED_1("image_embeddings.1", longArrayOf(1, 64, 144, 144)),
            IMAGE_EMBED_2("image_embeddings.2", longArrayOf(1, 256, 72, 72)),
        }

        const val INPUT_NAME = "pixel_values"
        const val OUTPUT_EMB_0 = "image_embeddings.0"
        const val OUTPUT_EMB_1 = "image_embeddings.1"
        const val OUTPUT_EMB_2 = "image_embeddings.2"
    }
}