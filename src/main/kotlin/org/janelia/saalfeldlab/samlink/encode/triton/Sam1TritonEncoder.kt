package org.janelia.saalfeldlab.samlink.encode.triton

import org.janelia.saalfeldlab.samlink.TritonClient
import org.janelia.saalfeldlab.samlink.encode.EncodeParameter
import org.janelia.saalfeldlab.samlink.encode.EncodeParameter.Companion.getAsTensor
import org.janelia.saalfeldlab.samlink.encode.Sam1EncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam1TritonOptions
import org.janelia.saalfeldlab.samlink.encode.intRGBtoCHW
import org.janelia.saalfeldlab.samlink.encode.scaleToMaxEdgeSize
import org.janelia.saalfeldlab.samlink.encode.scaleWithPadding
import java.awt.image.BufferedImage

/**
 * SAM1 encoder using Triton Inference Server.
 *
 * Encodes images into embeddings for SAM1.
 *
 * @property org.janelia.saalfeldlab.samlink.encode.triton.SamTritonEncoder.client Triton Client
 * @property org.janelia.saalfeldlab.samlink.encode.triton.SamTritonEncoder.model model name on the Triton server
 */
class Sam1TritonEncoder : SamTritonEncoder<Sam1EncoderResult, Sam1TritonOptions> {

    constructor(client: TritonClient, model: String) : super(client, model)
    constructor(host: String, port: Int, model: String, responseTimeout: Long = 30_000) : super(host, port, model, responseTimeout)

    override fun options(): Sam1TritonOptions = Sam1TritonOptions()

    override suspend fun encode(image: BufferedImage, options: Sam1TritonOptions): Sam1EncoderResult {

        val maxEdgeSize = INPUT_SIZE.toInt()
        val (scaledWidth, scaledHeight) = scaleToMaxEdgeSize(image.width, image.height, maxEdgeSize)
        val scaledPaddedImg = scaleWithPadding(image, scaledWidth, scaledHeight, maxEdgeSize, maxEdgeSize)

        val inputs = listOf(
            TritonClient.InferenceInput(
                name = Inputs.IMAGE.parameter,
                datatype = "FP32",
                shape = Inputs.IMAGE.shape,
                data = scaledPaddedImg.intRGBtoCHW()
            ),
        )

        val response = client.infer(model, inputs)
        val imageEmbedding = response.getAsTensor(Outputs.IMAGE_EMBEDDINGS)

        return Sam1EncoderResult(
            imageEmbedding = imageEmbedding,
            imageWidth = scaledWidth,
            imageHeight = scaledHeight
        )
    }

    companion object {

        const val INPUT_SIZE = 1024L

        private enum class Inputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            IMAGE("image", longArrayOf(1L, 3L, INPUT_SIZE, INPUT_SIZE))
        }

        private enum class Outputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            IMAGE_EMBEDDINGS("image_embeddings", longArrayOf(1, 256, 64, 64))
        }
    }
}