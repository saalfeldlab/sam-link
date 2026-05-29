package org.janelia.saalfeldlab.samlink.encode.triton

import inference.inferParameter
import io.github.oshai.kotlinlogging.KotlinLogging
import org.janelia.saalfeldlab.samlink.TritonClient
import org.janelia.saalfeldlab.samlink.encode.EncodeParameter
import org.janelia.saalfeldlab.samlink.encode.EncodeParameter.Companion.getAsTensor
import org.janelia.saalfeldlab.samlink.encode.Sam2EncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam2TritonOptions
import org.janelia.saalfeldlab.samlink.encode.intRGBtoCHW
import org.janelia.saalfeldlab.samlink.encode.scaleToMaxEdgeSize
import org.janelia.saalfeldlab.samlink.encode.scaleWithPadding
import java.awt.image.BufferedImage

/**
 * SAM2 encoder using Triton Inference Server.
 *
 * Encodes images into embeddings for SAM2.
 *
 * @property org.janelia.saalfeldlab.samlink.encode.triton.SamTritonEncoder.client Triton Client
 * @property org.janelia.saalfeldlab.samlink.encode.triton.SamTritonEncoder.model model name on the Triton server
 */
class Sam2TritonEncoder : SamTritonEncoder<Sam2EncoderResult, Sam2TritonOptions> {

    constructor(client: TritonClient, model: String) : super(client, model)
    constructor(host: String, port: Int, model: String, responseTimeout: Long = 30_000) : super(host, port, model, responseTimeout)

    override fun options(): Sam2TritonOptions = Sam2TritonOptions()

    override suspend fun encode(image: BufferedImage, options: Sam2TritonOptions): Sam2EncoderResult {

        val maxEdgeSize = INPUT_SIZE.toInt()
        val (scaledWidth, scaledHeight) = scaleToMaxEdgeSize(image.width, image.height, maxEdgeSize)
        val scaledPaddedImg = scaleWithPadding(image, scaledWidth, scaledHeight, maxEdgeSize, maxEdgeSize)


        val inputs = listOf(
            TritonClient.InferenceInput(
                name = Inputs.IMAGE.parameter,
                shape = Inputs.IMAGE.shape,
                datatype = "FP32",
                data = scaledPaddedImg.intRGBtoCHW()
            ),
        )


        val params = mapOf("priority" to inferParameter { int64Param = options.priority })

        val response = client.infer(model, inputs, params)

        val (contentWidth, contentHeight) = scaleToMaxEdgeSize(image.width, image.height, INPUT_SIZE.toInt())

        return Sam2EncoderResult(
            imageEmbedding = response.getAsTensor(Outputs.IMAGE_EMBED),
            highResFeats0 = response.getAsTensor(Outputs.HIGH_RES_FEATS_0),
            highResFeats1 = response.getAsTensor(Outputs.HIGH_RES_FEATS_1),
            imageWidth = contentWidth,
            imageHeight = contentHeight
        )
    }

    companion object {

        private val LOG = KotlinLogging.logger {}
        const val INPUT_SIZE = 1024L

        private enum class Inputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            IMAGE("image", longArrayOf(1, 3, INPUT_SIZE, INPUT_SIZE)),
        }

        private enum class Outputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            IMAGE_EMBED("image_embed", longArrayOf(1, 256, 64, 64)),
            HIGH_RES_FEATS_0("high_res_feats_0", longArrayOf(1, 32, 256, 256)),
            HIGH_RES_FEATS_1("high_res_feats_1", longArrayOf(1, 64, 128, 128));
        }

    }
}