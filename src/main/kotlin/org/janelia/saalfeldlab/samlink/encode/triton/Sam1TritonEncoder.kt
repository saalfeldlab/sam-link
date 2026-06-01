package org.janelia.saalfeldlab.samlink.encode.triton

import org.janelia.saalfeldlab.samlink.TritonClient
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.intRGBtoCHW
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleToMaxEdgeSize
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleWithPadding
import org.janelia.saalfeldlab.samlink.models.EncodeParameter.Companion.getAsTensor
import org.janelia.saalfeldlab.samlink.encode.Sam1EncoderResult
import org.janelia.saalfeldlab.samlink.models.Sam1Model
import org.janelia.saalfeldlab.samlink.models.Sam1Model.Encoder.Inputs
import org.janelia.saalfeldlab.samlink.models.Sam1Model.Encoder.Outputs
import org.janelia.saalfeldlab.samlink.encode.Sam1TritonOptions
import java.awt.image.BufferedImage

/**
 * SAM1 encoder using Triton Inference Server.
 *
 * Encodes images into embeddings for SAM1.
 */
class Sam1TritonEncoder : SamTritonEncoder<Sam1EncoderResult, Sam1TritonOptions> {

    constructor(client: TritonClient, model: String) : super(client, model)
    constructor(host: String, port: Int, model: String, responseTimeout: Long = 30_000) : super(host, port, model, responseTimeout)

    override fun options(): Sam1TritonOptions = Sam1TritonOptions()

    override suspend fun encode(image: BufferedImage, options: Sam1TritonOptions): Sam1EncoderResult {

        val maxEdgeSize = Sam1Model.Encoder.INPUT_EDGE_SIZE.toInt()
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
            scaledWidth = scaledWidth,
            scaledHeight = scaledHeight,
            sourceWidth = image.width,
            sourceHeight = image.height,
        )
    }

}