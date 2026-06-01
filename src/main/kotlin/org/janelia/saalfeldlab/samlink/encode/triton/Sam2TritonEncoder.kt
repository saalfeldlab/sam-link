package org.janelia.saalfeldlab.samlink.encode.triton

import inference.inferParameter
import org.janelia.saalfeldlab.samlink.TritonClient
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.intRGBtoCHW
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleToMaxEdgeSize
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleWithPadding
import org.janelia.saalfeldlab.samlink.models.EncodeParameter.Companion.getAsTensor
import org.janelia.saalfeldlab.samlink.encode.Sam2EncoderResult
import org.janelia.saalfeldlab.samlink.models.Sam2Model
import org.janelia.saalfeldlab.samlink.models.Sam2Model.Encoder.Inputs
import org.janelia.saalfeldlab.samlink.models.Sam2Model.Encoder.Outputs
import org.janelia.saalfeldlab.samlink.encode.Sam2TritonOptions
import java.awt.image.BufferedImage

/**
 * SAM2 encoder using Triton Inference Server.
 *
 * Encodes images into embeddings for SAM2.
 */
class Sam2TritonEncoder : SamTritonEncoder<Sam2EncoderResult, Sam2TritonOptions> {

    constructor(client: TritonClient, model: String) : super(client, model)
    constructor(host: String, port: Int, model: String, responseTimeout: Long = 30_000) : super(
        host,
        port,
        model,
        responseTimeout
    )

    override fun options(): Sam2TritonOptions = Sam2TritonOptions()

    override suspend fun encode(image: BufferedImage, options: Sam2TritonOptions): Sam2EncoderResult {

        val maxEdgeSize = Sam2Model.Encoder.INPUT_EDGE_SIZE.toInt()
        val (scaledWidth, scaledHeight) = scaleToMaxEdgeSize(
            image.width,
            image.height,
            maxEdgeSize
        )
        val scaledPaddedImg = scaleWithPadding(
            image,
            scaledWidth,
            scaledHeight,
            maxEdgeSize,
            maxEdgeSize
        )


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

        return Sam2EncoderResult(
            imageEmbedding = response.getAsTensor(Outputs.IMAGE_EMBED),
            highResFeats0 = response.getAsTensor(Outputs.HIGH_RES_FEATS_0),
            highResFeats1 = response.getAsTensor(Outputs.HIGH_RES_FEATS_1),
            scaledWidth = scaledWidth,
            scaledHeight = scaledHeight,
            sourceWidth = image.width,
            sourceHeight = image.height,
        )
    }
}