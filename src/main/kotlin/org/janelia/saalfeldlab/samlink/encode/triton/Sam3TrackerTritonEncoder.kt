package org.janelia.saalfeldlab.samlink.encode.triton

import org.janelia.saalfeldlab.samlink.models.Sam3TrackerModel
import org.janelia.saalfeldlab.samlink.models.Sam3TrackerModel.Encoder.Inputs
import org.janelia.saalfeldlab.samlink.models.Sam3TrackerModel.Encoder.Outputs
import org.janelia.saalfeldlab.samlink.TritonClient
import org.janelia.saalfeldlab.samlink.models.EncodeParameter.Companion.getAsTensor
import org.janelia.saalfeldlab.samlink.encode.Sam3TrackerEncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam3TrackerTritonOptions
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.intRGBtoCHW
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleToMaxEdgeSize
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleWithPadding
import java.awt.image.BufferedImage

/**
 * SAM3 Tracker encoder using Triton Inference Server.
 *
 * Encodes images into multiscale embeddings for the SAM3 tracker decoder.
 */
class Sam3TrackerTritonEncoder : SamTritonEncoder<Sam3TrackerEncoderResult, Sam3TrackerTritonOptions> {

    constructor(client: TritonClient, model: String) : super(client, model)
    constructor(host: String, port: Int, model: String, responseTimeout: Long = 30_000) : super(host, port, model, responseTimeout)

    override fun options(): Sam3TrackerTritonOptions = Sam3TrackerTritonOptions()

    override suspend fun encode(image: BufferedImage, options: Sam3TrackerTritonOptions): Sam3TrackerEncoderResult {

        val maxEdgeSize = Sam3TrackerModel.Encoder.INPUT_EDGE_SIZE.toInt()

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

        return Sam3TrackerEncoderResult(
            imageEmbeddings0 = response.getAsTensor(Outputs.IMAGE_EMBED_0),
            imageEmbeddings1 = response.getAsTensor(Outputs.IMAGE_EMBED_1),
            imageEmbeddings2 = response.getAsTensor(Outputs.IMAGE_EMBED_2),
            scaledWidth = scaledWidth,
            scaledHeight = scaledHeight,
            sourceWidth = image.width,
            sourceHeight = image.height,
        )
    }
}