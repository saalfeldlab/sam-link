package org.janelia.saalfeldlab.samlink.encode.triton

import org.janelia.saalfeldlab.samlink.TritonClient
import org.janelia.saalfeldlab.samlink.encode.Normalization
import org.janelia.saalfeldlab.samlink.encode.Sam3TrackerEncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam3TrackerTritonOptions
import org.janelia.saalfeldlab.samlink.models.EncodeParameter
import org.janelia.saalfeldlab.samlink.models.EncodeParameter.Companion.getAsTensor
import org.janelia.saalfeldlab.samlink.models.Sam3TrackerModel
import org.janelia.saalfeldlab.samlink.models.Sam3TrackerModel.Encoder.Inputs
import org.janelia.saalfeldlab.samlink.models.Sam3TrackerModel.Encoder.Outputs
import java.awt.image.BufferedImage

/**
 * SAM3 Tracker encoder using Triton Inference Server.
 *
 * Encodes images into multiscale embeddings for the SAM3 tracker decoder.
 */
class Sam3TrackerTritonEncoder : SamTritonEncoder<Sam3TrackerEncoderResult, Sam3TrackerTritonOptions> {

    constructor(client: TritonClient, model: String) : super(client, model)
    constructor(host: String, port: Int, model: String, responseTimeout: Long = 30_000) : super(host, port, model, responseTimeout)

    override val inputEdgeSize = Sam3TrackerModel.Encoder.INPUT_EDGE_SIZE
    override val rawInput: EncodeParameter = Inputs.PIXEL_VALUES
    override val normalization = Normalization.SYMMETRIC

    override fun options(): Sam3TrackerTritonOptions = Sam3TrackerTritonOptions()

    override suspend fun encode(image: BufferedImage, options: Sam3TrackerTritonOptions): Sam3TrackerEncoderResult {

        val fitted = fitImage(image)
        val response = infer(rawInputFor(fitted), options)

        return Sam3TrackerEncoderResult(
            imageEmbeddings0 = response.getAsTensor(Outputs.IMAGE_EMBED_0),
            imageEmbeddings1 = response.getAsTensor(Outputs.IMAGE_EMBED_1),
            imageEmbeddings2 = response.getAsTensor(Outputs.IMAGE_EMBED_2),
            scaledWidth = fitted.scaledWidth,
            scaledHeight = fitted.scaledHeight,
            sourceWidth = image.width,
            sourceHeight = image.height,
        )
    }
}
