package org.janelia.saalfeldlab.samlink.encode.triton

import org.janelia.saalfeldlab.samlink.TritonClient
import org.janelia.saalfeldlab.samlink.encode.Normalization
import org.janelia.saalfeldlab.samlink.encode.Sam1EncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam1TritonOptions
import org.janelia.saalfeldlab.samlink.models.EncodeParameter
import org.janelia.saalfeldlab.samlink.models.EncodeParameter.Companion.getAsTensor
import org.janelia.saalfeldlab.samlink.models.Sam1Model
import org.janelia.saalfeldlab.samlink.models.Sam1Model.Encoder.Inputs
import org.janelia.saalfeldlab.samlink.models.Sam1Model.Encoder.Outputs
import java.awt.image.BufferedImage

/**
 * SAM1 encoder using Triton Inference Server.
 *
 * Encodes images into embeddings for SAM1.
 */
class Sam1TritonEncoder : SamTritonEncoder<Sam1EncoderResult, Sam1TritonOptions> {

    constructor(client: TritonClient, model: String) : super(client, model)
    constructor(host: String, port: Int, model: String, responseTimeout: Long = 30_000) : super(host, port, model, responseTimeout)

    override val inputEdgeSize = Sam1Model.Encoder.INPUT_EDGE_SIZE
    override val rawInput: EncodeParameter = Inputs.IMAGE
    override val jpegInput: EncodeParameter = Inputs.JPEG_IMAGE
    override val normalization = Normalization.IMAGENET

    override fun options(): Sam1TritonOptions = Sam1TritonOptions()

    override suspend fun encode(image: BufferedImage, options: Sam1TritonOptions): Sam1EncoderResult {

        val fitted = fitImage(image)
        val response = infer(inputFor(fitted, options), options)

        return Sam1EncoderResult(
            imageEmbedding = response.getAsTensor(Outputs.IMAGE_EMBEDDINGS),
            scaledWidth = fitted.scaledWidth,
            scaledHeight = fitted.scaledHeight,
            sourceWidth = image.width,
            sourceHeight = image.height,
        )
    }

}
