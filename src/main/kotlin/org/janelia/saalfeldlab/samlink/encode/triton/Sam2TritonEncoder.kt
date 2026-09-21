package org.janelia.saalfeldlab.samlink.encode.triton

import org.janelia.saalfeldlab.samlink.InferenceInput
import org.janelia.saalfeldlab.samlink.TritonClient
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.asTritonBytesElement
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.toJpegByteString
import org.janelia.saalfeldlab.samlink.encode.ImageEncoding
import org.janelia.saalfeldlab.samlink.encode.Normalization
import org.janelia.saalfeldlab.samlink.encode.Sam2EncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam2TritonOptions
import org.janelia.saalfeldlab.samlink.models.EncodeParameter
import org.janelia.saalfeldlab.samlink.models.EncodeParameter.Companion.getAsTensor
import org.janelia.saalfeldlab.samlink.models.Sam2Model
import org.janelia.saalfeldlab.samlink.models.Sam2Model.Encoder.Inputs
import org.janelia.saalfeldlab.samlink.models.Sam2Model.Encoder.Outputs
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

    override val inputEdgeSize = Sam2Model.Encoder.INPUT_EDGE_SIZE
    override val rawInput: EncodeParameter = Inputs.IMAGE
    override val normalization = Normalization.IMAGENET

    override fun options(): Sam2TritonOptions = Sam2TritonOptions(ImageEncoding.RAW)

    override suspend fun encode(image: BufferedImage, options: Sam2TritonOptions): Sam2EncoderResult {

        val fitted = fitImage(image)
        val input = when (options.imageEncoding) {
            ImageEncoding.RAW -> rawInputFor(fitted)
            ImageEncoding.JPEG -> InferenceInput(
                name = Inputs.JPEG_IMAGE.parameter,
                shape = Inputs.JPEG_IMAGE.shape,
                datatype = "BYTES",
                data = fitted.image.toJpegByteString(options.quality).asTritonBytesElement()
            )
        }
        val response = infer(input, options)

        return Sam2EncoderResult(
            imageEmbedding = response.getAsTensor(Outputs.IMAGE_EMBED),
            highResFeats0 = response.getAsTensor(Outputs.HIGH_RES_FEATS_0),
            highResFeats1 = response.getAsTensor(Outputs.HIGH_RES_FEATS_1),
            scaledWidth = fitted.scaledWidth,
            scaledHeight = fitted.scaledHeight,
            sourceWidth = image.width,
            sourceHeight = image.height,
        )
    }
}
