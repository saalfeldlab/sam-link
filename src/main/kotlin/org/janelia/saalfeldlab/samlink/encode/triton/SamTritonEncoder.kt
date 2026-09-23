package org.janelia.saalfeldlab.samlink.encode.triton

import inference.GrpcService
import inference.inferParameter
import org.janelia.saalfeldlab.samlink.InferenceInput
import org.janelia.saalfeldlab.samlink.TritonClient
import org.janelia.saalfeldlab.samlink.UnaryTritonClient
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.asTritonBytesElement
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.intRGBtoCHW
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleToMaxEdgeSize
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleWithPadding
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.toJpegByteString
import org.janelia.saalfeldlab.samlink.encode.EncoderResult
import org.janelia.saalfeldlab.samlink.encode.ImageEncoding
import org.janelia.saalfeldlab.samlink.encode.Normalization
import org.janelia.saalfeldlab.samlink.encode.SamEncoder
import org.janelia.saalfeldlab.samlink.encode.TritonEncodeOptions
import org.janelia.saalfeldlab.samlink.models.EncodeParameter
import java.awt.image.BufferedImage

/**
 * An image scaled and padded to the size this endpoint expects, and where the content ended up
 * inside it. The decode side needs the content extent to map prompts back to source pixels.
 */
internal class FittedImage(val image: BufferedImage, val scaledWidth: Int, val scaledHeight: Int)

abstract class SamTritonEncoder<R : EncoderResult, O : TritonEncodeOptions> : SamEncoder<R, O> {

    protected val client: TritonClient
    protected val model: String

    /** the input edge this endpoint requires */
    protected abstract val inputEdgeSize: Long

    /** the FP32 CHW input */
    protected abstract val rawInput: EncodeParameter

    /** the BYTES input holding the image fit to [inputEdgeSize] and JPEG encoded */
    protected abstract val jpegInput: EncodeParameter

    protected abstract val normalization: Normalization

    var responseTimeout: Long
        get() = client.timeoutMs
        set(value) {
            client.timeoutMs = value
        }

    constructor(client: TritonClient, model: String) {
        this.client = client
        this.model = model
    }

    constructor(
        host: String,
        port: Int = 8001,
        model: String,
        responseTimeout: Long = 30_000
    ) : this(UnaryTritonClient(host, port), model) {
        this.responseTimeout = responseTimeout
    }

    /** Scale [image] to the size this endpoint expects, padding to keep the aspect ratio */
    internal fun fitImage(image: BufferedImage): FittedImage {
        val maxEdgeSize = inputEdgeSize.toInt()
        val (scaledWidth, scaledHeight) = scaleToMaxEdgeSize(image.width, image.height, maxEdgeSize)
        val fitted = scaleWithPadding(image, scaledWidth, scaledHeight, maxEdgeSize, maxEdgeSize)
        return FittedImage(fitted, scaledWidth, scaledHeight)
    }

    /**
     * [fitted] as this endpoint's image input.
     *
     * Both inputs are optional server side and exactly one is sent. The raw path normalizes here; the
     * JPEG path leaves that to the server, which applies the same constants after decoding.
     */
    internal fun inputFor(fitted: FittedImage, options: TritonEncodeOptions) = when (options.imageEncoding) {
        ImageEncoding.RAW -> InferenceInput(
            name = rawInput.parameter,
            shape = rawInput.shape,
            datatype = "FP32",
            data = fitted.image.intRGBtoCHW(normalization)
        )
        ImageEncoding.JPEG -> InferenceInput(
            name = jpegInput.parameter,
            shape = jpegInput.shape,
            datatype = "BYTES",
            data = fitted.image.toJpegByteString(options.quality).asTritonBytesElement()
        )
    }

    internal suspend fun infer(input: InferenceInput, options: TritonEncodeOptions): GrpcService.ModelInferResponse {
        val params = buildMap {
            if (options.requestFp16)
                put("request_fp16", inferParameter { boolParam = true })
        }
        return client.infer(model, listOf(input), options.priority, params)
    }

    override suspend fun isReady() = client.isModelReady(model)

    override fun close() {
        client.close()
    }
}
