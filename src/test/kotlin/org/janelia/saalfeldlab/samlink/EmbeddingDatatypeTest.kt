package org.janelia.saalfeldlab.samlink

import ai.onnxruntime.OnnxJavaType
import kotlinx.coroutines.runBlocking
import org.janelia.saalfeldlab.samlink.TestUtils.rectangleImage
import org.janelia.saalfeldlab.samlink.decode.DecoderModel
import org.janelia.saalfeldlab.samlink.decode.Sam2Decoder
import org.janelia.saalfeldlab.samlink.encode.ImageEncoding
import org.janelia.saalfeldlab.samlink.models.Sam2Model
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A decoder converts the embeddings to whatever its own graph declares, so an endpoint returning half
 * precision still feeds a decoder that takes single precision.
 *
 * The narrowing is done here rather than by asking the endpoint for fp16, which is a later change; what
 * matters is that the decoder is handed a datatype it did not ask for.
 */
@Tag("integration")
class EmbeddingDatatypeTest {

    @Test
    fun `an fp32 decoder accepts half precision embeddings`() = runBlocking {
        val edgeSize = Sam2Model.Encoder.INPUT_EDGE_SIZE.toInt()
        val image = rectangleImage(edgeSize, edgeSize)
        val prompt = rectangleTestPrompt(edgeSize, edgeSize, borderPercent = 0.25)

        TritonEnv.newSam2Encoder(ImageEncoding.RAW).use { encoder ->
            encoder.encode(image).use { result ->
                assertEquals(OnnxJavaType.FLOAT, result.imageEmbedding.info.type, "endpoint returns fp32 today")

                val halved = result.copy(
                    imageEmbedding = result.imageEmbedding.asType(OnnxJavaType.FLOAT16),
                    highResFeats0 = result.highResFeats0.asType(OnnxJavaType.FLOAT16),
                    highResFeats1 = result.highResFeats1.asType(OnnxJavaType.FLOAT16),
                )

                Sam2Decoder(DecoderModel.SAM2.load()).use { decoder ->
                    val fromFloat = decoder.decode(result, prompt).bestMask
                    val fromHalf = halved.use { decoder.decode(it, prompt) }.bestMask

                    val agree = fromFloat.indices.count { (fromFloat[it] > 0f) == (fromHalf[it] > 0f) }
                    val worst = fromFloat.indices.maxOf { abs(fromFloat[it] - fromHalf[it]) }
                    assertTrue(
                        agree.toDouble() / fromFloat.size > 0.999,
                        "masks should agree; ${fromFloat.size - agree} pixels differed, worst logit gap $worst"
                    )
                }
            }
        }
    }
}
