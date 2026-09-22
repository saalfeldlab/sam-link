package org.janelia.saalfeldlab.samlink

import ai.onnxruntime.OnnxJavaType
import kotlinx.coroutines.runBlocking
import org.janelia.saalfeldlab.samlink.TestUtils.rectangleImage
import org.janelia.saalfeldlab.samlink.decode.DecoderModel
import org.janelia.saalfeldlab.samlink.decode.Sam1Decoder
import org.janelia.saalfeldlab.samlink.decode.Sam2Decoder
import org.janelia.saalfeldlab.samlink.encode.Sam1TritonOptions
import org.janelia.saalfeldlab.samlink.encode.Sam2TritonOptions
import org.janelia.saalfeldlab.samlink.models.Sam1Model
import org.janelia.saalfeldlab.samlink.models.Sam2Model
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Each decoder converts the embeddings to whatever its own graph declares, and the two bundled
 * decoders want different things: SAM1 takes single precision and SAM2 half. So each endpoint gets
 * asked for both datatypes, and one of the two answers has to be converted either way.
 */
@Tag("integration")
class EmbeddingDatatypeTest {

    /** the fraction of mask pixels that must land on the same side of the threshold */
    private fun assertMasksAgree(label: String, expected: FloatArray, actual: FloatArray) {
        val agree = expected.indices.count { (expected[it] > 0f) == (actual[it] > 0f) }
        assertTrue(
            agree.toDouble() / expected.size > 0.999,
            "$label: ${expected.size - agree} of ${expected.size} mask pixels differed"
        )
    }

    @Test
    fun `sam1 widens a half precision response for its fp32 decoder`() = runBlocking {
        val edgeSize = Sam1Model.Encoder.INPUT_EDGE_SIZE.toInt()
        val image = rectangleImage(edgeSize, edgeSize)
        val prompt = rectangleTestPrompt(edgeSize, edgeSize, borderPercent = 0.25)

        TritonEnv.newSam1Encoder().use { encoder ->
            Sam1Decoder(DecoderModel.SAM1.load()).use { decoder ->
                encoder.encode(image, Sam1TritonOptions(requestFp16 = false)).use { single ->
                    encoder.encode(image, Sam1TritonOptions(requestFp16 = true)).use { half ->
                        assertEquals(OnnxJavaType.FLOAT, single.imageEmbedding.info.type)
                        assertEquals(OnnxJavaType.FLOAT16, half.imageEmbedding.info.type)
                        assertMasksAgree(
                            "sam1",
                            decoder.decode(single, prompt).bestMask,
                            decoder.decode(half, prompt).bestMask
                        )
                    }
                }
            }
        }
    }

    @Test
    fun `sam2 narrows a single precision response for its fp16 decoder`() = runBlocking {
        val edgeSize = Sam2Model.Encoder.INPUT_EDGE_SIZE.toInt()
        val image = rectangleImage(edgeSize, edgeSize)
        val prompt = rectangleTestPrompt(edgeSize, edgeSize, borderPercent = 0.25)

        TritonEnv.newSam2Encoder().use { encoder ->
            Sam2Decoder(DecoderModel.SAM2.load()).use { decoder ->
                encoder.encode(image, Sam2TritonOptions(requestFp16 = false)).use { single ->
                    encoder.encode(image, Sam2TritonOptions(requestFp16 = true)).use { half ->
                        assertEquals(OnnxJavaType.FLOAT, single.imageEmbedding.info.type)
                        assertEquals(OnnxJavaType.FLOAT16, half.imageEmbedding.info.type)
                        assertMasksAgree(
                            "sam2",
                            decoder.decode(single, prompt).bestMask,
                            decoder.decode(half, prompt).bestMask
                        )
                    }
                }
            }
        }
    }
}
