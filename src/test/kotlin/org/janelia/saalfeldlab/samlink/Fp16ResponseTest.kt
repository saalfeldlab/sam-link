package org.janelia.saalfeldlab.samlink

import inference.GrpcService
import inference.inferParameter
import kotlinx.coroutines.runBlocking
import org.janelia.saalfeldlab.samlink.TestUtils.rectangleImage
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.intRGBtoCHW
import org.janelia.saalfeldlab.samlink.encode.Normalization
import org.janelia.saalfeldlab.samlink.models.EncodeParameter.Companion.getFloatArray
import org.junit.jupiter.api.Tag
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * `request_fp16` halves the response on every encoder, and the values survive the round trip.
 *
 * The parameter is read off the wire here rather than through the encoders, which widen half
 * precision back to single on arrival and so cannot show which datatype was sent.
 */
@Tag("integration")
class Fp16ResponseTest {

    enum class Encoder(val model: String, val input: String, val edgeSize: Int, val normalization: Normalization, val outputs: List<String>) {
        SAM1("sam1_encoder", "image", 1024, Normalization.IMAGENET, listOf("image_embeddings")),
        SAM2(
            "sam2.1_large_encoder", "image", 1024, Normalization.IMAGENET,
            listOf("image_embed", "high_res_feats_0", "high_res_feats_1")
        ),
        SAM3(
            "sam3_tracker_encoder_fp16", "pixel_values", 1008, Normalization.SYMMETRIC,
            listOf("image_embeddings.0", "image_embeddings.1", "image_embeddings.2")
        )
    }

    @ParameterizedTest
    @EnumSource(Encoder::class)
    fun `request_fp16 halves the response without losing the values`(encoder: Encoder) = runBlocking {
        val square = rectangleImage(encoder.edgeSize, encoder.edgeSize)
        val input = InferenceInput(
            name = encoder.input,
            shape = longArrayOf(1, 3, encoder.edgeSize.toLong(), encoder.edgeSize.toLong()),
            datatype = "FP32",
            data = square.intRGBtoCHW(encoder.normalization)
        )

        TritonEnv.newClient().use { client ->
            val singlePrecision = client.infer(encoder.model, listOf(input))
            val halfPrecision = client.infer(
                encoder.model,
                listOf(input),
                params = mapOf("request_fp16" to inferParameter { boolParam = true })
            )

            for (name in encoder.outputs) {
                assertEquals("FP32", singlePrecision.datatypeOf(name), "$name without the parameter")
                assertEquals("FP16", halfPrecision.datatypeOf(name), "$name with request_fp16")
                assertEquals(
                    singlePrecision.bytesOf(name) / 2, halfPrecision.bytesOf(name),
                    "$name should be half the bytes"
                )

                /* fp16 keeps 10 mantissa bits, so a value lands within ~5e-4 of its single precision self */
                val expected = singlePrecision.getFloatArray(name)
                val actual = halfPrecision.getFloatArray(name)
                val peak = expected.maxOf { abs(it) }
                val worst = expected.indices.maxOf { abs(expected[it] - actual[it]) }
                assertTrue(worst / peak <= 1e-3f, "$name drifted by $worst against a peak of $peak")
            }
        }
    }

    private fun GrpcService.ModelInferResponse.datatypeOf(name: String) =
        outputsList.first { it.name == name }.datatype

    private fun GrpcService.ModelInferResponse.bytesOf(name: String) =
        rawOutputContentsList[outputsList.indexOfFirst { it.name == name }].size()
}
