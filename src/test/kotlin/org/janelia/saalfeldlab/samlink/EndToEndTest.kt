package org.janelia.saalfeldlab.samlink

import kotlinx.coroutines.runBlocking
import org.janelia.saalfeldlab.samlink.TestUtils.assertDecodedSquareImage
import org.janelia.saalfeldlab.samlink.TestUtils.squareImage
import org.janelia.saalfeldlab.samlink.decode.DecoderModel
import org.janelia.saalfeldlab.samlink.decode.Sam1Decoder
import org.janelia.saalfeldlab.samlink.decode.Sam2Decoder
import org.janelia.saalfeldlab.samlink.decode.Sam3TrackerDecoder
import org.janelia.saalfeldlab.samlink.decode.SamDecoder
import org.janelia.saalfeldlab.samlink.decode.SamPointLabel
import org.janelia.saalfeldlab.samlink.decode.SamPrompt
import org.janelia.saalfeldlab.samlink.encode.EncoderResult
import org.janelia.saalfeldlab.samlink.encode.SamEncoder
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import kotlin.test.assertTrue

@Tag("integration")
class EndToEndTest {

    @Test
    fun `Sam 1`() = runBlocking {
        val encoder = TritonEnv.newSam1Encoder()
        val decoder = Sam1Decoder(DecoderModel.SAM1.load())
        val centerPointPrompt = SamPrompt().addPoint(512f, 512f, SamPointLabel.FOREGROUND)


        testSquareWithBorder(
            encoder = encoder,
            decode = { decoder.decode(it, centerPointPrompt) }
        )
    }

    @Test
    fun `Sam 2`() = runBlocking {
        val encoder = TritonEnv.newSam2Encoder()
        val decoder = Sam2Decoder(DecoderModel.SAM2.load())
        val centerPointPrompt = SamPrompt().addPoint(512f, 512f, SamPointLabel.FOREGROUND)


        testSquareWithBorder(
            encoder = encoder,
            decode = { decoder.decode(it, centerPointPrompt) }
        )
    }

    @Test
    fun `Sam 3`() = runBlocking {

        val encoder = TritonEnv.newSam3TrackerEncoder()
        val decoder = Sam3TrackerDecoder(DecoderModel.SAM3_TRACKER_FP16.load())
        val centerPointPrompt = SamPrompt().addPoint(512f, 512f, SamPointLabel.FOREGROUND)


        testSquareWithBorder(
            imageEdge = 1008,
            decodedImageEdge = 288,
            encoder = encoder,
            decode = { decoder.decode(it, centerPointPrompt) }
        )
    }
    private suspend fun <T : EncoderResult> testSquareWithBorder(
        imageEdge: Int = 1204,
        decodedImageEdge: Int = 256,
        borderPercent: Double = .25,
        encoder: SamEncoder<T, *>,
        decode: (T) -> SamDecoder.DecoderResult,
    ) {
        val image = squareImage(imageEdge, imageEdge, borderPercent = borderPercent)
        encoder.encode(image).use { encodeResult ->
            val decodeResult = decode(encodeResult)
            assertTrue(decodeResult.ious.all { it.isFinite() })
            assertDecodedSquareImage(decodeResult.bestMask, decodedImageEdge, decodedImageEdge, borderPercent)
        }
    }
}
