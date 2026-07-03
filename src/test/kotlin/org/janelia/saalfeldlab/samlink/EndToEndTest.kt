package org.janelia.saalfeldlab.samlink

import kotlinx.coroutines.runBlocking
import org.janelia.saalfeldlab.samlink.decode.DecoderModel
import org.janelia.saalfeldlab.samlink.decode.Sam1Decoder
import org.janelia.saalfeldlab.samlink.decode.Sam2Decoder
import org.janelia.saalfeldlab.samlink.decode.Sam3TrackerDecoder
import org.janelia.saalfeldlab.samlink.encode.ImageEncoding
import org.janelia.saalfeldlab.samlink.encode.Sam2TritonOptions
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource

@Tag("integration")
class EndToEndTest {

    enum class ImageDimensions(val width: Int, val height: Int) {
        SQUARE(1024, 1024),
        SAM3_SQUARE(1008, 1008),
        SMALL_SQUARE(512, 512),
        LARGE_SQUARE(2048, 2048),
        LARGE_WIDER(2048, 1024),
        LARGE_TALLER(1024, 2048),
        SMALL_WIDER(512, 256),
        SMALL_TALLER(256, 512)
    }

    @ParameterizedTest
    @EnumSource(ImageDimensions::class)
    fun `Sam 1`(imageDims: ImageDimensions) = runBlocking {
        val encoder = TritonEnv.newSam1Encoder()
        val decoder = Sam1Decoder(DecoderModel.SAM1.load())

        testSquareWithBorder(
            width = imageDims.width,
            height = imageDims.height,
            borderPercent = 0.25,
            encoder = encoder,
            decode = decoder::decode
        )
    }

    @ParameterizedTest
    @EnumSource(ImageDimensions::class)
    fun `Sam 2 Raw`(imageDims: ImageDimensions) = runBlocking {
        val encoder = TritonEnv.newSam2Encoder(ImageEncoding.RAW)
        val decoder = Sam2Decoder(DecoderModel.SAM2.load())

        testSquareWithBorder(
            width = imageDims.width,
            height = imageDims.height,
            borderPercent = 0.25,
            encoder = encoder,
            decode = decoder::decode
        )
    }

    @Test
    fun `Sam 2 JPEG`() = runBlocking {
        val encoder = TritonEnv.newSam2Encoder(ImageEncoding.JPEG)
        val decoder = Sam2Decoder(DecoderModel.SAM2.load())

        /* the client always pads to a 1024x1024 input before encoding, so the JPEG
         * satisfies the server's fixed-size contract for any source dimensions */
        testSquareWithBorder(
            width = ImageDimensions.SQUARE.width,
            height = ImageDimensions.SQUARE.height,
            borderPercent = 0.25,
            encoder = encoder,
            decode = decoder::decode,
            options = Sam2TritonOptions(imageEncoding = ImageEncoding.JPEG),
        )
    }

    @ParameterizedTest
    @EnumSource(ImageDimensions::class)
    fun `Sam 3`(imageDims: ImageDimensions) = runBlocking {
        val encoder = TritonEnv.newSam3TrackerEncoder()
        val decoder = Sam3TrackerDecoder(DecoderModel.SAM3_TRACKER_FP16.load())

        testSquareWithBorder(
            width = imageDims.width, height = imageDims.height,
            decodedImageEdge = 288,
            borderPercent = 0.25,
            encoder = encoder,
            decode = decoder::decode
        )
    }

}
