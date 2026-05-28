package org.janelia.saalfeldlab.samlink

import kotlinx.coroutines.runBlocking
import org.janelia.saalfeldlab.samlink.TestUtils.squareImage
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import kotlin.test.assertContentEquals

/**
 * basic test just to verify the outputs of the models used to encode, and the shape of the returned outputs.
 */
@Tag("integration")
class EncoderOutputTest {

    @Test
    fun `Sam 1`() = runBlocking {
        TritonEnv.newSam1Encoder().use { encoder ->
            val image = squareImage(encoder.inputSize, encoder.inputSize)
            encoder.encode(image).use { result ->
                assertContentEquals(longArrayOf(1, 256, 64, 64), result.imageEmbedding.info.shape)
            }
        }
    }

    @Test
    fun `Sam 2`() = runBlocking {
        TritonEnv.newSam2Encoder().use { encoder ->
            val image = squareImage(encoder.inputSize, encoder.inputSize)
            encoder.encode(image).use { result ->
                assertContentEquals(longArrayOf(1, 256, 64, 64), result.imageEmbedding.info.shape)
                assertContentEquals(longArrayOf(1, 32, 256, 256), result.highResFeats0.info.shape)
                assertContentEquals(longArrayOf(1, 64, 128, 128), result.highResFeats1.info.shape)
            }
        }
    }

    @Test
    fun `Sam 3 Tracker`() = runBlocking {
        TritonEnv.newSam3TrackerEncoder().use { encoder ->
            val image = squareImage(encoder.inputSize, encoder.inputSize)
            encoder.encode(image).use { result ->
                assertContentEquals(longArrayOf(1, 32, 288, 288), result.imageEmbeddings0.info.shape)
                assertContentEquals(longArrayOf(1, 64, 144, 144), result.imageEmbeddings1.info.shape)
                assertContentEquals(longArrayOf(1, 256, 72, 72), result.imageEmbeddings2.info.shape)
            }
        }
    }
}
