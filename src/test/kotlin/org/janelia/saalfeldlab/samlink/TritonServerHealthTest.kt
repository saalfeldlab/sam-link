package org.janelia.saalfeldlab.samlink

import kotlinx.coroutines.runBlocking
import org.janelia.saalfeldlab.samlink.encode.ImageEncoding
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import kotlin.test.assertTrue

/**
 * Verification that the configured Triton server is serving the three encoder models.
 */
@Tag("integration")
class TritonServerHealthTest {

    @Test
    fun `each configured encoder model reports ready`() = runBlocking {
        val host = TritonEnv.host()
        val port = TritonEnv.port()
        TritonEnv.newClient().use { client ->
            val sam1Model = TritonEnv.sam1Model()
            assertTrue(client.isModelReady(sam1Model), "Sam1 model $sam1Model not ready on $host:$port")

            val sam2RawModel = TritonEnv.sam2Model(ImageEncoding.RAW)
            assertTrue(client.isModelReady(sam2RawModel), "Sam2 raw model $sam2RawModel not ready on $host:$port")

            val sam2JpegModel = TritonEnv.sam2Model(ImageEncoding.JPEG)
            assertTrue(client.isModelReady(sam2JpegModel), "Sam2 JPEG model $sam2JpegModel not ready on $host:$port")

            val sam3Model = TritonEnv.sam3TrackerModel()
            assertTrue(client.isModelReady(sam3Model), "Sam3Tracker model $sam3Model not ready on $host:$port")
        }
    }
}
