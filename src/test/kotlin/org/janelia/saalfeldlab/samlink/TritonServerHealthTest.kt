package org.janelia.saalfeldlab.samlink

import kotlinx.coroutines.runBlocking
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

            val sam2Model = TritonEnv.sam2Model()
            assertTrue(client.isModelReady(sam2Model), "Sam2 model $sam2Model not ready on $host:$port")

            val sam3Model = TritonEnv.sam3TrackerModel()
            assertTrue(client.isModelReady(sam3Model), "Sam3Tracker model $sam3Model not ready on $host:$port")
        }
    }
}
