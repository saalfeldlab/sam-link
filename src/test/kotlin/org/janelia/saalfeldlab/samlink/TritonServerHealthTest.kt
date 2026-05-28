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
        TritonEnv.newClient().use { client ->
            assertTrue(client.isModelReady(TritonEnv.sam1Model), "Sam1 model ${TritonEnv.sam1Model} not ready on ${TritonEnv.host}:${TritonEnv.port}")
            assertTrue(client.isModelReady(TritonEnv.sam2Model), "Sam2 model ${TritonEnv.sam2Model} not ready on ${TritonEnv.host}:${TritonEnv.port}")
            assertTrue(client.isModelReady(TritonEnv.sam3TrackerModel), "Sam3Tracker model ${TritonEnv.sam3TrackerModel} not ready on ${TritonEnv.host}:${TritonEnv.port}")
        }
    }
}
