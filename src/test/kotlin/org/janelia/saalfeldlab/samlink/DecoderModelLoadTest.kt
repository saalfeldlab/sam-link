package org.janelia.saalfeldlab.samlink

import org.janelia.saalfeldlab.samlink.decode.DecoderModel
import org.junit.jupiter.api.assertNotNull
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource

/**
 * resource-packaging test
 */
class DecoderModelLoadTest {

    @ParameterizedTest
    @EnumSource(DecoderModel::class)
    fun `each bundled decoder model loads successfully`(model: DecoderModel) {
        model.load().use { session ->
            assertNotNull(session.inputInfo) { "Failed to load Decoder mode: $model"}
        }
    }
}
