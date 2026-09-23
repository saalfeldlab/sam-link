package org.janelia.saalfeldlab.samlink

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.TensorInfo
import org.janelia.saalfeldlab.samlink.decode.DecoderModel
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertNotNull
import kotlin.test.assertEquals
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource

/**
 * test load of the bundled models
 */
class DecoderModelLoadTest {

    @ParameterizedTest
    @EnumSource(DecoderModel::class)
    fun `each bundled decoder model loads successfully`(model: DecoderModel) {
        model.load().use { session ->
            assertNotNull(session.inputInfo) { "Failed to load Decoder mode: $model"}
        }
    }

    @Test
    fun `the sam2 decoder takes fp16 embeddings`() {
        DecoderModel.SAM2.load().use { session ->
            for (name in listOf("image_embed", "high_res_feats_0", "high_res_feats_1"))
                assertEquals(OnnxJavaType.FLOAT16, (session.inputInfo.getValue(name).info as TensorInfo).type, name)
            assertEquals(OnnxJavaType.INT64, (session.inputInfo.getValue("orig_im_size").info as TensorInfo).type)
        }
    }
}
