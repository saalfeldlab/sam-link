package org.janelia.saalfeldlab.samlink

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import org.junit.jupiter.api.Test
import java.nio.FloatBuffer
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

class OnnxTensorsTest {

    private val values = floatArrayOf(0f, 1f, -1f, 0.5f, -2.117904f, 7.141f, 0.000061f)
    private val shape = longArrayOf(1, 7)

    private fun <T> withFloatTensor(block: (OnnxTensor) -> T) =
        OnnxTensor.createTensor(ORT_ENV, FloatBuffer.wrap(values), shape).use(block)

    @Test
    fun `the same type is not copied`() = withFloatTensor { tensor ->
        assertSame(tensor, tensor.asType(OnnxJavaType.FLOAT))
    }

    @Test
    fun `half precision survives the round trip within its own precision`() = withFloatTensor { tensor ->
        tensor.asType(OnnxJavaType.FLOAT16).use { half ->
            assertEquals(OnnxJavaType.FLOAT16, half.info.type)
            assertContentEquals(shape, half.info.shape)
            half.asType(OnnxJavaType.FLOAT).use { widened ->
                val out = FloatArray(values.size).also { widened.floatBuffer.get(it) }
                /* fp16 keeps 10 mantissa bits, so ~1e-3 relative */
                for (i in values.indices)
                    assertEquals(values[i], out[i], 0.005f, "index $i")
            }
        }
    }

    @Test
    fun `an unsupported conversion is rejected`() = withFloatTensor { tensor ->
        assertFailsWith<IllegalArgumentException> { tensor.asType(OnnxJavaType.INT64) }
    }
}
