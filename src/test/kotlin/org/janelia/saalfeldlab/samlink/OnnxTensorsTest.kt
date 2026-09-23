package org.janelia.saalfeldlab.samlink

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import org.junit.jupiter.api.Test
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

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
    fun `bfloat16 survives the round trip within its own precision`() = withFloatTensor { tensor ->
        tensor.asType(OnnxJavaType.BFLOAT16).use { wide ->
            assertEquals(OnnxJavaType.BFLOAT16, wide.info.type)
            assertContentEquals(shape, wide.info.shape)
            wide.asType(OnnxJavaType.FLOAT).use { widened ->
                val out = FloatArray(values.size).also { widened.floatBuffer.get(it) }
                /* bf16 keeps 7 mantissa bits, so ~1e-2 relative */
                for (i in values.indices)
                    assertEquals(values[i], out[i], abs(values[i]) * 0.01f + 1e-6f, "index $i")
            }
        }
    }

    @Test
    fun `the two 16 bit types are not interchangeable`() = withFloatTensor { tensor ->
        tensor.asType(OnnxJavaType.FLOAT16).use { half ->
            tensor.asType(OnnxJavaType.BFLOAT16).use { wide ->
                val halfBits = ShortArray(values.size).also { half.shortBuffer.get(it) }
                val wideBits = ShortArray(values.size).also { wide.shortBuffer.get(it) }
                assertFalse(halfBits.contentEquals(wideBits), "same width, different layout")
            }
        }
    }

    @Test
    fun `bfloat16 keeps the exponent range that fp16 cannot`() {
        /* 1e30 overflows fp16, whose largest finite value is 65504 */
        val big = floatArrayOf(1e30f)
        OnnxTensor.createTensor(ORT_ENV, FloatBuffer.wrap(big), longArrayOf(1)).use { tensor ->
            tensor.asType(OnnxJavaType.FLOAT16).use { half ->
                val out = FloatArray(1).also { half.asType(OnnxJavaType.FLOAT).use { w -> w.floatBuffer.get(it) } }
                assertTrue(out[0].isInfinite(), "fp16 should overflow")
            }
            tensor.asType(OnnxJavaType.BFLOAT16).use { wide ->
                val out = FloatArray(1).also { wide.asType(OnnxJavaType.FLOAT).use { w -> w.floatBuffer.get(it) } }
                assertTrue(out[0].isFinite() && out[0] > 9e29f, "bf16 should hold it, got ${out[0]}")
            }
        }
    }

    @Test
    fun `conversion between the 16 bit types goes through single precision`() = withFloatTensor { tensor ->
        tensor.asType(OnnxJavaType.FLOAT16).use { half ->
            half.asType(OnnxJavaType.BFLOAT16).use { wide ->
                assertEquals(OnnxJavaType.BFLOAT16, wide.info.type)
                wide.asType(OnnxJavaType.FLOAT).use { widened ->
                    val out = FloatArray(values.size).also { widened.floatBuffer.get(it) }
                    for (i in values.indices)
                        assertEquals(values[i], out[i], abs(values[i]) * 0.02f + 1e-6f, "index $i")
                }
            }
        }
    }

    @Test
    fun `an unsupported conversion is rejected`() = withFloatTensor { tensor ->
        assertFailsWith<IllegalArgumentException> { tensor.asType(OnnxJavaType.INT64) }
    }
}
