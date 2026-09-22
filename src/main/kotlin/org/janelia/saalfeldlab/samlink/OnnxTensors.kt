package org.janelia.saalfeldlab.samlink

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import java.lang.Float
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * This tensor as [type], or itself when it already matches.
 *
 * An endpoint's datatype and the decoder's need not agree; the fp16 SAM2 decoder takes half precision
 * embeddings where the others take single. Converting between the two 16-bit types goes through single
 * precision, since they lay out their bits differently.
 *
 * The result is a new tensor whenever a conversion happened, so the caller owns and must close it;
 * compare identity against the receiver to tell.
 */
fun OnnxTensor.asType(type: OnnxJavaType): OnnxTensor {
    if (info.type == type)
        return this

    val floats = widened()
    return when (type) {
        OnnxJavaType.FLOAT -> {
            val direct = ByteBuffer.allocateDirect(floats.remaining() * 4).order(ByteOrder.nativeOrder())
            direct.asFloatBuffer().put(floats)
            OnnxTensor.createTensor(ORT_ENV, direct.asFloatBuffer(), info.shape)
        }
        OnnxJavaType.FLOAT16 -> narrowed(floats, type) { Float.floatToFloat16(it) }
        OnnxJavaType.BFLOAT16 -> narrowed(floats, type) { it.toBFloat16() }
        else -> throw IllegalArgumentException("Cannot convert ${info.type} to $type")
    }
}

/** this tensor's elements as single precision, whatever width they are stored at */
private fun OnnxTensor.widened(): FloatBuffer = when (info.type) {
    OnnxJavaType.FLOAT -> floatBuffer
    OnnxJavaType.FLOAT16 -> shortBuffer.let { halves ->
        FloatBuffer.wrap(FloatArray(halves.remaining()) { Float.float16ToFloat(halves.get(it)) })
    }
    OnnxJavaType.BFLOAT16 -> shortBuffer.let { halves ->
        FloatBuffer.wrap(FloatArray(halves.remaining()) { halves.get(it).bFloat16ToFloat() })
    }
    else -> throw IllegalArgumentException("Cannot read ${info.type} as single precision")
}

private fun OnnxTensor.narrowed(floats: FloatBuffer, type: OnnxJavaType, narrow: (kotlin.Float) -> Short): OnnxTensor {
    val direct = ByteBuffer.allocateDirect(floats.remaining() * 2).order(ByteOrder.nativeOrder())
    val halves = direct.asShortBuffer()
    for (i in 0 until floats.remaining())
        halves.put(i, narrow(floats.get(i)))
    return OnnxTensor.createTensor(ORT_ENV, halves, info.shape, type)
}

/** bfloat16 is the top half of the single precision bit pattern, so widening is a shift */
private fun Short.bFloat16ToFloat() = Float.intBitsToFloat(toInt() shl 16)

/** the top half of the bit pattern, rounded to nearest even rather than truncated */
private fun kotlin.Float.toBFloat16(): Short {
    if (isNaN())
        return 0x7FC0.toShort()
    val bits = Float.floatToIntBits(this)
    return ((bits + 0x7FFF + ((bits ushr 16) and 1)) ushr 16).toShort()
}
