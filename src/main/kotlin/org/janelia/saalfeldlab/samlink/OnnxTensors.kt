package org.janelia.saalfeldlab.samlink

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import java.lang.Float
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * This tensor as [type], or itself when it already matches.
 *
 * An endpoint's datatype and the decoder's need not agree; the fp16 SAM2 decoder takes half precision
 * embeddings where the others take single. Only those two convert.
 *
 * The result is a new tensor whenever a conversion happened, so the caller owns and must close it;
 * compare identity against the receiver to tell.
 */
fun OnnxTensor.asType(type: OnnxJavaType): OnnxTensor {
    if (info.type == type)
        return this

    val shape = info.shape
    return when {
        info.type == OnnxJavaType.FLOAT16 && type == OnnxJavaType.FLOAT -> {
            val halves = shortBuffer
            val floats = FloatArray(halves.remaining()) { Float.float16ToFloat(halves.get(it)) }
            val direct = ByteBuffer.allocateDirect(floats.size * 4).order(ByteOrder.nativeOrder())
            direct.asFloatBuffer().put(floats)
            OnnxTensor.createTensor(ORT_ENV, direct.asFloatBuffer(), shape)
        }
        info.type == OnnxJavaType.FLOAT && type == OnnxJavaType.FLOAT16 -> {
            val floats = floatBuffer
            val direct = ByteBuffer.allocateDirect(floats.remaining() * 2).order(ByteOrder.nativeOrder())
            val halves = direct.asShortBuffer()
            for (i in 0 until floats.remaining())
                halves.put(i, Float.floatToFloat16(floats.get(i)))
            OnnxTensor.createTensor(ORT_ENV, halves, shape, OnnxJavaType.FLOAT16)
        }
        else -> throw IllegalArgumentException("Cannot convert ${info.type} to $type")
    }
}
