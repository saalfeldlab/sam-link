package org.janelia.saalfeldlab.samlink.models

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import org.janelia.saalfeldlab.samlink.ORT_ENV
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer

interface ModelParameter {

    val parameter: String
    val shape: LongArray

    /**
     * Create an OnnxTensor from a FloatArray backed by a direct buffer.
     */
    fun allocateDirectTensor(data: FloatArray, shape : LongArray = this.shape): OnnxTensor {
        val byteSize = data.size * 4
        val directBuffer = ByteBuffer.allocateDirect(byteSize).order(ByteOrder.nativeOrder())
        directBuffer.asFloatBuffer().put(data)
        directBuffer.position(0)
        return OnnxTensor.createTensor(ORT_ENV, directBuffer.asFloatBuffer(), shape)
    }



    /**
     * Create a half precision OnnxTensor from the wire bytes, without widening.
     *
     * [bytes] holds the little endian fp16 or bf16 payload exactly as the server sent it, so this is
     * a bulk copy rather than a per element conversion.
     */
    fun allocateDirectHalfTensor(
        bytes: ByteBuffer,
        type: OnnxJavaType = OnnxJavaType.FLOAT16,
        shape: LongArray = this.shape
    ): OnnxTensor {
        val directBuffer = ByteBuffer.allocateDirect(bytes.remaining()).order(ByteOrder.nativeOrder())
        directBuffer.put(bytes.duplicate())
        directBuffer.position(0)
        return OnnxTensor.createTensor(ORT_ENV, directBuffer.asShortBuffer(), shape, type)
    }

    /** Wrap a FloatArray as an OnnxTensor */
    fun wrapAsTensor(data: FloatArray, shape : LongArray = this.shape): OnnxTensor {

        return OnnxTensor.createTensor(ORT_ENV, FloatBuffer.wrap(data), shape)
    }

    /** Wrap an IntArray as an OnnxTensor */
    fun wrapAsTensor(data: IntArray, shape : LongArray = this.shape): OnnxTensor {

        return OnnxTensor.createTensor(ORT_ENV, IntBuffer.wrap(data), shape)
    }

    /** Wrap a LongArray as an OnnxTensor */
    fun wrapAsTensor(data: LongArray, shape : LongArray = this.shape): OnnxTensor {

        return OnnxTensor.createTensor(ORT_ENV, LongBuffer.wrap(data), shape)
    }

    companion object {
        operator fun MutableMap<String, OnnxTensor>.set(key: ModelParameter, value: OnnxTensor) {
            this[key.parameter] = value
        }
    }

}