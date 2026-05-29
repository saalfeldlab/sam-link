package org.janelia.saalfeldlab.samlink.encode

import ai.onnxruntime.OnnxTensor
import inference.GrpcService
import org.janelia.saalfeldlab.samlink.ORT_ENV
import org.janelia.saalfeldlab.samlink.TritonClient
import java.nio.ByteBuffer
import java.nio.ByteOrder

interface EncodeParameter {

    val parameter: String
    val shape: LongArray

    companion object {

        fun GrpcService.ModelInferResponse.getAsTensor(parameter: EncodeParameter) =
            createTensor(getFloatArray(parameter.parameter), parameter.shape)

        /**
         * Create an OnnxTensor from a FloatArray backed by a direct buffer.
         */
        fun createTensor(data: FloatArray, shape: LongArray): OnnxTensor {
            val byteSize = data.size * 4
            val directBuffer = ByteBuffer.allocateDirect(byteSize).order(ByteOrder.nativeOrder())
            directBuffer.asFloatBuffer().put(data)
            directBuffer.position(0)
            return OnnxTensor.createTensor(ORT_ENV, directBuffer.asFloatBuffer(), shape)
        }

        /**
         * Get output tensor by name as a float array.
         * Automatically handles FP16 to FP32 conversion if needed.
         *
         * Throws IllegalArgumentException if output [name] is not in the response.
         *
         * @return FloatArray or throws IllegalArgumentException
         */
        fun GrpcService.ModelInferResponse.getFloatArray(name: String) =
            (outputsList zip rawOutputContentsList)
                .firstOrNull { (output, _) -> output.name == name }
                ?.let { (output, content) ->

                    //TODO: could I use output.contents somehow instead of rawOutputContentsList?
                    // maybe output.contents.toByteArray() ? or something else ?
                    val buffer = content.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN)

                    if (output.datatype == "FP16")
                        TritonClient.fp16BytesToFloatArray(buffer)
                    else
                        FloatArray(buffer.remaining() / 4).also { buffer.asFloatBuffer().get(it) }
                }
                ?: throw IllegalArgumentException("Missing output '$name' in inference response")
    }
}
