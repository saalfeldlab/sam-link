package org.janelia.saalfeldlab.samlink.models

import ai.onnxruntime.OnnxTensor
import inference.GrpcService
import java.nio.ByteOrder

interface EncodeParameter : ModelParameter {

    companion object {

        /**
         * Get output tensor by name. If the decoder requires fp16 or fp32, and the
         * encode result returned the other, support auto-converting for those types.
         *
         * If received and expected types are the same, no conversion takes place.
         */
        fun GrpcService.ModelInferResponse.getAsTensor(encodeParam: EncodeParameter): OnnxTensor {
            val name = encodeParam.parameter
            val (output, content) = (outputsList zip rawOutputContentsList)
                .firstOrNull { (output, _) -> output.name == name }
                ?: throw IllegalArgumentException("Missing output '$name' in inference response")

            return when (val datatype = output.datatype) {
                "FP32" -> encodeParam.allocateDirectTensor(getFloatArray(name))
                "FP16" -> encodeParam.allocateDirectHalfTensor(
                    content.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN)
                )
                else -> throw IllegalArgumentException(
                    "Unsupported datatype '$datatype' for output '$name'"
                )
            }
        }

        /**
         * Get output tensor by name as a float array.
         *
         * Half precision is widened here. To keep it use [getAsTensor], which preserves whatever the
         * endpoint sent.
         *
         * Throws IllegalArgumentException if output [name] is not in the response.
         */
        fun GrpcService.ModelInferResponse.getFloatArray(name: String) =
            (outputsList zip rawOutputContentsList)
                .firstOrNull { (output, _) -> output.name == name }
                ?.let { (output, content) ->
                    val buffer = content.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN)
                    when (val datatype = output.datatype) {
                        "FP32" -> FloatArray(buffer.remaining() / 4).also { buffer.asFloatBuffer().get(it) }
                        "FP16" -> buffer.asShortBuffer().let { halves ->
                            FloatArray(halves.remaining()) { java.lang.Float.float16ToFloat(halves.get(it)) }
                        }
                        else -> throw IllegalArgumentException(
                            "Unsupported datatype '$datatype' for output '$name'"
                        )
                    }
                }
                ?: throw IllegalArgumentException("Missing output '$name' in inference response")
    }
}