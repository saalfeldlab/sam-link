package org.janelia.saalfeldlab.samlink.models

import inference.GrpcService
import java.nio.ByteOrder

interface EncodeParameter : ModelParameter {

    companion object {

        fun GrpcService.ModelInferResponse.getAsTensor(encodeParam: EncodeParameter) =
            encodeParam.allocateDirectTensor(getFloatArray(encodeParam.parameter))

        /**
         * Get output tensor by name as a float array.
         *
         * Throws IllegalArgumentException if output [name] is not in the response.
         */
        fun GrpcService.ModelInferResponse.getFloatArray(name: String) =
            (outputsList zip rawOutputContentsList)
                .firstOrNull { (output, _) -> output.name == name }
                ?.let { (output, content) ->
                    check(output.datatype != "FP16") { "FP16 datatype is not currently supported" }

                    val buffer = content.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN)
                    FloatArray(buffer.remaining() / 4).also { buffer.asFloatBuffer().get(it) }
                }
                ?: throw IllegalArgumentException("Missing output '$name' in inference response")
    }
}