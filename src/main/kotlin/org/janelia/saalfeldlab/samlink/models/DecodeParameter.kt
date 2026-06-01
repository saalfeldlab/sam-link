package org.janelia.saalfeldlab.samlink.models

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession

interface DecodeParameter : ModelParameter {

    companion object {

        operator fun OrtSession.Result.get(parameter: DecodeParameter) : OnnxTensor {
            return get(parameter.parameter).get() as OnnxTensor
        }
    }
}