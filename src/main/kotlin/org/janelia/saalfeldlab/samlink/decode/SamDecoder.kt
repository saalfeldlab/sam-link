package org.janelia.saalfeldlab.samlink.decode

import org.janelia.saalfeldlab.samlink.encode.EncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam1EncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam2EncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam3TrackerEncoderResult
import kotlin.math.ln


/**
 * Base interface for SAM decoder implementations.
 *
 * Decoders accept an [EncoderResult] and a [SamPrompt] and output a [DecoderResult]
 *
 */
sealed interface SamDecoder<E : EncoderResult> : AutoCloseable {

    fun decode(encoderResult: E, prompt: SamPrompt): DecoderResult

    /** Whether this decoder supports iterative mask refinement via mask_input */
    val supportsMaskRefinement: Boolean get() = false

    companion object {

        /** Convert binary mask to logits;    */
        fun binaryToLogits(mask: FloatArray): FloatArray {
            return FloatArray(mask.size) { i ->
                val p = if (mask[i] > 0.5f) 1f - 1e-3f else 1e-3f
                ln(p / (1f - p))
            }
        }

        fun matches(decoder: SamDecoder<*>, encoderResult: EncoderResult): Boolean {
            return when (decoder) {
                is Sam1Decoder -> encoderResult is Sam1EncoderResult
                is Sam2Decoder -> encoderResult is Sam2EncoderResult
                is Sam3TrackerDecoder -> encoderResult is Sam3TrackerEncoderResult
            }
        }

        /**
         * Create the appropriate decoder for the given encoder result.
         * Loads the decoder ONNX model from classpath or the current directory.
         */
        fun newDecoder(encoderResult: EncoderResult): SamDecoder<*> {
            return when (encoderResult) {
                is Sam1EncoderResult -> Sam1Decoder(DecoderModel.SAM1.load())
                is Sam2EncoderResult -> Sam2Decoder(DecoderModel.SAM2.load())
                is Sam3TrackerEncoderResult -> Sam3TrackerDecoder(DecoderModel.SAM3_TRACKER_FP16.load())
            }
        }

        /**
         * Decode with automatic encoder result type checking.
         */
        @Suppress("UNCHECKED_CAST")
        fun decode(
            decoder: SamDecoder<*>,
            encoderResult: EncoderResult,
            prompt: SamPrompt
        ): DecoderResult {

            return when (decoder) {
                is Sam1Decoder -> decoder.decode(encoderResult as Sam1EncoderResult, prompt)
                is Sam2Decoder -> decoder.decode(encoderResult as Sam2EncoderResult, prompt)
                is Sam3TrackerDecoder -> decoder.decode(encoderResult as Sam3TrackerEncoderResult, prompt)
            }
        }
    }
}

class DecoderResult(
    val masks: Array<FloatArray>,
    val ious: FloatArray,
    val maskSize: Int, /* edge length of the presumed square mask result */
) {
    val bestIndex: Int get() = ious.withIndex().maxByOrNull { it.value }?.index ?: 0
    val bestMask: FloatArray get() = masks[bestIndex]
}