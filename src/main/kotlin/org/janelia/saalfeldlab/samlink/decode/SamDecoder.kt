package org.janelia.saalfeldlab.samlink.decode

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import org.janelia.saalfeldlab.samlink.ORT_ENV
import org.janelia.saalfeldlab.samlink.encode.EncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam1EncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam2EncoderResult
import org.janelia.saalfeldlab.samlink.encode.Sam3TrackerEncoderResult
import java.nio.FloatBuffer
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.math.ln

enum class SamPointLabel(val value: Float) {
    PADDING(-1f),
    BACKGROUND(0f),
    FOREGROUND(1f),
    BOX_TOP_LEFT(2f),
    BOX_BOTTOM_RIGHT(3f)
}

/** Single point prompt with coordinates and label */
data class PointPrompt(val x: Float, val y: Float, val label: SamPointLabel) : SamPromptBase

/** Marker interface for SAM prompt types */
sealed interface SamPromptBase

/**
 * Composable prompt container. Accumulates [SamPromptBase]s
 * for [SamDecoder.decode]. Decoders iterate [prompts] and
 * handle each type directly.
 */
open class SamPrompt : SamPromptBase {
    val prompts = mutableListOf<SamPromptBase>()

    fun add(prompt: SamPromptBase) = apply {
        when (prompt) {
            is MaskPrompt -> addMaskPrompt(prompt)
            else -> prompts.add(prompt)
        }
    }

    /**
     * Add a point prompt at (x,y) with the given label type.
     *
     * @param x coordinate
     * @param y coordinate
     * @param label point type
     */
    fun addPoint(x: Float, y: Float, label: SamPointLabel) = apply {
        add(PointPrompt(x, y, label))
    }

    /**
     * Add a box prompt with the given coordinates. Sorted so the top-left corner is
     * always min(x1,x2), min(y1,y2) and the bottom-right corner is max(x1,x2), max(y1,y2).
     *
     * Some decoders support multiple box prompts, but not all. The behavior of multiple box prompts
     * is defined by the decoder.
     *
     * @param x1 first x coordinate
     * @param y1 first y coordinate
     * @param x2 second x coordinate
     * @param y2 second y coordinate
     */
    fun addBox(x1: Float, y1: Float, x2: Float, y2: Float) = apply { add(BoxPrompt(x1, y1, x2, y2)) }

    /**
     * Add mask prompt. Only up to 1 MaskPrompt is allowed. Existing MaskPrompt will be replaced if present.
     * [mask] is expected to be an array of logits with num elements equal to the decoder's mask size.
     *
     * @param mask
     */
    fun addMask(mask: FloatArray) = apply { add(MaskPrompt(mask)) }

    private fun addMaskPrompt(prompt: MaskPrompt) {
        prompts.removeAll { it is MaskPrompt }
        prompts.add(prompt)
    }

    /** Recursively collect all leaf prompts via flatMap. Semantic
     *  composites like [BoxPrompt] override to return themselves. */
    open fun flatten(): List<SamPromptBase> = prompts.flatMap {
        when (it) {
            is SamPrompt -> it.flatten()
            else -> listOf(it)
        }
    }

    open fun copy(): SamPrompt {
        val copy = SamPrompt()
        for (prompt in prompts) {
            when (prompt) {
                is MaskPrompt -> copy.addMask(prompt.mask.copyOf())
                is PointPrompt -> copy.add(prompt.copy())
                is SamPrompt -> copy.add(prompt.copy())
            }
        }
        return copy
    }
}

/** Box prompt — a composite of two point prompts (top-left, bottom-right) */
@ConsistentCopyVisibility
data class BoxPrompt private constructor(val topLeft: PointPrompt, val bottomRight: PointPrompt) : SamPrompt() {

    constructor(x1: Float, y1: Float, x2: Float, y2: Float) : this(
        PointPrompt(minOf(x1, x2), minOf(y1, y2), SamPointLabel.BOX_TOP_LEFT),
        PointPrompt(maxOf(x1, x2), maxOf(y1, y2), SamPointLabel.BOX_BOTTOM_RIGHT)
    )

    init {
        prompts.add(topLeft)
        prompts.add(bottomRight)
    }

    override fun copy(): SamPrompt {
        return BoxPrompt(topLeft.copy(), bottomRight.copy())
    }

    /* semantic composite — flatten preserves the box as a unit */
    override fun flatten(): List<SamPromptBase> = listOf(this)
}

/** Mask prompt — binary mask for refinement (typically 256x256 low-res) */
class MaskPrompt(val mask: FloatArray) : SamPromptBase

/**
 * Base interface for all SAM decoder implementations.
 *
 * Each decoder consumes a [SamPrompt] and builds its own ONNX inputs.
 *
 * Use [SamDecoder.newDecoder] to instantiate the appropriate decoder
 * based on the ONNX model's input signature.
 *
 * Model families:
 * - SAM1: Supports point/mask/box prompts, outputs 4 candidate masks at 256x256
 * - SAM2.1: Supports point/mask/box prompts, outputs single mask at 256x256
 * - SAM3: Supports point/box prompts, outputs single mask at 288x288
 */
enum class DecoderModel(val resourceName: String) {
    SAM1("sam_vit_h_4b8939.onnx"),
    SAM2_LARGE("sam2.1_hiera_large_decoder.onnx"),
    SAM3_TRACKER_FP16("sam3_tracker_fp16_decoder.onnx");

    fun load(): OrtSession {
        /* Load from packaged jar, or from file*/
        val bytes = SamDecoder::class.java.getResourceAsStream(resourceName)?.readAllBytes()
            ?: Files.readAllBytes(Paths.get(resourceName))
        return ORT_ENV.createSession(bytes)
    }
}

sealed interface SamDecoder<E : EncoderResult> : AutoCloseable {

    class DecoderResult(
        val masks: Array<FloatArray>,
        val ious: FloatArray,
        val maskSize: Int, /* edge length of the presumed square mask result */
    ) {
        val bestIndex: Int get() = ious.withIndex().maxByOrNull { it.value }?.index ?: 0
        val bestMask: FloatArray get() = masks[bestIndex]
    }

    fun decode(encoderResult: E, prompt: SamPrompt): DecoderResult

    /** Whether this decoder supports iterative mask refinement via mask_input */
    val supportsMaskRefinement: Boolean get() = false

    companion object {

        /** Write sparse points as `point_coords` (1, N, 2) + `point_labels` (1, N) float tensors */
        fun addPointInputs(
            points: List<PointPrompt>,
            inputs: MutableMap<String, OnnxTensor>,
            owned: MutableList<OnnxTensor>
        ) {
            val coordsData = FloatArray(points.size * 2)
            val labelsData = FloatArray(points.size)
            points.forEachIndexed { i, pt ->
                coordsData[i * 2] = pt.x
                coordsData[i * 2 + 1] = pt.y
                labelsData[i] = pt.label.value
            }
            inputs["point_coords"] = OnnxTensor.createTensor(
                ORT_ENV,
                FloatBuffer.wrap(coordsData), longArrayOf(1, points.size.toLong(), 2)
            ).also { owned += it }
            inputs["point_labels"] = OnnxTensor.createTensor(
                ORT_ENV,
                FloatBuffer.wrap(labelsData), longArrayOf(1, points.size.toLong())
            ).also { owned += it }
        }

        /**
         * Add the "has no mask" input if a mask is expected.
         *
         * @param inputs to add the "empty" mask input to
         * @param owned OnnxTensors requires cleanup
         */
        fun addEmptyMaskInputs(inputs: MutableMap<String, OnnxTensor>, owned: MutableList<OnnxTensor>) {
            addMaskInputs(FloatArray(256 * 256), inputs, hasMask = false, owned = owned)
        }

        fun addMaskInputs(
            maskData: FloatArray,
            inputs: MutableMap<String, OnnxTensor>,
            hasMask: Boolean = true,
            owned: MutableList<OnnxTensor>
        ) {
            val maskData = FloatBuffer.wrap(maskData)
            val maskShape = longArrayOf(1, 1, 256, 256)
            inputs["mask_input"] = OnnxTensor.createTensor(ORT_ENV, maskData, maskShape).also { owned += it }

            val hasMaskData = FloatBuffer.wrap(floatArrayOf(if (hasMask) 1f else 0f))
            val hasMaskShape = longArrayOf(1)
            inputs["has_mask_input"] = OnnxTensor.createTensor(ORT_ENV, hasMaskData, hasMaskShape).also { owned += it }
        }

        /** Convert binary mask to logits  */
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
                is Sam2EncoderResult -> Sam2Decoder(DecoderModel.SAM2_LARGE.load())
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
