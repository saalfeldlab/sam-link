package org.janelia.saalfeldlab.samlink.decode

import org.janelia.saalfeldlab.samlink.encode.EncoderResult

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

    /**
     * Return a copy of this prompt with all point/box coordinates multiplied by
     * ([xScale], [yScale]). [MaskPrompt]s are passed through unchanged.
     *
     * If the scale factors are both `1f` then this prompt will be returned unchanged.
     */
    open fun scale(xScale: Float, yScale: Float): SamPrompt {
        if (xScale == 1f && yScale == 1f) return this
        val out = SamPrompt()
        for (prompt in prompts) {
            when (prompt) {
                is BoxPrompt -> out.add(prompt.scale(xScale, yScale))
                is PointPrompt -> out.add(PointPrompt(prompt.x * xScale, prompt.y * yScale, prompt.label))
                is MaskPrompt -> out.add(prompt)
                is SamPrompt -> out.add(prompt.scale(xScale, yScale))
            }
        }
        return out
    }

    /**
     * Convenience over [scale] that derives the scale factors from an [EncoderResult].
     *
     * ```kotlin
     *  encoderRes.run {
     *      scale(
     *          xScale = contentWidth.toFloat() / sourceWidth,
     *          yScale = contentHeight.toFloat() / sourceHeight,
     *      )
     * }
     * ```
     * Useful for decoders to map prompt points into the encoded input space
     */
    fun scaleToEncodeInput(encodeRes: EncoderResult): SamPrompt = scale(
        xScale = encodeRes.scaledWidth.toFloat() / encodeRes.sourceWidth,
        yScale = encodeRes.scaledHeight.toFloat() / encodeRes.sourceHeight,
    )


    /**
     * Convenience over [scale] that derives the scale factors from an [EncoderResult]
     * and a [DecoderResult].
     *
     * ```kotlin
     *  val maskPerInput = decodeRes.maskSize.toFloat() / encoderRes.inputSize
     *  encoderRes.run {
     *      scale(
     *          xScale = contentWidth.toFloat() / sourceWidth * maskPerInput,
     *          yScale = contentHeight.toFloat() / sourceHeight * maskPerInput,
     *      )
     *  }
     * ```
     * Useful for mapping prompt points into the decoder's output mask space
     */
    fun scaleToDecodeOutput(encoderRes: EncoderResult, decodeRes: DecoderResult): SamPrompt {
        val maskPerInput = decodeRes.maskSize.toFloat() / encoderRes.inputSize
        return scale(
            xScale = encoderRes.scaledWidth.toFloat() / encoderRes.sourceWidth * maskPerInput,
            yScale = encoderRes.scaledHeight.toFloat() / encoderRes.sourceHeight * maskPerInput,
        )
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

    override fun scale(xScale: Float, yScale: Float): BoxPrompt {
        if (xScale == 1f && yScale == 1f) return this
        return BoxPrompt(
            topLeft.x * xScale, topLeft.y * yScale,
            bottomRight.x * xScale, bottomRight.y * yScale,
        )
    }
}

/** Mask prompt — binary mask for refinement (typically 256x256 low-res) */
class MaskPrompt(val mask: FloatArray) : SamPromptBase