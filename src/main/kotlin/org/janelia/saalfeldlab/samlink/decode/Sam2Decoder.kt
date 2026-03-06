package org.janelia.saalfeldlab.samlink.decode

import ai.onnxruntime.*
import org.janelia.saalfeldlab.samlink.encode.Sam2EncoderResult
import org.janelia.saalfeldlab.samlink.ORT_ENV
import java.nio.IntBuffer

/**
 * SAM2.1 decoder implementation.
 *
 * Supports composable prompts: multiple points, boxes, and masks
 * can be combined in a single decode call.
 *
 * Input shapes:
 * - image_embed: (1, 256, 64, 64)
 * - high_res_feats_0: (1, 32, 256, 256)
 * - high_res_feats_1: (1, 64, 128, 128)
 * - point_coords: (num_labels, num_points, 2) - prompt coordinates in 1024x1024 space
 * - point_labels: (num_labels, num_points) - 1=foreground, 0=background, 2=box TL, 3=box BR, -1=padding
 * - mask_input: (num_labels, 1, 256, 256) - prior mask for refinement
 * - has_mask_input: (num_labels,) - 1.0 if mask_input is valid, 0.0 otherwise
 * - orig_im_size: (2,) INT32 - original image size, always (1024, 1024) for this implementation
 *
 * Output shapes:
 * - masks: (1, 1, 1024, 1024) - single mask at full resolution (downsampled to 256x256 internally)
 * - iou_predictions: (1, 1) - IOU score for the mask
 */
class Sam2Decoder(
    private val session: OrtSession
) : SamDecoder<Sam2EncoderResult> {

    override val supportsMaskRefinement = true

    override fun decode(encoderResult: Sam2EncoderResult, prompt: SamPrompt): SamDecoder.DecoderResult {
        val owned = mutableListOf<OnnxTensor>()
        val inputs = mutableMapOf<String, OnnxTensor>()
        inputs["image_embed"] = encoderResult.imageEmbedding
        inputs["high_res_feats_0"] = encoderResult.highResFeats0
        inputs["high_res_feats_1"] = encoderResult.highResFeats1
        addPromptInputs(prompt, inputs, owned)
        return runDecoder(inputs, owned)
    }

    private fun addPromptInputs(
        prompt: SamPromptBase,
        inputs: MutableMap<String, OnnxTensor>,
        owned: MutableList<OnnxTensor>
    ) {
        when (prompt) {
            is PointPrompt -> SamDecoder.addPointInputs(listOf(prompt), inputs, owned)
            is MaskPrompt -> SamDecoder.addMaskInputs(prompt.mask, inputs, owned = owned)
            is SamPrompt -> {
                val allPoints = mutableListOf<PointPrompt>()
                var lastMask: FloatArray? = null
                for (leaf in prompt.flatten()) {
                    when (leaf) {
                        is BoxPrompt -> allPoints.addAll(leaf.prompts.filterIsInstance<PointPrompt>())
                        is PointPrompt -> allPoints.add(leaf)
                        is MaskPrompt -> lastMask = leaf.mask
                        else -> {}
                    }
                }
                if (allPoints.isEmpty()) allPoints.add(PointPrompt(0f, 0f, SamPointLabel.PADDING))
                SamDecoder.addPointInputs(allPoints, inputs, owned)
                if (lastMask != null) SamDecoder.addMaskInputs(lastMask, inputs, owned = owned)
                else SamDecoder.addEmptyMaskInputs(inputs, owned)
            }
        }
    }

    private fun origImSizeInput(): OnnxTensor {
        val desiredOutputSize = IntBuffer.wrap(intArrayOf(MASK_SIZE, MASK_SIZE))
        return OnnxTensor.createTensor( ORT_ENV, desiredOutputSize, longArrayOf(2) )
    }

    private fun runDecoder(inputs: Map<String, OnnxTensor>, owned: List<OnnxTensor>): SamDecoder.DecoderResult {
        val allInputs = inputs.toMutableMap()

        /* This determines the size of the output mask  */
        val origImSize = origImSizeInput()
        allInputs["orig_im_size"] = origImSize

        val results = session.run(allInputs)

        val masksResult = results.get("masks").get() as OnnxTensor
        val iouResult = results.get("iou_predictions").get() as OnnxTensor

        val shape = masksResult.info.shape
        val numMasks = shape[1].toInt()
        val numPixels = MASK_SIZE * MASK_SIZE
        val masks = masksResult.floatBuffer.array()
        val ious = iouResult.floatBuffer.array()

        val allMasks = Array(numMasks) { maskIdx ->
            FloatArray(numPixels).also { System.arraycopy(masks, maskIdx * numPixels, it, 0, numPixels) }
        }
        val allIous = FloatArray(numMasks) { ious[it] }

        owned.forEach { it.close() }
        origImSize.close()
        results.close()

        return SamDecoder.DecoderResult(
            allMasks,
            allIous,
            MASK_SIZE
        )
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val MASK_SIZE = 256
    }
}
