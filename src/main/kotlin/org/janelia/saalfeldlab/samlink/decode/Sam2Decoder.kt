package org.janelia.saalfeldlab.samlink.decode

import ai.onnxruntime.*
import org.janelia.saalfeldlab.samlink.encode.Sam2EncoderResult
import org.janelia.saalfeldlab.samlink.models.DecodeParameter.Companion.get
import org.janelia.saalfeldlab.samlink.models.ModelParameter.Companion.set
import org.janelia.saalfeldlab.samlink.models.Sam2Model.Decoder.Inputs
import org.janelia.saalfeldlab.samlink.models.Sam2Model.Decoder.OUTPUT_EDGE_SIZE
import org.janelia.saalfeldlab.samlink.models.Sam2Model
import org.janelia.saalfeldlab.samlink.models.Sam2Model.Decoder.Inputs.IMAGE_EMBED
import org.janelia.saalfeldlab.samlink.models.Sam2Model.Decoder.Inputs.MASK_INPUT
import org.janelia.saalfeldlab.samlink.models.Sam2Model.Decoder.Inputs.ORIG_IM_SIZE

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

    override fun decode(encoderResult: Sam2EncoderResult, prompt: SamPrompt): DecoderResult {
        val owned = mutableListOf<OnnxTensor>()
        val inputs = mutableMapOf<String, OnnxTensor>()
        inputs[IMAGE_EMBED] = encoderResult.imageEmbedding
        inputs[Sam2Model.Decoder.Inputs.HIGH_RES_FEATS_0] = encoderResult.highResFeats0
        inputs[Sam2Model.Decoder.Inputs.HIGH_RES_FEATS_1] = encoderResult.highResFeats1

        val promptInEncodeInput = prompt.scaleToEncodeInput(encoderResult)
        addPromptInputs(promptInEncodeInput, inputs, owned)
        return runDecoder(inputs, owned)
    }

    private fun addPromptInputs(
        prompt: SamPrompt,
        inputs: MutableMap<String, OnnxTensor>,
        owned: MutableList<OnnxTensor>
    ) {
        val points = mutableListOf<PointPrompt>()
        var mask: MaskPrompt? = null
        for (leaf in prompt.flatten()) {
            when (leaf) {
                is PointPrompt -> points += leaf
                is BoxPrompt -> points += leaf.prompts.filterIsInstance<PointPrompt>()
                is MaskPrompt -> mask = leaf
                else -> {}
            }
        }
        if (points.isEmpty())
            points += PointPrompt(0f, 0f, SamPointLabel.PADDING)

        addPoints(points, inputs, owned)

        val (maskPrompt, hasMask) = mask?.let { it to true } ?: (EMPTY_MASK_PROMPT to false)
        addMask(maskPrompt, inputs, owned, hasMask)
    }

    private fun addMask(
        prompt: MaskPrompt,
        inputs: MutableMap<String, OnnxTensor>,
        owned: MutableList<OnnxTensor>,
        hasMask: Boolean = true
    ) {

        inputs[MASK_INPUT] = MASK_INPUT.allocateDirectTensor(prompt.mask).also {
            owned += it
        }

        val hasMaskData = floatArrayOf(if (hasMask) 1f else 0f)
        inputs[Inputs.HAS_MASK_INPUT] = Inputs.HAS_MASK_INPUT.wrapAsTensor(hasMaskData).also {
            owned += it
        }
    }

    private fun addPoints(
        points: List<PointPrompt>,
        inputs: MutableMap<String, OnnxTensor>,
        owned: MutableList<OnnxTensor>
    ) {

        val (coords, labels) = Inputs.createPointTensors(points)
        inputs[Inputs.POINT_COORDS] = coords.also { owned += it }
        inputs[Inputs.POINT_LABELS] = labels.also { owned += it }
    }

    private fun runDecoder(inputs: Map<String, OnnxTensor>, owned: MutableList<OnnxTensor>): DecoderResult {
        val allInputs = inputs.toMutableMap()

        /* This determines the size of the output mask. We don 't want the resizing, so we just ask
        * for the output edge size   */
        val sizeArray = intArrayOf(OUTPUT_EDGE_SIZE, OUTPUT_EDGE_SIZE)
        allInputs[ORIG_IM_SIZE] = ORIG_IM_SIZE.wrapAsTensor(sizeArray).also {
            owned += it
        }

        val results = session.run(allInputs)

        val masksResult = results[Sam2Model.Decoder.Outputs.MASKS]
        val iouResult = results[Sam2Model.Decoder.Outputs.IOU_PREDICTIONS]

        val shape = masksResult.info.shape
        val numMasks = shape[1].toInt()
        val numPixels = OUTPUT_EDGE_SIZE * OUTPUT_EDGE_SIZE
        val masks = masksResult.floatBuffer.array()
        val ious = iouResult.floatBuffer.array()

        val allMasks = Array(numMasks) { maskIdx ->
            FloatArray(numPixels).also { System.arraycopy(masks, maskIdx * numPixels, it, 0, numPixels) }
        }
        val allIous = FloatArray(numMasks) { ious[it] }

        owned.forEach { it.close() }
        results.close()

        return DecoderResult(
            allMasks,
            allIous,
            OUTPUT_EDGE_SIZE
        )
    }

    override fun close() {
        session.close()
    }

    companion object {

        val EMPTY_MASK_PROMPT = MaskPrompt(FloatArray(OUTPUT_EDGE_SIZE * OUTPUT_EDGE_SIZE))
    }
}
