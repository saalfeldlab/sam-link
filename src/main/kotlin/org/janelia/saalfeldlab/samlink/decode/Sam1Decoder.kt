package org.janelia.saalfeldlab.samlink.decode

import ai.onnxruntime.*
import ai.onnxruntime.OnnxTensor
import org.janelia.saalfeldlab.samlink.encode.Sam1EncoderResult
import org.janelia.saalfeldlab.samlink.models.DecodeParameter.Companion.get
import org.janelia.saalfeldlab.samlink.models.ModelParameter.Companion.set
import org.janelia.saalfeldlab.samlink.models.Sam1Model
import org.janelia.saalfeldlab.samlink.models.Sam1Model.Decoder.Inputs
import org.janelia.saalfeldlab.samlink.models.Sam1Model.Decoder.OUTPUT_EDGE_SIZE
import org.janelia.saalfeldlab.samlink.models.Sam1Model.Decoder.Outputs

/**
 * SAM1 decoder implementation backed by onnx model.
 */
class Sam1Decoder(private val session: OrtSession) : SamDecoder<Sam1EncoderResult> {

    override val supportsMaskRefinement = true

    override fun decode(encoderResult: Sam1EncoderResult, prompt: SamPrompt): DecoderResult {
        val owned = mutableListOf<OnnxTensor>()
        val inputs = mutableMapOf<String, OnnxTensor>()

        inputs[Inputs.IMAGE_EMBEDDINGS.parameter] = encoderResult.imageEmbedding
        inputs[Inputs.ORIG_IM_SIZE.parameter] =
            Inputs.ORIG_IM_SIZE.allocateDirectTensor(floatArrayOf(1024f, 1024f)).also { owned += it }

        val scaledPrompt = prompt.scaleToEncodeInput(encoderResult)

        addPromptInputs(scaledPrompt, inputs, owned)

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
            points += PointPrompt(0f, 0f, SamPointLabel.BACKGROUND)

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

        inputs[Inputs.MASK_INPUT] = Inputs.MASK_INPUT.allocateDirectTensor(prompt.mask).also {
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

    private fun runDecoder(inputs: Map<String, OnnxTensor>, owned: List<OnnxTensor>): DecoderResult {

        val results = session.run(inputs)

        /* "masks" is just upsampled; we do some operations on the low resolution masks
        * and then upsample ourselves, so we can just ignore the high res mask */
        val masksResult = results[Outputs.LOW_RES_MASKS]
        val iouResult = results[Outputs.IOU_PREDICTIONS]

        val shape = masksResult.info.shape
        val numMasks = shape[1].toInt()
        val numPixels = OUTPUT_EDGE_SIZE * OUTPUT_EDGE_SIZE
        val masksArray = masksResult.floatBuffer.array()
        val iousArray = iouResult.floatBuffer.array()

        val masks = Array(numMasks) { i ->
            FloatArray(numPixels).also { System.arraycopy(masksArray, i * numPixels, it, 0, numPixels) }
        }
        val ious = FloatArray(numMasks) { iousArray[it] }

        owned.forEach { it.close() }
        results.close()

        return DecoderResult(masks, ious, OUTPUT_EDGE_SIZE)
    }

    override fun close() {
        session.close()
    }

    companion object {

        val EMPTY_MASK_PROMPT = MaskPrompt(FloatArray(OUTPUT_EDGE_SIZE * OUTPUT_EDGE_SIZE))
    }
}
