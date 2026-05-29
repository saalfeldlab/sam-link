package org.janelia.saalfeldlab.samlink.decode

import ai.onnxruntime.*
import org.janelia.saalfeldlab.samlink.encode.Sam1EncoderResult
import org.janelia.saalfeldlab.samlink.ORT_ENV
import java.nio.FloatBuffer

/**
 * SAM1 decoder implementation backed by onnx model.
 */
class Sam1Decoder(private val session: OrtSession) : SamDecoder<Sam1EncoderResult> {

    override val supportsMaskRefinement = true

    override fun decode(encoderResult: Sam1EncoderResult, prompt: SamPrompt): SamDecoder.DecoderResult {
        val owned = mutableListOf<OnnxTensor>()
        val inputs = mutableMapOf<String, OnnxTensor>()
        inputs["image_embeddings"] = encoderResult.imageEmbedding
        addPromptInputs(prompt, inputs, owned)
        inputs["orig_im_size"] =
            OnnxTensor.createTensor( ORT_ENV,
            FloatBuffer.wrap(floatArrayOf(1024f, 1024f)), longArrayOf(2)
        ).also { owned += it }
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
                if (allPoints.isEmpty())
                    allPoints.add(PointPrompt(0f, 0f, SamPointLabel.BACKGROUND))
                SamDecoder.addPointInputs(allPoints, inputs, owned)

                if (lastMask != null)
                    SamDecoder.addMaskInputs(SamDecoder.binaryToLogits(lastMask), inputs, owned = owned)
                else
                    SamDecoder.addEmptyMaskInputs(inputs, owned)
            }
        }
    }

    private fun runDecoder(inputs: Map<String, OnnxTensor>, owned: List<OnnxTensor>): SamDecoder.DecoderResult {

        val results = session.run(inputs)

        /* "masks" is just upsampled; we do some operations on the low resolution masks
        * and then upsample ourselves, so we can just ignore the high res mask */
        val masksResult = results.get("low_res_masks").get() as OnnxTensor
        val iouResult = results.get("iou_predictions").get() as OnnxTensor

        val shape = masksResult.info.shape
        val numMasks = shape[1].toInt()
        val numPixels = MASK_SIZE * MASK_SIZE
        val masksArray = masksResult.floatBuffer.array()
        val iousArray = iouResult.floatBuffer.array()

        val masks = Array(numMasks) { i ->
            FloatArray(numPixels).also { System.arraycopy(masksArray, i * numPixels, it, 0, numPixels) }
        }
        val ious = FloatArray(numMasks) { iousArray[it] }

        owned.forEach { it.close() }
        results.close()

        return SamDecoder.DecoderResult(masks, ious, MASK_SIZE)
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val MASK_SIZE = 256
    }
}
