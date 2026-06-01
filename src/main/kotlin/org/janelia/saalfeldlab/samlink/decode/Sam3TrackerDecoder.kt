package org.janelia.saalfeldlab.samlink.decode

import ai.onnxruntime.*
import org.janelia.saalfeldlab.samlink.encode.Sam3TrackerEncoderResult
import org.janelia.saalfeldlab.samlink.models.DecodeParameter.Companion.get
import org.janelia.saalfeldlab.samlink.models.ModelParameter.Companion.set
import org.janelia.saalfeldlab.samlink.models.Sam3TrackerModel.Decoder
import org.janelia.saalfeldlab.samlink.models.Sam3TrackerModel.Decoder.Inputs

/**
 * SAM3 Tracker decoder implementation.
 *
 * Supports direct point and box prompts without a separate geometry encoder.
 * Points and boxes are in image space (0-1008), xyxy format for boxes.
 *
 * Input shapes:
 * - input_points: (1, 1, num_points, 2) - point coordinates in image space
 * - input_labels: (1, 1, num_points) - point labels (1=foreground, 0=background, -1=padding)
 * - input_boxes: (1, num_boxes, 4) - boxes as xyxy in image space
 * - image_embeddings.0: (1, 32, 288, 288)
 * - image_embeddings.1: (1, 64, 144, 144)
 * - image_embeddings.2: (1, 256, 72, 72)
 *
 * Output shapes:
 * - pred_masks: (1, num_objects, 3, H, W) - 3 candidate masks per object
 * - iou_scores: (1, num_objects, 3) - IOU score per candidate mask
 * - object_score_logits: (1, num_objects, 1) - object presence score
 */
class Sam3TrackerDecoder(
    private val session: OrtSession
) : SamDecoder<Sam3TrackerEncoderResult> {

    override fun decode(encoderResult: Sam3TrackerEncoderResult, prompt: SamPrompt): DecoderResult {
        val owned = mutableListOf<OnnxTensor>()
        val inputs = mutableMapOf<String, OnnxTensor>()
        inputs[Inputs.IMAGE_EMBED_0] = encoderResult.imageEmbeddings0
        inputs[Inputs.IMAGE_EMBED_1] = encoderResult.imageEmbeddings1
        inputs[Inputs.IMAGE_EMBED_2] = encoderResult.imageEmbeddings2

        val encodeInputPrompt = prompt.scaleToEncodeInput(encoderResult)

        addPromptInputs(encodeInputPrompt, inputs, owned)
        return runDecoder(inputs, owned)
    }

    private fun addPromptInputs(
        prompt: SamPromptBase,
        inputs: MutableMap<String, OnnxTensor>,
        owned: MutableList<OnnxTensor>
    ) {
        val flat = if (prompt is SamPrompt) prompt.flatten() else listOf(prompt)

        val allPoints = flat.filterIsInstance<PointPrompt>()
            .filter { it.label == SamPointLabel.FOREGROUND || it.label == SamPointLabel.BACKGROUND }
            .ifEmpty { listOf(PointPrompt(0f, 0f, SamPointLabel.PADDING)) }
        addPoints(allPoints, inputs, owned)

        val allBoxes = flat.filterIsInstance<BoxPrompt>()
        addBoxes(allBoxes, inputs, owned)
    }

    private fun addPoints(
        points: List<PointPrompt>,
        inputs: MutableMap<String, OnnxTensor>,
        owned: MutableList<OnnxTensor>
    ) {

        val (coords, labels) = Inputs.createPointTensors(points)
        inputs[Inputs.INPUT_POINTS] = coords.also { owned += it }
        inputs[Inputs.INPUT_LABELS] = labels.also { owned += it }
    }

    private fun addBoxes(
        boxes: List<BoxPrompt>,
        inputs: MutableMap<String, OnnxTensor>,
        owned: MutableList<OnnxTensor>
    ) {
        inputs[Inputs.INPUT_BOXES] = Inputs.createBoxTensors(boxes).also {
            owned += it
        }
    }

    private fun runDecoder(inputs: Map<String, OnnxTensor>, owned: List<OnnxTensor>): DecoderResult {
        val results = session.run(inputs)

        val masksResult = results[Decoder.Outputs.PRED_MASKS]
        val iouResult = results[Decoder.Outputs.IOU_SCORES]

        val masksShape = masksResult.info.shape
        val masks = masksResult.floatBuffer.array()
        val ious = iouResult.floatBuffer.array()

        val numMasks = masksShape[2].toInt()  /* 3 candidates */
        val maskH = masksShape[3].toInt()
        val maskW = masksShape[4].toInt()
        val maskPixels = maskH * maskW

        val allMasks = Array(numMasks) { maskIdx ->
            FloatArray(maskPixels).also { System.arraycopy(masks, maskIdx * maskPixels, it, 0, maskPixels) }
        }
        val allIous = FloatArray(numMasks) { ious[it] }

        owned.forEach { it.close() }
        results.close()

        return DecoderResult(allMasks, allIous, Decoder.OUTPUT_EDGE_SIZE)
    }

    override fun close() {
        session.close()
    }

}
