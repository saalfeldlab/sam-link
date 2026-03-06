package org.janelia.saalfeldlab.samlink.decode

import ai.onnxruntime.*
import org.janelia.saalfeldlab.samlink.encode.Sam3TrackerEncoderResult
import org.janelia.saalfeldlab.samlink.ORT_ENV
import java.nio.FloatBuffer
import java.nio.LongBuffer

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

    override fun decode(encoderResult: Sam3TrackerEncoderResult, prompt: SamPrompt): SamDecoder.DecoderResult {
        val owned = mutableListOf<OnnxTensor>()
        val inputs = mutableMapOf<String, OnnxTensor>()
        inputs["image_embeddings.0"] = encoderResult.imageEmbeddings0
        inputs["image_embeddings.1"] = encoderResult.imageEmbeddings1
        inputs["image_embeddings.2"] = encoderResult.imageEmbeddings2
        addPromptInputs(prompt, inputs, owned)
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
        val allBoxes = flat.filterIsInstance<BoxPrompt>()
        addPointInputs(allPoints, inputs, owned)
        addBoxInputs(allBoxes, inputs, owned)
    }

    private fun addPointInputs(
        points: List<PointPrompt>,
        inputs: MutableMap<String, OnnxTensor>,
        owned: MutableList<OnnxTensor>
    ) {
        val coordsData = FloatArray(points.size * 2)
        val labelsData = LongArray(points.size)
        points.forEachIndexed { i, pt ->
            coordsData[i * 2] = pt.x
            coordsData[i * 2 + 1] = pt.y
            labelsData[i] = pt.label.value.toLong()
        }
        inputs["input_points"] = OnnxTensor.createTensor(ORT_ENV,
            FloatBuffer.wrap(coordsData), longArrayOf(1, 1, points.size.toLong(), 2)
        ).also { owned += it }
        inputs["input_labels"] = OnnxTensor.createTensor(ORT_ENV,
            LongBuffer.wrap(labelsData), longArrayOf(1, 1, points.size.toLong())
        ).also { owned += it }
    }

    private fun addBoxInputs(
        boxes: List<BoxPrompt>,
        inputs: MutableMap<String, OnnxTensor>,
        owned: MutableList<OnnxTensor>
    ) {
        if (boxes.isNotEmpty()) {
            val boxData = FloatArray(boxes.size * 4)
            boxes.forEachIndexed { i, box ->
                boxData[i * 4] = box.topLeft.x
                boxData[i * 4 + 1] = box.topLeft.y
                boxData[i * 4 + 2] = box.bottomRight.x
                boxData[i * 4 + 3] = box.bottomRight.y
            }
            inputs["input_boxes"] = OnnxTensor.createTensor(ORT_ENV,
                FloatBuffer.wrap(boxData), longArrayOf(1, boxes.size.toLong(), 4)
            ).also { owned += it }
        } else {
            inputs["input_boxes"] = OnnxTensor.createTensor(ORT_ENV,
                FloatBuffer.wrap(FloatArray(0)), longArrayOf(1, 0, 4)
            ).also { owned += it }
        }
    }

    private fun runDecoder(inputs: Map<String, OnnxTensor>, owned: List<OnnxTensor>): SamDecoder.DecoderResult {
        val results = session.run(inputs)

        val masksResult = results.get("pred_masks").get() as OnnxTensor
        val iouResult = results.get("iou_scores").get() as OnnxTensor

        val masksShape = masksResult.info.shape
        val masks = masksResult.floatBuffer.array()
        val ious = iouResult.floatBuffer.array()

        val numMasks = masksShape[2].toInt()  /* 3 candidates */
        val maskH = masksShape[3].toInt()
        val maskW = masksShape[4].toInt()
        val maskPixels = maskH * maskW

        /* extract and downsample all candidates to MASK_SIZE x MASK_SIZE */
        val allMasks = Array(numMasks) { maskIdx ->
            val fullMask = FloatArray(maskPixels)
            System.arraycopy(masks, maskIdx * maskPixels, fullMask, 0, maskPixels)
            FloatArray(MASK_SIZE * MASK_SIZE) { i ->
                val y = i / MASK_SIZE
                val x = i % MASK_SIZE
                val srcX = (x * maskW / MASK_SIZE).coerceIn(0, maskW - 1)
                val srcY = (y * maskH / MASK_SIZE).coerceIn(0, maskH - 1)
                fullMask[srcY * maskW + srcX]
            }
        }
        val allIous = FloatArray(numMasks) { ious[it] }

        owned.forEach { it.close() }
        results.close()

        return SamDecoder.DecoderResult(allMasks, allIous, MASK_SIZE)
    }

    override fun close() {
        session.close()
    }

    companion object {
        /* use 288 to match SAM3's native feature resolution */
        const val MASK_SIZE = 288
    }
}
