package org.janelia.saalfeldlab.samlink.models

import ai.onnxruntime.OnnxTensor
import org.janelia.saalfeldlab.samlink.decode.BoxPrompt
import org.janelia.saalfeldlab.samlink.decode.PointPrompt

/** shared logic across some models  */
private fun createPointTensors(
    coords: DecodeParameter,
    labels: DecodeParameter,
    points: List<PointPrompt>
): Pair<OnnxTensor, OnnxTensor> {
    val coordsData = FloatArray(points.size * 2)
    val labelsData = FloatArray(points.size)
    points.forEachIndexed { i, pt ->
        coordsData[i * 2] = pt.x
        coordsData[i * 2 + 1] = pt.y
        labelsData[i] = pt.label.value
    }
    val numPoints = points.size.toLong()
    val coordsShape = coords.shape.copyOf().also { shape ->
        shape[shape.indexOf(-1)] = numPoints
    }
    val coordsTensor = coords.wrapAsTensor(coordsData, coordsShape)

    val labelsShape = labels.shape.copyOf().also { shape ->
        shape[shape.indexOf(-1)] = numPoints
    }
    val labelsTensor = labels.wrapAsTensor(labelsData, labelsShape)

    return coordsTensor to labelsTensor
}

object Sam1Model {

    object Encoder {

        const val INPUT_EDGE_SIZE = 1024L

        internal enum class Outputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            IMAGE_EMBEDDINGS("image_embeddings", longArrayOf(1, 256, 64, 64))
        }

        internal enum class Inputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            IMAGE("image", longArrayOf(1L, 3L, INPUT_EDGE_SIZE, INPUT_EDGE_SIZE))
        }
    }

    object Decoder {

        const val OUTPUT_EDGE_SIZE = 256

        internal enum class Inputs(override val parameter: String, override val shape: LongArray) : DecodeParameter {
            IMAGE_EMBEDDINGS(Encoder.Outputs.IMAGE_EMBEDDINGS),
            ORIG_IM_SIZE("orig_im_size", longArrayOf(2)),
            MASK_INPUT("mask_input", longArrayOf(1, 1, OUTPUT_EDGE_SIZE.toLong(), OUTPUT_EDGE_SIZE.toLong())),
            HAS_MASK_INPUT("has_mask_input", longArrayOf(1)),
            POINT_COORDS("point_coords", longArrayOf(1, -1, 2)),
            POINT_LABELS("point_labels", longArrayOf(1, -1)),
            ;

            constructor(parameter: ModelParameter) : this(parameter.parameter, parameter.shape)

            companion object {
                fun createPointTensors(points: List<PointPrompt>) =
                    createPointTensors(POINT_COORDS, POINT_LABELS, points)
            }
        }

        internal enum class Outputs(override val parameter: String, override val shape: LongArray) : DecodeParameter {
            LOW_RES_MASKS("low_res_masks", longArrayOf()),
            IOU_PREDICTIONS("iou_predictions", longArrayOf())
            ;

        }

    }

}

object Sam2Model {

    object Encoder {
        const val INPUT_EDGE_SIZE = 1024L

        internal enum class Inputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            IMAGE("image", longArrayOf(1, 3, INPUT_EDGE_SIZE, INPUT_EDGE_SIZE)),
        }

        internal enum class Outputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            IMAGE_EMBED("image_embed", longArrayOf(1, 256, 64, 64)),
            HIGH_RES_FEATS_0("high_res_feats_0", longArrayOf(1, 32, 256, 256)),
            HIGH_RES_FEATS_1("high_res_feats_1", longArrayOf(1, 64, 128, 128));
        }
    }

    object Decoder {
        const val OUTPUT_EDGE_SIZE = 256

        internal enum class Inputs(override val parameter: String, override val shape: LongArray) : DecodeParameter {
            IMAGE_EMBED(Encoder.Outputs.IMAGE_EMBED),
            HIGH_RES_FEATS_0(Encoder.Outputs.HIGH_RES_FEATS_0),
            HIGH_RES_FEATS_1(Encoder.Outputs.HIGH_RES_FEATS_1),
            ORIG_IM_SIZE(Sam1Model.Decoder.Inputs.ORIG_IM_SIZE),
            MASK_INPUT(Sam1Model.Decoder.Inputs.MASK_INPUT),
            HAS_MASK_INPUT(Sam1Model.Decoder.Inputs.HAS_MASK_INPUT),
            POINT_COORDS(Sam1Model.Decoder.Inputs.POINT_COORDS),
            POINT_LABELS(Sam1Model.Decoder.Inputs.POINT_LABELS),
            ;

            constructor(parameter: ModelParameter) : this(parameter.parameter, parameter.shape)

            companion object {
                fun createPointTensors(points: List<PointPrompt>) =
                    createPointTensors(POINT_COORDS, POINT_LABELS, points)
            }
        }

        internal enum class Outputs(override val parameter: String, override val shape: LongArray) : DecodeParameter {
            MASKS("masks", longArrayOf()),
            IOU_PREDICTIONS("iou_predictions", longArrayOf())
        }

    }
}

object Sam3TrackerModel {

    object Encoder {

        const val INPUT_EDGE_SIZE = 1008L

        internal enum class Inputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            PIXEL_VALUES("pixel_values", longArrayOf(1L, 3L, INPUT_EDGE_SIZE, INPUT_EDGE_SIZE)),
        }

        internal enum class Outputs(override val parameter: String, override val shape: LongArray) : EncodeParameter {
            IMAGE_EMBED_0("image_embeddings.0", longArrayOf(1, 32, 288, 288)),
            IMAGE_EMBED_1("image_embeddings.1", longArrayOf(1, 64, 144, 144)),
            IMAGE_EMBED_2("image_embeddings.2", longArrayOf(1, 256, 72, 72)),
        }

    }

    object Decoder {
        const val OUTPUT_EDGE_SIZE = 288;

        internal enum class Inputs(override val parameter: String, override val shape: LongArray) : DecodeParameter {
            IMAGE_EMBED_0(Encoder.Outputs.IMAGE_EMBED_0),
            IMAGE_EMBED_1(Encoder.Outputs.IMAGE_EMBED_1),
            IMAGE_EMBED_2(Encoder.Outputs.IMAGE_EMBED_2),

            INPUT_POINTS("input_points", longArrayOf(1, 1, -1, 2)),
            INPUT_LABELS("input_labels", longArrayOf(1, 1, -1)),

            INPUT_BOXES("input_boxes", longArrayOf(1, -1, 4))
            ;

            constructor(parameter: ModelParameter) : this(parameter.parameter, parameter.shape)

            companion object {
                fun createPointTensors(points: List<PointPrompt>): Pair<OnnxTensor, OnnxTensor> {
                    val coordsData = FloatArray(points.size * 2)
                    val labelsData = LongArray(points.size)
                    points.forEachIndexed { i, pt ->
                        coordsData[i * 2] = pt.x
                        coordsData[i * 2 + 1] = pt.y
                        labelsData[i] = pt.label.value.toLong()
                    }
                    val numPoints = points.size.toLong()
                    val coordsShape = INPUT_POINTS.shape.copyOf().also { shape ->
                        shape[shape.indexOf(-1)] = numPoints
                    }
                    val coordsTensor = INPUT_POINTS.wrapAsTensor(coordsData, coordsShape)
                    val labelsShape = INPUT_LABELS.shape.copyOf().also { shape ->
                        shape[shape.indexOf(-1)] = numPoints
                    }
                    val labelsTensor = INPUT_LABELS.wrapAsTensor(labelsData, labelsShape)
                    return coordsTensor to labelsTensor
                }

                fun createBoxTensors(boxes: List<BoxPrompt>): OnnxTensor {

                    if (boxes.isEmpty()) {
                        return INPUT_BOXES.wrapAsTensor(floatArrayOf(), longArrayOf(1, 0, 4))
                    }

                    val boxData = FloatArray(boxes.size * 4)
                    boxes.forEachIndexed { i, box ->
                        boxData[i * 4] = box.topLeft.x
                        boxData[i * 4 + 1] = box.topLeft.y
                        boxData[i * 4 + 2] = box.bottomRight.x
                        boxData[i * 4 + 3] = box.bottomRight.y
                    }
                    return INPUT_BOXES.wrapAsTensor(boxData, longArrayOf(1, boxes.size.toLong(), 4))
                }
            }
        }

        internal enum class Outputs(override val parameter: String, override val shape: LongArray) : DecodeParameter {
            PRED_MASKS("pred_masks", longArrayOf()),
            IOU_SCORES("iou_scores", longArrayOf())

        }

    }

}


