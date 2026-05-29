package org.janelia.saalfeldlab.samlink

import org.janelia.saalfeldlab.samlink.decode.SamDecoder
import org.janelia.saalfeldlab.samlink.decode.SamPointLabel
import org.janelia.saalfeldlab.samlink.decode.SamPrompt
import org.janelia.saalfeldlab.samlink.encode.EncoderResult
import org.janelia.saalfeldlab.samlink.encode.SamEncoder
import org.janelia.saalfeldlab.samlink.encode.scaleToMaxEdgeSize
import java.awt.Color
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import kotlin.math.exp
import kotlin.test.assertTrue

object TestUtils {

    /**
     * Naive test to check the expected decode result of a square image generated with [rectangleImage]
     *
     */
    fun assertDecodedRectangleImage(mask: FloatArray, width: Int, height: Int, borderPercent: Double = .25) {

        val innerWidth = (width * (2 * borderPercent))
        val innerHeight = (height * (2 * borderPercent))

        val expectedInnerArea = innerHeight * innerWidth

        var actualInnerArea = 0
        mask.forEach { logit ->
            if (logit > 0f) actualInnerArea ++
        }

        val outerArea = (width * height).toDouble()
        val expectedPercent = expectedInnerArea / outerArea
        val actualPercentForeground = actualInnerArea / outerArea

        assertTrue("Percent foreground should be within 10% of $expectedPercent but was $actualPercentForeground") {
            expectedPercent * .9 <= actualPercentForeground && actualPercentForeground <= expectedPercent * 1.1
        }
    }

    /**
     * Copy the top-left [cropWidth]x[cropHeight] region out of
     * [maskEdge]x[maskEdge] mask. Used to drop padding rows/cols before
     * checking foreground percentages.
     */
    fun cropTopLeft(mask: FloatArray, maskEdge: Int, cropWidth: Int, cropHeight: Int): FloatArray {
        val out = FloatArray(cropWidth * cropHeight)
        for (y in 0 until cropHeight) {
            System.arraycopy(mask, y * maskEdge, out, y * cropWidth, cropWidth)
        }
        return out
    }

    /**
     * Render a decoder logits array as an 8-bit grayscale image.
     * This is mainly for debugging.
     *
     * Each logit is mapped through `sigmoid` to and scaled to [0, 255].
     */
    @Suppress("unused")
    fun imageFromFloatArray(mask: FloatArray, width: Int, height: Int): BufferedImage {
        require(mask.size == width * height) { "mask size ${mask.size} does not match $width × $height" }
        val image = BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
        val pixels = (image.raster.dataBuffer as DataBufferByte).data
        for (i in mask.indices) {
            val probability = 1f / (1f + exp(-mask[i]))
            pixels[i] = (probability * 255f).toInt().coerceIn(0, 255).toByte()
        }
        return image
    }

    /**
     * Generate a rectangle image of dimensions [width]x[height] with a black background
     * and an inner red square from (width*.25, height*.25) -> (width*.75, height*.75)
     *
     * @param width of image
     * @param height of image
     * @param borderPercent of image that makes of the border, on each side of the inner square
     * @return the image
     */
    fun rectangleImage(width: Int, height: Int, borderPercent: Double = .25): BufferedImage {
        val left = (width * borderPercent).toInt()
        val top = (height * borderPercent).toInt()
        val squareWidth = (width * .5).toInt()
        val squareHeight = (height * .5).toInt()

        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        image.createGraphics().apply {
            color = Color.BLACK
            fillRect(0, 0, width, height)

            color = Color.RED
            fillRect(left, top, squareWidth, squareHeight)
        }
        return image
    }
}

suspend fun <T : EncoderResult> testSquareWithBorder(
    width: Int, height: Int,
    decodedImageEdge: Int = 256,
    borderPercent: Double = .25,
    encoder: SamEncoder<T, *>,
    decode: ((T, SamPrompt) -> SamDecoder.DecoderResult)?,
) {
    val image = TestUtils.rectangleImage(width, height, borderPercent = borderPercent)
    encoder.encode(image).use { encodeResult ->
        val (scaledWidth, scaledHeight) = scaleToMaxEdgeSize(width, height, encodeResult.inputSize.toInt())
        val prompt = rectangleTestPrompt(scaledWidth, scaledHeight, borderPercent)
        decode?.let {

            val decodeResult = decode(encodeResult, prompt)
            assertTrue(decodeResult.ious.all { it.isFinite() })
            /* Crop so we can expect the correct foreground percentage relative to the input image not the padded, decoded image */
            val scale = decodedImageEdge / encodeResult.inputSize.toDouble()
            val contentWidth = (scale * encodeResult.imageWidth).toInt()
            val contentHeight = (scale * encodeResult.imageHeight).toInt()

            val cropped = TestUtils.cropTopLeft(decodeResult.bestMask, decodedImageEdge, contentWidth, contentHeight)

            TestUtils.imageFromFloatArray(cropped, contentWidth, contentHeight)
            TestUtils.assertDecodedRectangleImage(cropped, contentWidth, contentHeight, borderPercent)
        }
    }
}

/**
 * Box Prompt around the test rectangle, and a center foreground point
 */
fun rectangleTestPrompt(width: Int, height: Int, borderPercent: Double): SamPrompt {
    val x1 = (width * borderPercent).toFloat()
    val y1 = (height * borderPercent).toFloat()
    val x2 = (width * (1.0 - borderPercent)).toFloat()
    val y2 = (height * (1.0 - borderPercent)).toFloat()
    val centerX = x1 + (x2 - x1) / 2f
    val centerY = y1 + (y2 - y1) / 2f
    return SamPrompt()
        .addBox(x1, y1, x2, y2)
        .addPoint(centerX, centerY, SamPointLabel.FOREGROUND)
}