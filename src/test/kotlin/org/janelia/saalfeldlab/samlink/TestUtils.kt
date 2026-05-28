package org.janelia.saalfeldlab.samlink

import java.awt.Color
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.test.assertTrue

object TestUtils {

    /**
     * Naive test to check the expected decode result of a square image generated with [squareImage]
     *
     */
    fun assertDecodedSquareImage(mask: FloatArray, width: Int, height: Int, borderPercent: Double = .25) {

        val innerWidth = (width * (2 * borderPercent))
        val innerHeight = (height * (2 * borderPercent))

        val expectedInnerArea = innerHeight * innerWidth

        var actualInnerArea = 0
        mask.forEach { logit ->
            actualInnerArea += min(1, max(0, logit.toInt()))
        }

        val outerArea = (width * height).toDouble()
        val expectedPercent = expectedInnerArea / outerArea
        val actualPercentForeground = actualInnerArea / outerArea

        assertTrue("Percent foreground should be within 10% of $expectedPercent but was $actualPercentForeground") {
            expectedPercent * .9 <= actualPercentForeground && actualPercentForeground <= expectedPercent * 1.1
        }
    }

    /**
     * Render a decoder logits array as an 8-bit grayscale image.
     * This is mainly for debugging.
     *
     * Each logit is mapped through `sigmoid` to and scaled to [0, 255].
     */
    fun imageFromDecodeResult(mask: FloatArray, width: Int, height: Int): BufferedImage {
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
     * Generate a square image of dimensions [width]x[height] with a black background
     * and an inner red square from (width*.25, height*.25) -> (width*.75, height*.75)
     *
     * @param width of image
     * @param height of image
     * @param borderPercent of image that makes of the border, on each side of the inner square
     * @return the image
     */
    fun squareImage(width: Int, height: Int, borderPercent: Double = .25): BufferedImage {
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