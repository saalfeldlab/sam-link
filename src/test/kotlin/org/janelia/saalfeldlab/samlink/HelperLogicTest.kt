package org.janelia.saalfeldlab.samlink

import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.intRGBtoCHW
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleToMaxEdgeSize
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleWithPadding
import org.janelia.saalfeldlab.samlink.encode.Normalization
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.image.BufferedImage
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for helper logic.
 *
 */

class HelperLogicTest {

    @Nested
    inner class ScaleToMaxEdge {

        @Test
        fun `scales when smaller than target`() {
            val (width, height) = scaleToMaxEdgeSize(640, 480, 1024)
            assertEquals(1024, width)
            assertEquals((480 * (1024/640.0)).toInt(), height)
        }

        @Test
        fun `scales the longest edge to target and preserves aspect`() {
            val (width, height) = scaleToMaxEdgeSize(1920, 1080, 1024)
            assertEquals(1024, width)
            /* aspect ratio 16:9 → expected height = 1024 * 1080 / 1920 = 576 */
            assertTrue(abs(height - 576) <= 1, "height=$height, expected ~576")
        }

        @Test
        fun `height more than width`() {
            val (width, height) = scaleToMaxEdgeSize(1080, 1920, 1024)
            assertEquals(1024, height)
            assertTrue(abs(width - 576) <= 1, "width=$width, expected ~576")
        }
    }


    @Nested
    inner class ScaleWithPadding {

        @Test
        fun `unchanged when dimensions match`() {
            val width = 1024
            val height = 1024
            val edgeSize = 1024
            val source = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
            val (scaledWidth, scaledHeight) = scaleToMaxEdgeSize(width, height, edgeSize)
            val out = scaleWithPadding(source, scaledWidth, scaledHeight, edgeSize, edgeSize)
            assertTrue(out === source, "expected identity")
        }

        @Test
        fun `smaller images at the top-left`() {
            val width = 100
            val height = 50
            val source = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB).apply {
                createGraphics().apply { color = Color.WHITE; fillRect(0, 0, width, height) }.dispose()
            }
            val edgeSize = 1024
            val (scaledWidth, scaledHeight) = scaleToMaxEdgeSize(width, height, edgeSize)
            val out = scaleWithPadding(source, scaledWidth, scaledHeight, edgeSize, edgeSize)
            assertEquals(edgeSize, out.width)
            assertEquals(edgeSize, out.height)

            val lastIdx = 1023

            /* top-left pixel inside the content region is white */
            /* top-right pixel inside the content region is white since the width is the longest edge */
            assertEquals(Color.WHITE.rgb and 0xFFFFFF, out.getRGB(0, 0) and 0xFFFFFF)
            assertEquals(Color.WHITE.rgb and 0xFFFFFF, out.getRGB(lastIdx, 0) and 0xFFFFFF)


            /* bottom-right is part of the black padding */
            /* bottom-left is part of the black padding */
            assertEquals(0, out.getRGB(lastIdx, lastIdx) and 0xFFFFFF)
            assertEquals(0, out.getRGB(0, lastIdx) and 0xFFFFFF)
        }

        @Test
        fun `a same-size image that is not int RGB is still redrawn`() {
            val edgeSize = 64
            val source = BufferedImage(edgeSize, edgeSize, BufferedImage.TYPE_3BYTE_BGR).apply {
                createGraphics().apply { color = Color.RED; fillRect(0, 0, edgeSize, edgeSize) }.dispose()
            }
            val out = scaleWithPadding(source, edgeSize, edgeSize, edgeSize, edgeSize)
            assertTrue(out !== source, "expected a redraw, not the original")
            assertEquals(BufferedImage.TYPE_INT_RGB, out.type)
            assertEquals(Color.RED.rgb and 0xFFFFFF, out.getRGB(0, 0) and 0xFFFFFF)
        }

        @Test
        fun `downscale large image to fit the longest edge`() {
            val width = 2048
            val height = 1024
            val edgeSize = 1024
            val source = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
            val (scaledWidth, scaledHeight) = scaleToMaxEdgeSize(width, height, edgeSize)
            val out = scaleWithPadding(source, scaledWidth, scaledHeight, edgeSize, edgeSize)
            assertEquals(edgeSize, out.width)
            assertEquals(edgeSize, out.height)
        }
    }

    @Nested
    inner class Normalize {

        /** read back the (red, green, blue) of the single pixel of a 1x1 CHW tensor */
        private fun normalizedPixel(color: Color, normalization: Normalization): Triple<Float, Float, Float> {
            val image = BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB).apply { setRGB(0, 0, color.rgb) }
            val floats = image.intRGBtoCHW(normalization)
                .asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return Triple(floats.get(0), floats.get(1), floats.get(2))
        }

        @Test
        fun `imagenet matches SAM's canonical mean and std`() {
            val (red, green, blue) = normalizedPixel(Color.WHITE, Normalization.IMAGENET)
            /* (255 - pixelMean) / pixelStd, with SAM's [123.675, 116.28, 103.53] / [58.395, 57.12, 57.375] */
            assertEquals((255f - 123.675f) / 58.395f, red, 1e-4f)
            assertEquals((255f - 116.28f) / 57.12f, green, 1e-4f)
            assertEquals((255f - 103.53f) / 57.375f, blue, 1e-4f)
        }

        @Test
        fun `symmetric maps the range onto -1 to 1`() {
            val (red, green, blue) = normalizedPixel(Color.WHITE, Normalization.SYMMETRIC)
            assertEquals(1f, red, 1e-6f)
            assertEquals(1f, green, 1e-6f)
            assertEquals(1f, blue, 1e-6f)

            val (darkRed, darkGreen, darkBlue) = normalizedPixel(Color.BLACK, Normalization.SYMMETRIC)
            assertEquals(-1f, darkRed, 1e-6f)
            assertEquals(-1f, darkGreen, 1e-6f)
            assertEquals(-1f, darkBlue, 1e-6f)
        }

        @Test
        fun `channels are laid out in CHW order`() {
            val image = BufferedImage(2, 1, BufferedImage.TYPE_INT_RGB).apply {
                setRGB(0, 0, Color.RED.rgb)
                setRGB(1, 0, Color.BLUE.rgb)
            }
            val floats = image.intRGBtoCHW(Normalization.SYMMETRIC)
                .asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()

            /* both red values first, then both green, then both blue */
            assertEquals(1f, floats.get(0), 1e-6f)
            assertEquals(-1f, floats.get(1), 1e-6f)
            assertEquals(-1f, floats.get(2), 1e-6f)
            assertEquals(-1f, floats.get(3), 1e-6f)
            assertEquals(-1f, floats.get(4), 1e-6f)
            assertEquals(1f, floats.get(5), 1e-6f)
        }
    }

}
