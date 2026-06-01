package org.janelia.saalfeldlab.samlink

import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleToMaxEdgeSize
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.scaleWithPadding
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.image.BufferedImage
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

}
