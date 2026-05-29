package org.janelia.saalfeldlab.samlink

import org.janelia.saalfeldlab.samlink.decode.BoxPrompt
import org.janelia.saalfeldlab.samlink.decode.MaskPrompt
import org.janelia.saalfeldlab.samlink.decode.PointPrompt
import org.janelia.saalfeldlab.samlink.decode.SamPointLabel
import org.janelia.saalfeldlab.samlink.decode.SamPrompt
import org.janelia.saalfeldlab.samlink.encode.scaleToMaxEdgeSize
import org.janelia.saalfeldlab.samlink.encode.scaleWithPadding
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.image.BufferedImage
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
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


    @Nested
    inner class SamPromptComposition {

        @Nested inner class BoxPromptTest {

            @Test
            fun `sort corners regardless order`() {
                val box = BoxPrompt(x1 = 200f, y1 = 150f, x2 = 50f, y2 = 80f)
                assertEquals(50f, box.topLeft.x)
                assertEquals(80f, box.topLeft.y)
                assertEquals(200f, box.bottomRight.x)
                assertEquals(150f, box.bottomRight.y)
                assertEquals(SamPointLabel.BOX_TOP_LEFT, box.topLeft.label)
                assertEquals(SamPointLabel.BOX_BOTTOM_RIGHT, box.bottomRight.label)
            }

            @Test
            fun `flatten returns box as a single prompt`() {
                val outer = SamPrompt().add(SamPrompt().add(BoxPrompt(0f, 0f, 10f, 10f)))
                val flat = outer.flatten()
                assertEquals(1, flat.size)
                assertTrue(flat[0] is BoxPrompt)
            }
        }




        @Test
        fun `addMask twice keeps only the latest MaskPrompt`() {
            val prompt = SamPrompt()
                .addMask(FloatArray(256 * 256) { 0.25f })
                .addMask(FloatArray(256 * 256) { 0.75f })
            val masks = prompt.prompts.filterIsInstance<MaskPrompt>()
            assertEquals(1, masks.size)
            assertEquals(0.75f, masks[0].mask[0])
        }

        @Test
        fun `SamPrompt copy is independent of the original mask buffer`() {
            val original = SamPrompt().addMask(FloatArray(256 * 256) { 0.5f })
            val copy = original.copy()
            val originalMask = original.prompts.filterIsInstance<MaskPrompt>().single().mask
            originalMask[0] = 1.234f
            val copiedMask = copy.prompts.filterIsInstance<MaskPrompt>().single().mask
            assertNotEquals(copiedMask[0], 1.234f, "copy should not share mask storage")
        }

        @Test
        fun `flatten of a point-only prompt yields the point`() {
            val prompt = SamPrompt().addPoint(100f, 200f, SamPointLabel.FOREGROUND)
            val flat = prompt.flatten()
            assertEquals(1, flat.size)
            val point = flat[0] as PointPrompt
            assertEquals(100f, point.x)
            assertEquals(200f, point.y)
            assertEquals(SamPointLabel.FOREGROUND, point.label)
        }
    }

}
