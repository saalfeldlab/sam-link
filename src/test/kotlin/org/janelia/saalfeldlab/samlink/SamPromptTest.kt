package org.janelia.saalfeldlab.samlink

import org.janelia.saalfeldlab.samlink.decode.BoxPrompt
import org.janelia.saalfeldlab.samlink.decode.MaskPrompt
import org.janelia.saalfeldlab.samlink.decode.PointPrompt
import org.janelia.saalfeldlab.samlink.decode.SamPointLabel
import org.janelia.saalfeldlab.samlink.decode.SamPrompt
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class SamPromptTest {

    @Nested
    inner class PromptPointScaling {

        @Test
        fun `scales point coords independently per axis and preserves the label`() {
            val prompt = SamPrompt().addPoint(100f, 200f, SamPointLabel.FOREGROUND)
            val scaled = prompt.scale(xScale = 0.5f, yScale = 2f)
            val point = scaled.prompts.single() as PointPrompt
            assertEquals(50f, point.x)
            assertEquals(400f, point.y)
            assertEquals(SamPointLabel.FOREGROUND, point.label)
        }

        @Test
        fun `identity scale returns the same instance`() {
            val prompt = SamPrompt().addPoint(100f, 200f, SamPointLabel.FOREGROUND)
            assertTrue(prompt.scale(1f, 1f) === prompt, "expected identity short-circuit")
        }

        @Test
        fun `does not mutate the original prompt`() {
            val prompt = SamPrompt().addPoint(100f, 200f, SamPointLabel.FOREGROUND)
            prompt.scale(0.5f, 0.5f)
            val original = prompt.prompts.single() as PointPrompt
            assertEquals(100f, original.x)
            assertEquals(200f, original.y)
        }

        @Test
        fun `box keeps its type and scales both corners`() {
            val prompt = SamPrompt().addBox(10f, 20f, 100f, 200f)
            val scaled = prompt.scale(xScale = 2f, yScale = 0.5f)
            val box = scaled.prompts.single() as BoxPrompt
            assertEquals(20f, box.topLeft.x)
            assertEquals(10f, box.topLeft.y)
            assertEquals(200f, box.bottomRight.x)
            assertEquals(100f, box.bottomRight.y)
        }

        @Test
        fun `mask prompt passes through unchanged`() {
            val mask = FloatArray(256 * 256) { 0.25f }
            val prompt = SamPrompt().addMask(mask)
            val scaled = prompt.scale(2f, 2f)
            val scaledMask = scaled.prompts.filterIsInstance<MaskPrompt>().single().mask
            assertTrue(scaledMask === mask, "mask buffer should be passed through")
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
