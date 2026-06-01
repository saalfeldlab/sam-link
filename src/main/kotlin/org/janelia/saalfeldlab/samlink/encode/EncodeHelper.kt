package org.janelia.saalfeldlab.samlink.encode

import com.google.protobuf.ByteString
import com.google.protobuf.UnsafeByteOperations
import java.awt.image.BufferedImage
import java.nio.ByteBuffer
import java.nio.ByteOrder

object EncodeHelper {

    /**
     * Given an image with dimensions [imageWidth]x[imageHeight], calculate the size of the image data
     * after scaling such that the longest edge becomes [maxEdgeSize]. The resulting dimensions may be
     * larger or smaller than the original dimensions, but will maintain the aspect ratio.
     */
    fun scaleToMaxEdgeSize(imageWidth: Int, imageHeight: Int, maxEdgeSize: Int): Pair<Int, Int> {
        val actualMaxEdge = maxOf(imageWidth, imageHeight)

        if (actualMaxEdge == maxEdgeSize)
            return imageWidth to imageHeight

        val scale = maxEdgeSize.toFloat() / actualMaxEdge.toFloat()

        return when {
            imageWidth == imageHeight -> maxEdgeSize to maxEdgeSize
            imageWidth > imageHeight -> maxEdgeSize to (imageHeight * scale).toInt()
            else  -> (imageWidth * scale).toInt() to maxEdgeSize
        }
    }

    /**
     * Scale [image] to a resulting [java.awt.image.BufferedImage] of the [targetWidth]×[targetHeight].
     * Maintains the aspect ratio and pads with black to fit the target dimensions.
     * If [image] dimensions match the target dimensions this will return the original image.
     */
    fun scaleWithPadding(image: BufferedImage, contentWidth: Int, contentHeight: Int, targetWidth: Int, targetHeight: Int): BufferedImage {
        if (image.width == targetWidth && image.height == targetHeight)
            return image

        return BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB).apply {
            val graphics = createGraphics()
            graphics.drawImage(image, 0, 0, contentWidth, contentHeight, null)
            graphics.dispose()
        }
    }

    /**
     * Convert a [BufferedImage] to a [com.google.protobuf.ByteString] in channel-height-width format (CHW).
     * Assumes input is [BufferedImage.TYPE_INT_RGB].
     *
     * Maps RGB values from [0,255] to [outputRange]
     *
     * @param outputRange range to normalize pixel values to, default is [0f, 1f]
     */
    fun BufferedImage.intRGBtoCHW(outputRange : ClosedFloatingPointRange<Float> = 0f..1f): ByteString {
        val dataBuffer = raster.dataBuffer
        val numPixels = width * height
        val rgbCount = 3 * numPixels
        val buffer = ByteBuffer.allocate(rgbCount * 4).order(ByteOrder.LITTLE_ENDIAN)
        val floatView = buffer.asFloatBuffer()

        val rgbRange = 255f
        val scale = (outputRange.endInclusive - outputRange.start) / rgbRange
        val offset = outputRange.start

        /* reorder to channel-height-width format and normalize to output range*/
        repeat(numPixels) { redIdx ->
            val greenIdx = redIdx + numPixels
            val blueIdx = redIdx + numPixels * 2

            val rgbInt = dataBuffer.getElem(redIdx)
            val red = ((rgbInt shr 16) and 0xFF) * scale + offset
            val green = ((rgbInt shr 8) and 0xFF) * scale + offset
            val blue = (rgbInt and 0xFF) * scale + offset

            floatView.put(redIdx, red)
            floatView.put(greenIdx, green)
            floatView.put(blueIdx, blue)
        }

        return UnsafeByteOperations.unsafeWrap(buffer)
    }
}
