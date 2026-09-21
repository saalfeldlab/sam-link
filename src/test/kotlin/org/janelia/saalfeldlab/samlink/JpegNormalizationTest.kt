package org.janelia.saalfeldlab.samlink

import com.google.protobuf.ByteString
import kotlinx.coroutines.runBlocking
import org.janelia.saalfeldlab.samlink.TestUtils.rectangleImage
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.asTritonBytesElement
import org.janelia.saalfeldlab.samlink.encode.EncodeHelper.toJpegByteString
import org.janelia.saalfeldlab.samlink.models.EncodeParameter.Companion.getFloatArray
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import kotlin.math.sqrt
import kotlin.test.assertTrue

/**
 * One JPEG sent as `jpeg_image`, and that same JPEG decoded locally and sent as the raw tensor,
 * should return the same embeddings. They only do if the client applies the normalization the server
 * applies after its own decode, so this is what pins those constants. Comparing a raw image against a
 * JPEG of it would not; JPEG compression alone moves cosine further than a normalization error would.
 *
 * The input names and shapes are spelled out here rather than taken from the model definitions, so
 * the test pins the wire contract independently of them.
 */
@Tag("integration")
class JpegNormalizationTest {

    private fun cosine(left: FloatArray, right: FloatArray): Double {
        require(left.size == right.size) { "size ${left.size} != ${right.size}" }
        var dot = 0.0
        var leftNorm = 0.0
        var rightNorm = 0.0
        for (i in left.indices) {
            dot += left[i].toDouble() * right[i]
            leftNorm += left[i].toDouble() * left[i]
            rightNorm += right[i].toDouble() * right[i]
        }
        return dot / (sqrt(leftNorm) * sqrt(rightNorm))
    }

    /**
     * A JPEG of the test rectangle, and that same JPEG decoded back.
     *
     * The decoded image is handed to the encoder as ImageIO returns it, TYPE_3BYTE_BGR, which is the
     * shape a caller loading a JPEG off disk would have.
     */
    private fun jpegAndDecoded(edgeSize: Int): Pair<ByteString, BufferedImage> {
        val jpeg = rectangleImage(edgeSize, edgeSize).toJpegByteString(quality = 1.0f)
        return jpeg to ImageIO.read(ByteArrayInputStream(jpeg.toByteArray()))
    }

    private suspend fun jpegEmbedding(model: String, jpeg: ByteString, output: String): FloatArray {
        val input = InferenceInput(
            name = "jpeg_image",
            shape = longArrayOf(1, 1),
            datatype = "BYTES",
            data = jpeg.asTritonBytesElement()
        )
        return TritonEnv.newClient().use { client ->
            client.infer(model, listOf(input)).getFloatArray(output)
        }
    }

    private fun assertPathsAgree(name: String, raw: FloatArray, jpeg: FloatArray) {
        val cosine = cosine(raw, jpeg)
        println("$name raw-vs-jpeg cosine: $cosine")
        assertTrue(cosine > 0.9999, "$name: the two paths disagree, cosine $cosine")
    }

    @Test
    fun `sam 1`() = runBlocking {
        val (jpeg, decoded) = jpegAndDecoded(1024)
        val raw = TritonEnv.newSam1Encoder().use { encoder ->
            encoder.encode(decoded).use { it.imageEmbedding.floatBuffer.let { buf -> FloatArray(buf.remaining()).also(buf::get) } }
        }
        assertPathsAgree("sam1", raw, jpegEmbedding(TritonEnv.sam1Model(), jpeg, "image_embeddings"))
    }

    @Test
    fun `sam 2`() = runBlocking {
        val (jpeg, decoded) = jpegAndDecoded(1024)
        val raw = TritonEnv.newSam2Encoder().use { encoder ->
            encoder.encode(decoded).use { it.imageEmbedding.floatBuffer.let { buf -> FloatArray(buf.remaining()).also(buf::get) } }
        }
        assertPathsAgree("sam2", raw, jpegEmbedding(TritonEnv.sam2Model(), jpeg, "image_embed"))
    }

    @Test
    fun `sam 3 tracker`() = runBlocking {
        val (jpeg, decoded) = jpegAndDecoded(1008)
        val raw = TritonEnv.newSam3TrackerEncoder().use { encoder ->
            encoder.encode(decoded).use { it.imageEmbeddings2.floatBuffer.let { buf -> FloatArray(buf.remaining()).also(buf::get) } }
        }
        assertPathsAgree("sam3", raw, jpegEmbedding(TritonEnv.sam3TrackerModel(), jpeg, "image_embeddings.2"))
    }
}
