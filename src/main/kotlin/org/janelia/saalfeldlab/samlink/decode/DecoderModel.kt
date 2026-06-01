package org.janelia.saalfeldlab.samlink.decode

import ai.onnxruntime.OrtSession
import org.janelia.saalfeldlab.samlink.ORT_ENV
import java.nio.file.Files
import java.nio.file.Paths

/**
 * Model families:
 * - SAM1: Supports point/mask/box prompts, outputs 4 candidate masks at 256x256
 * - SAM2.1: Supports point/mask/box prompts, outputs single mask at 256x256
 * - SAM3: Supports point/box prompts, outputs single mask at 288x288
 */
enum class DecoderModel(val resourceName: String) {
    SAM1("sam_vit_h_4b8939.onnx"),
    SAM2("sam2.1_hiera_large_decoder.onnx"),
    SAM3_TRACKER_FP16("sam3_tracker_fp16_decoder.onnx");

    fun load(): OrtSession {
        /* Load from packaged jar, or from file*/
        val bytes = SamDecoder::class.java.getResourceAsStream(resourceName)?.readAllBytes()
            ?: Files.readAllBytes(Paths.get(resourceName))
        return ORT_ENV.createSession(bytes)
    }
}