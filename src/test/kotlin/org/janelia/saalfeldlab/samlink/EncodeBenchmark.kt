package org.janelia.saalfeldlab.samlink

import kotlinx.coroutines.Job
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.janelia.saalfeldlab.samlink.EncodeBenchmark.ImageDimensions.*
import org.janelia.saalfeldlab.samlink.decode.DecoderModel.SAM1
import org.janelia.saalfeldlab.samlink.decode.DecoderModel.SAM2
import org.janelia.saalfeldlab.samlink.decode.DecoderModel.SAM3_TRACKER_FP16
import org.janelia.saalfeldlab.samlink.decode.Sam1Decoder
import org.janelia.saalfeldlab.samlink.decode.Sam2Decoder
import org.janelia.saalfeldlab.samlink.decode.Sam3TrackerDecoder
import org.junit.jupiter.api.Tag
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
import kotlin.time.measureTime

@Tag("benchmark")
class EncodeBenchmark {

    enum class BenchmarkType(val count: Int = 10) {
        Parallel,
        Sequential
    }

    enum class ImageDimensions(val width: Int, val height: Int) {
        SQUARE(1024, 1024),
        SAM3_SQUARE(1008, 1008)
    }

    @ParameterizedTest
    @EnumSource(BenchmarkType::class)
    fun `Sam 1`(type: BenchmarkType) = runBlocking {
        val encoder = TritonEnv.newSam1Encoder()
        val model = SAM1

        val time = benchmark(type, type.count) {
            launch {
                testSquareWithBorder(
                    width = SQUARE.width,
                    height = SQUARE.height,
                    borderPercent = 0.25,
                    encoder = encoder,
                    decode = null
                )
            }
        }
        println("${model}($type) Elapsed: $time")
    }

    @ParameterizedTest
    @EnumSource(BenchmarkType::class)
    fun `Sam 2`(type: BenchmarkType) = runBlocking {
        val encoder = TritonEnv.newSam2Encoder()
        val model = SAM2

        val time = benchmark(type, type.count) {
            launch {
                testSquareWithBorder(
                    width = SQUARE.width,
                    height = SQUARE.height,
                    borderPercent = 0.25,
                    encoder = encoder,
                    decode = null
                )
            }
        }
        println("${model}($type) Elapsed: $time")
    }

    fun benchmark(type: BenchmarkType, count : Int, launch : () -> Job) = runBlocking {
        measureTime {
            when (type) {
                BenchmarkType.Parallel -> (0 until count).map { launch() }.also { it.joinAll() }
                BenchmarkType.Sequential -> repeat(count) { launch().join() }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(BenchmarkType::class)
    fun `Sam 3`(type: BenchmarkType) = runBlocking {
        val encoder = TritonEnv.newSam3TrackerEncoder()
        val model = SAM3_TRACKER_FP16

        val time = benchmark(type, type.count) {
            launch {
                testSquareWithBorder(
                    width = SAM3_SQUARE.width,
                    height = SAM3_SQUARE.height,
                    borderPercent = 0.25,
                    encoder = encoder,
                    decode = null
                )
            }
        }
        println("${model}($type) Elapsed: $time")
    }
}
