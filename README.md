[![Build Status](https://github.com/saalfeldlab/sam-link/actions/workflows/build.yml/badge.svg)](https://github.com/saalfeldlab/sam-link/actions/workflows/build.yml)

# SAM Link

### Segment Anything Kotlin Abstraction Layer

Intended to be used as a server-client model. The [SamEncoder.kt](src/main/kotlin/org/janelia/saalfeldlab/samlink/encode/SamEncoder.kt) 
interface is intended to query image embeddings from a Triton server using gRPC.

The returned embedding are then provide to a [SamDecoder.kt](src/main/kotlin/org/janelia/saalfeldlab/samlink/decode/SamDecoder.kt) along with 
a prompt to run the decoding locally. 

The usecase is to offload the slower/more expensive image embedding to a server, while running the decoder locally
to allow for real-time prompt refinement.

### Supported Models

The encoder models are dependent on what the triton server endpoint supports. However, the decoder models
must align with the encoder models. Below is the current list of supported decoders provided, and the matched encoder
models. All decoder models are `onnx` models. 

#### SAM 1

The `sam_vit_h_4b8983` model is supported by [Sam1Decoder.kt](src/main/kotlin/org/janelia/saalfeldlab/samlink/decode/Sam1Decoder.kt).

#### SAM 2

The `sam2.1_hiera_large_decoder` model is supported by [Sam2Decoder.kt](src/main/kotlin/org/janelia/saalfeldlab/samlink/decode/Sam2Decoder.kt).

#### SAM 3

Currently only the SAM 3 tracker model is supported. An fp16 variant is provided as `sam3_tracker_fp16_decoder` and 
supported by [Sam3Decoder.kt](src/main/kotlin/org/janelia/saalfeldlab/samlink/decode/Sam3Decoder.kt).


See [licenses_readme.md](src/main/resources/org/janelia/saalfeldlab/samlink/decode/licenses/licenses_readme.md) for 
license information on the provided models. 

## Example API Usage

Encode against a Triton server, decode locally, threshold the logits at `0`, and write the binary
mask to disk:

```kotlin

/* Any BufferedImage will do */
val image = ImageIO.read(File("input.png"))

/* Point the encoder to the server */
val encoder = Sam2TritonEncoder(
    host = "triton.local",
    port = 8001,
    model = "sam2.1_hiera_large_encoder"
)
val decoder = Sam2Decoder(DecoderModel.SAM2.load())

encoder.use { enc -> 
    decoder.use { dec -> 
        runBlocking {
            enc.encode(image).use { embedding ->
            /* Adds a FOREGROUND point at the center of the image. */
            val prompt = SamPrompt().addPoint(
                image.width / 2f,
                image.height / 2f,
                SamPointLabel.FOREGROUND,
            )
            val result = dec.decode(embedding, prompt)

            val mask = result.bestMask
            val size = result.maskSize
            val out = BufferedImage(size, size, BufferedImage.TYPE_BYTE_GRAY)
            val pixels = (out.raster.dataBuffer as DataBufferByte).data
            /* threshold the mask */
            for (i in mask.indices) 
                pixels[i] = if (mask[i] > 0f) 255.toByte() else 0
                
            ImageIO.write(out, "png", File("mask.png"))
            }
        }
    } 
}
```

