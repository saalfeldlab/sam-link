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
