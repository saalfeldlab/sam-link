# Encoder fixtures

These binaries are encoder outputs captured against a real Triton server and replayed
through the local ONNX decoders by the `Sam{1,2,3Tracker}DecoderTest` classes. Replaying
saved tensors lets us smoke-test the decode pipeline on every `mvn test` without standing
up a Triton instance.

## Layout

```
fixtures/
  sam1/synthetic/imageEmbedding.bin.gz
  sam2/synthetic/{imageEmbedding,highResFeats0,highResFeats1}.bin.gz
  sam3tracker/synthetic/{imageEmbeddings0,imageEmbeddings1,imageEmbeddings2}.bin.gz
```

`synthetic` refers to the deterministic procedural image produced by
`TestImages.generateSyntheticImage` — see the helper for the recipe.

## File format

Each `.bin.gz` is gzipped, with this layout once decompressed:

| offset | bytes  | meaning                                        |
| ------ | ------ | ---------------------------------------------- |
| 0      | 4      | magic `"SLF1"` (ASCII)                         |
| 4      | 1      | element size — `4` for fp32, `2` for fp16      |
| 5      | 4      | rank (`int32` little-endian)                   |
| 9      | 4·rank | shape (`int32[rank]` little-endian)            |
| ...    | rest   | payload — `elementSize * product(shape)` bytes |

Sam1 and Sam2 fixtures are fp32. Sam3Tracker fixtures are fp16 to keep the committed
binaries below ~20 MB total; the loader expands them to fp32 before building the ONNX
tensor, so the decoder still sees full single-precision input.

`FixtureIO` is the single source of truth for both reading and writing this format.

## Regenerating

The fixtures are produced by `FixtureGeneratorTest`, which lives in the integration
tier and requires a reachable Triton server with the configured encoder models loaded:

```sh
TRITON_HOST=triton.example  \
TRITON_PORT=8001            \
SAM1_MODEL=sam_vit_b_multi_encoder      \
SAM2_MODEL=sam2.1_large_encoder         \
SAM3_TRACKER_MODEL=sam3_tracker_encoder \
mvn -P integration test -Dtest=FixtureGeneratorTest
```

That writes fresh fixtures to `target/generated-fixtures/<variant>/synthetic/`. The
last line of each test logs a ready-to-paste `cp` command for promoting them into
this directory. After promoting, run `mvn test` to confirm the decoder tests pick the
new fixtures up.

## When to regenerate

- The synthetic image generator changes — fixtures encode that exact image, so a
  pixel-level change invalidates them.
- The Triton-side encoder model version changes in a way that would alter outputs.
- The `FixtureIO` on-disk format changes (bump the magic if you ever do this).
