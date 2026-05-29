package org.janelia.saalfeldlab.samlink.encode

/**
 * Base interface for encode options.
 * Each encoder type defines its own options class.
 */
sealed interface EncodeOptions

abstract class TritonEncodeOptions(var priority: Long) : EncodeOptions

/**
 * Options for SAM2 Triton encoder.
 */
class Sam1TritonOptions(priority: Long = 5) : TritonEncodeOptions(priority)

/**
 * Options for SAM2 Triton encoder.
 */
class Sam2TritonOptions(priority: Long = 5) : TritonEncodeOptions(priority)

/**
 * Options for SAM3 Tracker Triton encoder.
 */
class Sam3TrackerTritonOptions(priority: Long = 5) : TritonEncodeOptions(priority)