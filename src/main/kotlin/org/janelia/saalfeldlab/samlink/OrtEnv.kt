package org.janelia.saalfeldlab.samlink

import ai.onnxruntime.OrtEnvironment

/* method itself is synchronized, no need to add the overhead here.
* And we keep a reference to the delegate to check `isInitialized` elsewhere */
internal val ORT_ENV_LAZY = lazy(LazyThreadSafetyMode.NONE) { OrtEnvironment.getEnvironment() }
val ORT_ENV: OrtEnvironment by ORT_ENV_LAZY