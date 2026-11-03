# Diagnostic layer timeline: 01183c2992eb4153

Cold event medians are one-input diagnostics, not official per-shape medians.
CUDA activity includes device waits; union is not SM utilization.
Trace coordinates:0.1us. Aggregate span/union retain original resolution.

Recorded routing metadata valid: False.

| Case | Owned cold us | SGLang cold us | Owned profile span us | SGLang profile span us | Owned uncovered us |
|---|---:|---:|---:|---:|---:|
|d1s|85.696|221.248|80.481|194.658|2.497|
|d1f|115.872|247.904|102.208|219.998|2.686|
|d16s|165.568|318.784|152.641|292.446|2.978|
|d16f|228.448|472.864|216.897|444.124|2.145|
|d32s|191.264|302.720|180.033|282.047|1.888|
|d32f|353.184|682.496|342.301|660.195|1.056|
|v1s|165.472|349.568|152.865|321.602|1.664|
|v1f|137.568|312.960|126.081|284.256|1.953|
|v16s|292.416|572.608|281.118|553.977|2.947|
|v16f|415.968|825.568|404.960|804.611|2.944|
|v32s|343.264|691.680|332.577|674.364|2.817|
|v32f|543.936|1205.600|530.911|1181.633|3.263|
