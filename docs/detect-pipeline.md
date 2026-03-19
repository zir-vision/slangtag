# How the Detection Pipeline Works

This detector is a staged funnel. It starts with raw grayscale pixels and progressively narrows the search to only shapes that look like valid tags, then decodes and verifies them.

## Big Picture

The pipeline answers four questions in sequence:

1. Where are high-contrast black/white regions?
2. Which region boundaries look like tag borders?
3. Which of those boundaries form plausible quadrilaterals?
4. Do those quads decode to valid AprilTag payloads?

At each step, weak candidates are discarded and strong candidates get more expensive processing.

## Stage 1: Build a Clean Binary View of the Image

The detector first normalizes local contrast and thresholds the image so pixels become mostly black/white classes.

Why this exists:

- Real scenes have uneven lighting.
- A global threshold fails in shadows or glare.
- Local min/max filtering makes thresholding more stable per neighborhood.

Optional decimation (downsampling) can be applied to reduce work and improve throughput, trading some precision for speed.

## Stage 2: Find Connected Blobs and Their Boundaries

After thresholding, connected-component labeling groups neighboring pixels into blobs.

Then the pipeline looks at blob-to-blob boundary transitions and keeps only boundaries that are large enough to matter.

Why this exists:

- Tags are made of structured black/white regions.
- Small noisy blobs (texture, sensor noise) are not useful.
- Boundary transitions carry more geometric signal than raw pixels.

## Stage 3: Form Candidate Border Extents

Boundary points are grouped into extents that could represent sides/rings around a marker.

Extents are filtered using size and cluster constraints so only physically plausible tag-like structures remain.

Why this exists:

- Many edges are present in natural scenes.
- Tag borders have expected scale and continuity.
- Early geometric filtering removes most false positives cheaply.

## Stage 4: Fit Lines, Detect Peaks, and Assemble Quads

For each surviving extent, the detector computes ordered points, fits line behavior, and finds strong corner-like peak events.

Peaks are grouped and combined into quadrilateral hypotheses. Quads must satisfy geometric quality constraints (fit error, angular consistency, minimum size).

Why this exists:

- A tag is a planar square-like target.
- Good quads are a strong prior before decode.
- Geometry rejects a large amount of non-tag clutter.

## Stage 5: Sample the Tag Grid and Decode Bits

Each fitted quad is warped into a canonical grid, then sampled cell-by-cell to obtain an 8x8 bit map (including border).

The detector:

1. Checks border consistency (the border should match expected polarity).
2. Optionally tries inverted polarity and keeps whichever is more consistent.
3. Rejects candidates with too many border errors.
4. Extracts the inner payload bits.

Why this exists:

- Perspective, blur, and lighting can distort raw samples.
- Border checks provide a strong validity test before ID lookup.

## Stage 6: Identify AprilTag ID and Rotation

The payload bits are converted to a compact code and matched against the AprilTag 36h11 family across all 4 rotations.

Two modes are possible:

- Exact match only.
- Error-correcting match (accept nearest code within a configured Hamming distance budget).

If accepted, the reported corner order is rotated to align with decoded orientation.

## What Makes It Robust

- Local thresholding instead of global thresholding.
- Progressive filtering from cheap tests to expensive tests.
- Joint use of geometry (quad quality) and coding theory (family match).
- Optional inversion handling for contrast polarity changes.
- Error tolerance via controlled correction budget.

## What the Output Represents

Each final detection includes:

- Tag identity (if decode succeeds).
- Corner coordinates in image space.
- Orientation-consistent corner ordering.
- A confidence-like score from quad fitting.
- Raw sampled bits (with border) and extracted payload bits.

In practice, the detector is designed so that by the time decode happens, candidates are already strongly constrained by connected regions and quad geometry, which is why decoding can be both fast and reliable.
