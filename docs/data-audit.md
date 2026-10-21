# Dataset Audit Results

The audit covers all 975 files in the downloaded Kaggle dataset. Pillow decoded every file successfully, including one GIF and one extensionless JPEG. The class folders contain 539 AI-generated images and 436 real images.

## Class statistics

| Class | Images | Width min | Width median | Width max | Height min | Height median | Height max | File size min | File size median | File size max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AI generated | 539 | 225 px | 1,024 px | 6,000 px | 135 px | 800 px | 4,001 px | 11,738 B | 199,760 B | 5,626,183 B |
| Real | 436 | 183 px | 920 px | 8,495 px | 158 px | 720 px | 7,091 px | 11,311 B | 144,840 B | 10,195,352 B |

The real class contains the widest, tallest, and largest individual files. The AI-generated class has higher median width, height, and file size. A classifier could therefore learn acquisition or export characteristics instead of visual generation artifacts. Aspect-ratio-preserving padding, augmentation, and error analysis are used to reduce and monitor that risk.

## Integrity findings

- Corrupt or unreadable files: 0
- Files participating in exact-duplicate groups: 2
- Exact-duplicate groups: 1
- Duplicate files: `ai_generated/image33.jpg` and `ai_generated/image34.jpg`
- Unique files used for splitting: 973
- Deterministic split: 681 train, 146 validation, and 146 test

The two byte-identical files were removed before splitting so the same image cannot leak across partitions. The split manifests record relative paths, labels, and SHA-256 values and are committed to Git. Raw image bytes are represented by `data/raw.dvc` and stored through the DagsHub DVC remote.

## Formats and color modes

The AI-generated class contains 429 JPEG and 110 PNG images. The real class contains 403 JPEG and 33 PNG images. Most files are RGB, although both classes also contain grayscale, palette, and RGBA images. The input pipeline decodes all inputs to three RGB channels to provide a consistent tensor shape.

## Visual interpretation

The sample grid shows broad subject and composition variation within each class, which makes superficial subject-specific rules unreliable. The dimension histograms confirm overlapping distributions but also show long tails and class-level differences. Every image is therefore resized with padding rather than distortion, and the test analysis must check whether dimensions, borders, watermarks, or compression correlate with errors.
