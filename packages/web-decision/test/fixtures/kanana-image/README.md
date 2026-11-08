# Kanana image parity fixtures

These small fixtures were generated with Pillow 12.1.1. Each `.rgb` file uses
the pinned `processing_kanana_v_decide.py::to_rgb` conversion: open the image,
composite mode `RGBA` over white using its alpha channel, and call
`convert('RGB').tobytes()` for every other mode. This distinction matters for
16-bit RGBA: Pillow's direct `convert('RGB')` drops alpha, while `to_rgb`
composites it.

The JPEG with EXIF orientation 6 is saved in its original 64×32 raster; its
golden bytes retain that raster orientation.

The PNG fixtures cover RGBA white compositing, palette transparency, RGB
`tRNS`, and luminance-plus-alpha (`LA`). Their goldens record Pillow's RGB
conversion behavior for those source modes.

The 16-bit TIFF fixtures cover unsigned `I;16` and `I;16B` grayscale plus
chunky 16-bit RGB. Samples include 255, 256, 4095, and 4096. Little- and
big-endian files exercise uncompressed, LZW, and Adobe Deflate strips; the wide
LZW fixture exercises TIFF code-width changes, and the RGB Deflate fixture
contains multiple strips. Goldens come from Pillow 12.1.1 `convert('RGB')`.
