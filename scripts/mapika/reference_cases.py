"""Representative requests for the pinned Mapika v12 source oracle."""

import base64
import struct
import zlib


def choice(count):
    return {
        "type": "choice",
        "instructions": "Which option fits?",
        "criteria": {f"option_{index}": None for index in range(count)},
    }


def score(count):
    return {
        "type": "score",
        "instructions": "How certain is this?",
        "criteria": [f"level {index}" for index in range(count)],
    }


def png_chunk(name, data):
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", zlib.crc32(name + data))
    )


def image_asset(rgb, width=84, height=56):
    rows = (b"\x00" + bytes(rgb) * width) * height
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(rows))
        + png_chunk(b"IEND", b"")
    )
    return {
        "type": "image",
        "format": "png",
        "data_base64": base64.b64encode(payload).decode("ascii"),
    }


def case(id_, state, questions, *, independent=True, assets=None):
    request = {"state": state, "questions": questions, "independent": independent}
    if assets is not None:
        request["assets"] = assets
    return {"id": id_, "request": request}


def fixtures():
    typed = {
        "10": {
            "type": "choice",
            "instructions": {"text": "Which queue?", "exceptions": [None, "café"]},
            "criteria": {"10": "Billing", "2": {"text": "Support"}},
        },
        "2": {
            "type": "noul",
            "criteria": {"false": "No duplicate charge", "true": "A duplicate charge"},
        },
        "rating": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": {"2": "2: urgent", "0": "0: low", "1": "1: medium"},
        },
    }
    state = {
        "ticket": "I was charged twice for café supplies.",
        "attempts": list(range(8)),
    }
    red = image_asset((255, 0, 0))
    blue = image_asset((0, 0, 255), 56, 84)
    colors = {
        "color": {
            "type": "choice",
            "instructions": "What color is the first image?",
            "criteria": ["red", "blue"],
        }
    }
    return [
        case("typed-independent", state, typed),
        case("typed-packed", state, typed, independent=False),
        case("score-nonisolated", state, {"s": {**score(3), "isolated": False}}),
        case(
            "choice-narrow-wide",
            "Identify the label.",
            {"ten": choice(10), "eleven": choice(11)},
            independent=False,
        ),
        case("choice-limit", "Identify the label.", {"limit": choice(255)}),
        case("choice-over-limit", "Identify the label.", {"limit": choice(256)}),
        case("score-limit", state, {"limit": score(10)}),
        case("score-over-limit", state, {"limit": score(11)}),
        case("empty-questions", state, {}),
        case("state-at-limit", " a" * 32765, {"one": choice(2)}),
        case("state-truncates", " a" * 32766, {"one": choice(2)}),
        case("row-limit", "A", {f"q_{i}": choice(2) for i in range(1024)}),
        case("row-over-limit", "A", {f"q_{i}": choice(2) for i in range(1025)}),
        case("image-single", {"photo": "img"}, colors, assets={"img": red}),
        case(
            "image-repeated",
            {"photos": ["img2", "img", "img2", "img"]},
            colors,
            assets={"img": red, "img2": blue},
        ),
        case("image-over-limit", {"photos": ["img"] * 5}, colors, assets={"img": red}),
        case(
            "image-unreferenced", "There is no photo here.", colors, assets={"img": red}
        ),
    ]
