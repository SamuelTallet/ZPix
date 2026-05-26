"""Resolutions and aspect ratios."""

from re import search


def get_aspects_and_resolutions() -> tuple:
    """Get aspect ratios and resolutions.

    Returns:
        Tuple of (
            resolutions by aspect,
            default resolution choices,
            aspect ratio choices,
            default aspect ratio
        )
    """
    # TODO Move hardcodes to curated_models.json?
    default_aspect_ratio = "16:9"

    resolutions_by_aspect = {
        "1:1": [
            "1024x1024",
            "1280x1280",
            "1440x1440",
        ],
        "16:9": [
            "1280x720",
            "1920x1088",
        ],
        "9:16": [
            "720x1280",
            "1088x1920",
        ],
        "4:3": [
            "1152x864",
            "1440x1088",
            "1920x1440",
        ],
        "3:4": [
            "864x1152",
            "1088x1440",
            "1440x1920",
        ],
        "5:4": [
            "960x768",
            "1280x1024",
            "2000x1600",
        ],
        "4:5": [
            "768x960",
            "1024x1280",
            "1600x2000",
        ],
        "3:2": [
            "1088x720",
            "1536x1024",
            "2048x1360",
        ],
        "2:3": [
            "720x1088",
            "1024x1536",
            "1360x2048",
        ],
        "16:10": [
            "1280x800",
            "1440x912",
            "1920x1200",
        ],
        "10:16": [
            "800x1280",
            "912x1440",
            "1200x1920",
        ],
        "21:9": [
            "1568x672",
            "1792x768",
            "2016x864",
        ],
        "9:20": [
            "720x1600",
            "864x1920",
        ],
        "2:1": [
            "1536x768",
            "2048x1024",
        ],
    }

    default_resolution_choices = resolutions_by_aspect[default_aspect_ratio]
    aspect_ratio_choices = list(resolutions_by_aspect.keys())

    return (
        resolutions_by_aspect,
        default_resolution_choices,
        aspect_ratio_choices,
        default_aspect_ratio,
    )


def parse_resolution(resolution: str) -> tuple[int, int]:
    """Parse resolution string into width and height.

    Args:
        resolution: Resolution string in format "WIDTHxHEIGHT".

    Raises:
        ValueError: If resolution string is invalid.

    Returns:
        Tuple of (width, height) as integers.
    """
    match = search(r"(?P<width>\d+)\s*x\s*(?P<height>\d+)", resolution)

    if not match or len(match.groups()) != 2:
        raise ValueError(f"{resolution} isn't in WIDTHxHEIGHT format")

    width = int(match.group("width"))
    height = int(match.group("height"))

    return width, height
