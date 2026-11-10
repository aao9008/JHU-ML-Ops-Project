from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def is_real_image(path: Path) -> bool:
    """True for actual image files; False for folders, Mac junk and other files."""
    return (
        path.is_file()
        and path.suffix.casefold() in IMAGE_SUFFIXES
        and not path.name.startswith("._")
        and "__MACOSX" not in path.parts
        and path.parts.count("chest_xray") <= 1
    )


def find_images(data_path: str | Path) -> list[Path]:
    """Return a sorted list of all image files under data_path."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

    images = sorted(p for p in data_path.rglob("*") if is_real_image(p))
    print(f"Total image files found: {len(images)}")
    return images



def label_image(image_path: Path) -> str:
    """Label an image from its filename: 'normal', 'bacteria' or 'virus'.

    The parent folder is only a sanity check; a disagreement raises an error.
    """
    name = image_path.name.lower()
    folder = image_path.parent.name

    if folder not in {"NORMAL", "PNEUMONIA"}:
        raise ValueError(f"Unexpected folder '{folder}' for {image_path}")

    has_bacteria = "bacteria" in name
    has_virus = "virus" in name
    if has_bacteria and has_virus:
        raise ValueError(f"Ambiguous filename (bacteria and virus): {image_path}")

    if has_bacteria:
        pneumonia_type = "pneumonia_bacteria"
    elif has_virus:
        pneumonia_type = "pneumonia_virus"
    else:
        pneumonia_type = None  # no pneumonia keyword, so the filename says normal

    label = pneumonia_type or "normal"

    # Sanity check: the folder must agree with the filename.
    folder_says_pneumonia = folder == "PNEUMONIA"
    if folder_says_pneumonia != (pneumonia_type is not None):
        raise ValueError(f"Folder '{folder}' disagrees with filename label '{label}': {image_path}")

    return label


if __name__ == "__main__":
    # Example usage
    data_directory = Path.cwd().joinpath("data")  # Adjust the path as needed
    print(f"Preparing data from directory: {data_directory}")
    image_list = find_images(data_directory)

    # Label count test: expect normal 1583 and pneumonia_* adding up to 4273
    label_counts = Counter(label_image(img) for img in image_list)
    print(label_counts)
