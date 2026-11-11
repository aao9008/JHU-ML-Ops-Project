import re
import shutil
from collections import Counter
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def is_real_image(path: Path) -> bool:
    """True for actual image files; False for folders, Mac junk and other files."""
    return (
        path.is_file()
        and path.suffix.casefold() in IMAGE_SUFFIXES
        and not path.name.startswith("._")
        and "__MACOSX" not in path.parts
        and "prepared" not in path.parts  # our own exported copies
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


# personN_bacteria_15.jpeg (also personN_virus_921_1.jpeg)
PNEUMONIA_ID = re.compile(r"^(person\d+)_(bacteria|virus)_\d+", re.IGNORECASE)
# IM-0115-0001.jpeg, NORMAL2-IM-1427-0001.jpeg (also IM-0011-0001-0001.jpeg)
NORMAL_ID = re.compile(r"^(?:NORMAL2-)?(IM-\d+)-\d+", re.IGNORECASE)


def patient_group(image_path: Path) -> str:
    """Return a key identifying the patient an image belongs to.

    Images with the same key must always end up in the same split.
    """
    name = image_path.name

    match = PNEUMONIA_ID.match(name)
    if match:
        # The subtype is part of the key so that bacteria and virus images are never mixed in the same group.
        return f"{match.group(1)}_{match.group(2)}".lower()

    match = NORMAL_ID.match(name)
    if match:
        return match.group(1).upper()

    print(f"Warning: no patient ID found in {name}; using it as its own group")
    return image_path.stem

def build_dataframe(image_list: list[Path]) -> pd.DataFrame:
    """Build a DataFrame with columns for image path, label and patient group."""
    data = {
        "image_path": image_list,
        "label": [label_image(img) for img in image_list],
        "patient_group": [patient_group(img) for img in image_list],
    }
    df = pd.DataFrame(data)
    return df


# Split recipe: one of 7 folds is ~14.3% of the data (test); one of 6 folds of
# the remainder is ~14.3% of the total (val); the rest (~71.4%) is train.
TEST_FOLDS = 7
VAL_FOLDS = 6
RANDOM_STATE = 42


def take_one_fold(table: pd.DataFrame, n_splits: int):
    """Return the row labels (index) of one fold, keeping patient groups whole
    and class proportions about equal."""

    # Splitter is a generator; we only need the first fold. The rest are ignored.
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # The .split() method returns a tuple of (train_indices, test_indices) for each fold.
    _, fold_positions = next(splitter.split(table, table["label"], table["patient_group"]))

    # Positions count rows of `table`; .index turns them into the original row labels.
    return table.index[fold_positions]


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a 'split' column: 'train', 'val' or 'test'.

    About 71% train, 14% val, 14% test. A patient group is never divided
    between splits.
    """
    df = df.copy()
    df["split"] = "train"

    test_rows = take_one_fold(df, TEST_FOLDS)
    df.loc[test_rows, "split"] = "test"

    remaining = df[df["split"] != "test"]
    val_rows = take_one_fold(remaining, VAL_FOLDS)
    df.loc[val_rows, "split"] = "val"

    return df

def verify_splits(df: pd.DataFrame) -> None:
    """Check that the splits are valid; raise ValueError if not.

    Checks: every row has a valid split, no image is listed twice, and no
    patient group appears in more than one split. Also prints class counts.
    """
    # Every row must have a valid split (a missing value counts as invalid).
    if not df["split"].isin({"train", "val", "test"}).all():
        raise ValueError("Some rows have a missing or unexpected split value.")

    # The same image must not be listed twice.
    if not df["image_path"].is_unique:
        raise ValueError("Some image paths appear more than once.")

    # No patient group may appear in more than one split.
    groups_per_split = df.groupby("split")["patient_group"].apply(set)
    for split1, split2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = groups_per_split[split1] & groups_per_split[split2]
        if overlap:
            raise ValueError(f"Patient groups {sorted(overlap)} appear in both {split1} and {split2}.")

    # Class counts per split, for you to read
    counts = df.groupby(["split", "label"]).size().unstack()
    print(counts)

    percent = counts.div(counts.sum(axis=1), axis=0) * 100
    print(percent.round(1))


def export_images(df: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    """Copy images into <out_dir>/images/<label>/ and write <out_dir>/manifest.csv.

    The original files are not touched. Files already copied (same size) are
    skipped, so the function can be run again safely. Returns the manifest.
    """
    out_dir = Path(out_dir)

    # All labels share one folder per label, so file names must be unique.
    if not df["image_path"].map(lambda p: p.name).is_unique:
        raise ValueError("File names are not unique; copying would overwrite files.")

    relative_paths = []
    for row in df.itertuples():
        relative = Path("images") / row.label / row.image_path.name
        destination = out_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        already_copied = destination.exists() and destination.stat().st_size == row.image_path.stat().st_size
        if not already_copied:
            shutil.copy2(row.image_path, destination)

        relative_paths.append(relative.as_posix())  # forward slashes, relative to out_dir

    manifest = df.assign(
        relative_path=relative_paths,
        source_path=df["image_path"].astype(str),
    )[["relative_path", "label", "patient_group", "split", "source_path"]]

    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Copied {len(manifest)} images and wrote {out_dir / 'manifest.csv'}")
    return manifest


if __name__ == "__main__":
    # Example usage
    data_directory = Path.cwd().joinpath("data").resolve()  # absolute path; adjust as needed
    print(f"Preparing data from directory: {data_directory}")
    image_list = find_images(data_directory)

    # Label count test: expect normal 1583 and pneumonia_* adding up to 4273
    label_counts = Counter(label_image(img) for img in image_list)
    print(label_counts)

    # Group test: far fewer groups than images
    group_counts = Counter(patient_group(img) for img in image_list)
    print(f"Unique groups: {len(group_counts)}")
    print(f"Largest groups: {group_counts.most_common(3)}")

    df = build_dataframe(image_list)
    print(f"DataFrame shape: {df.shape}")
    print(df["label"].value_counts())
    print(df["patient_group"].nunique())

    df = assign_splits(df)
    print(df["split"].value_counts())

    verify_splits(df)

    export_images(df, data_directory / "prepared")
