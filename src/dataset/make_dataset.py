import os
import shutil
import numpy as np


RAW_DATA_DIR = os.path.join("..", "..", "archive", "data")
DATA_DIR = os.path.join("..", "..", "data")
CLASSES = ['desert', 'water', 'cloudy', 'green_area']


def create_directories(dirs: list[str], classes: list[str]) -> None:
    for dir in dirs:
        for class_name in classes:
            os.makedirs(f"{DATA_DIR}/{dir}/{class_name}", exist_ok=True)


def train_valid_test_split_by_percentage(files: list[str], percs: list[float]) -> list[np.array]:
    n_files = len(files)
    train_split_index = int(n_files * percs[0])
    valid_split_index = int(n_files * (percs[0] + percs[1]))
    train_files, valid_files, test_files = np.split(np.array(files), [train_split_index, valid_split_index])
    return train_files, valid_files, test_files


if __name__ == "__main__":
    create_directories(dirs=['train', 'valid', 'test'], classes=CLASSES)

    for class_name in CLASSES:
        src = os.path.join(RAW_DATA_DIR, class_name)
        files_for_current_class = os.listdir(src)

        train_files, valid_files, test_files = train_valid_test_split_by_percentage(
            files=files_for_current_class,
            percents=[0.70, 0.15],
        )

        for (directory, files) in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
            for file in files:
                shutil.copyfile(
                    src=f"{RAW_DATA_DIR}/{class_name}/{file}",
                    dst=f"{DATA_DIR}/{directory}/{class_name}/{file}"
                )
