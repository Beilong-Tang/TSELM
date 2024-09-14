import os


def make_path(file_path, is_dir=True):
    if is_dir:
        os.makedirs(file_path, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path
