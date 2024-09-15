import os


def get_source_list(file_path: str, ret_name=False):
    files = []
    names = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            l = line.replace("\n", "").split(" ")
            name = l[0]
            path = l[-1]
            files.append(path)
            names.append(name)
    if ret_name:
        return names, files
    return files
