from microbundlecompute import optional_preprocessing as op


def test_rename_folder():
    

def rename_folder(folder_path: Path, folder_name: str, new_folder_name: str) -> Path:
    """Given a path to a directory, a folder in the given directory, and a new folder name. 
    Will change the name of the folder."""
    original_folder_path = folder_path.joinpath(folder_name).resolve()
    new_folder_path = folder_path.joinpath(new_folder_name).resolve()
    os.rename(original_folder_path,new_folder_path)
    return new_folder_path