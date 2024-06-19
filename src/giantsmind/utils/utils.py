import os
from pathlib import Path


def set_env_vars():
    env_file = Path(__file__).parent.parent.parent.parent / ".env"

    if not env_file.exists():
        raise Exception(".env file not found.")

    for line in env_file.read_text().split("\n"):
        if not line:
            continue
        key, value = line.split("=")
        os.environ[key] = value


if __name__ == "__main__":
    from pathlib import Path

    test_pdf = (
        Path("/home/pierre/Data/giants")
        / "Allen et al. - 2022 - A massive 7T fMRI dataset to bridge cognitive neur.pdf"
    )
    output_folder = Path("/home/pierre/Data/giants") / "pages"
    split_pdf(test_pdf, output_folder)
