import os
import subprocess


def download_url(root: str, file_id: str, filename: str) -> None:
    if not isinstance(root, str):
        root = str(root)

    os.makedirs(root, exist_ok=True)

    cmd = "".join(
        [
            f"cd {root} && ",
            "wget --no-check-certificate --load-cookies /tmp/cookies.txt ",
            """\"https://drive.google.com/uc?export=download&confirm=true""",
            "$(wget --quiet --save-cookies /tmp/cookies.txt  ",
            "--keep-session-cookies --no-check-certificate ",
            f"'https://drive.google.com/uc?export=download&id={file_id}' ",
            "-O- | sed -rn "
            f"""'s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={file_id}\" -O""",
            f"{filename} && ",
            "rm -rf /tmp/cookies.txt",
        ]
    )

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )

    if process.returncode != 0:
        raise Exception(
            f"Error downloading file {filename} from Google Drive. "
            "Please download the file manually."
        )
