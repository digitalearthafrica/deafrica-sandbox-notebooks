import os
import sys
import time
from pathlib import Path

from poeditor import POEditorAPI


def main() -> int:
    project_id = os.getenv("POEDITOR_PROJECT_ID")
    api_token = os.getenv("POEDITOR_API_TOKEN")

    if not project_id:
        print(
            "Error: POEDITOR_PROJECT_ID is not set.",
            file=sys.stderr,
        )
        return 1

    if not api_token:
        print(
            "Error: POEDITOR_API_TOKEN is not set.",
            file=sys.stderr,
        )
        return 1

    file_paths = [Path(path) for path in sys.argv[1:]]

    if not file_paths:
        print(
            "Usage: python upload_french.py <file1.po> [file2.po ...]",
            file=sys.stderr,
        )
        return 1

    missing_files = [path for path in file_paths if not path.is_file()]

    if missing_files:
        for path in missing_files:
            print(
                f"Error: File not found: {path}",
                file=sys.stderr,
            )
        return 1

    client = POEditorAPI(api_token=api_token)

    try:
        project = client.view_project_details(project_id)

        print(
            f"Uploading French translations to {project['name']} "
            f"(id: {project['id']})."
        )

        for index, file_path in enumerate(file_paths):
            print(f"Uploading French translation file: {file_path}")

            result = client.upload(
                project_id=project_id,
                updating=client.UPDATING_TRANSLATIONS,
                file_path=str(file_path),
                language_code="fr",
                overwrite=True,
            )

            print(f"Upload result for {file_path}:")
            print(result)

            if index < len(file_paths) - 1:
                print("Waiting before the next upload...")
                time.sleep(20)

        print("French translation upload completed successfully.")
        return 0

    except Exception as error:
        print(
            f"POEditor translation upload failed: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
