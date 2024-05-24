# Inspired by https://raw.githubusercontent.com/digitalearthafrica/deafrica-docs/c6006f7d6f25a9a72b9433e55e7a1c3e34fe2d9e/download_translations.py
# This file downloads to the translations from POEDITOR to local 

import os
from pathlib import Path
from poeditor import POEditorAPI


def download_translation(file_path, project_id, api_token):
    client = POEditorAPI(api_token=api_token)
    # Create the parent folder of the file path specified.
    os.makedirs(Path(file_path).absolute().parent, exist_ok=True)
    # Get the file type.
    file_type = file_path[-2:]

    client.export(
        project_id=project_id,
        language_code='fr',
        file_type=file_type,
        local_file=file_path,
    )


if __name__ == '__main__':
    project_id = os.environ['POEDITOR_PROJECT_ID']
    api_token = os.environ['POEDITOR_API_TOKEN']

    file_path = 'Tools/deafrica_tools/locales/fr/LC_MESSAGES/deafrica_tools.mo'
    print(f"Downloading translation to {file_path}")
    download_translation(file_path, project_id, api_token)

    file_path = 'Tools/deafrica_tools/locales/fr/LC_MESSAGES/deafrica_tools.po'
    print(f"Downloading translation to {file_path}")
    download_translation(file_path, project_id, api_token)

    