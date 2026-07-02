import os
import sys
import time
from poeditor import POEditorAPI

project_id = os.environ["POEDITOR_PROJECT_ID"]
api_token = os.environ["POEDITOR_API_TOKEN"]

client = POEditorAPI(api_token=api_token)

project = client.view_project_details(project_id)
print(f"Before update, {project['name']} (id: {project['id']}) has {project['terms']} terms.")

for i, file_path in enumerate(sys.argv[1:]):
    print(f"Uploading file {file_path}...")

    update_results = client.upload(
        project_id=project_id,
        updating=client.UPDATING_TERMS,   # uploads terms only
        file_path=file_path,
        sync_terms=False                  # set True only if you want missing terms deleted
    )

    print("Upload result:")
    print(update_results)

    # POEditor allows no more than one upload request every 30 seconds
    if i < len(sys.argv[1:]) - 1:
        time.sleep(30)

project = client.view_project_details(project_id)
print(f"After update, {project['name']} (id: {project['id']}) has {project['terms']} terms.")
