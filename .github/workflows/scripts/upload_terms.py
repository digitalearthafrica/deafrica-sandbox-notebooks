
   
import os
import sys
from poeditor import POEditorAPI

project_id = os.environ['POEDITOR_PROJECT_ID']
api_token = os.environ['POEDITOR_API_TOKEN']
client = POEditorAPI(api_token=api_token)

project = client.view_project_details(project_id)
print(f"Before update, {project['name']} (id: {project['id']}) has {project['terms']} terms.")

for file_path in sys.argv[1:]:
    print(f"Uploading file {file_path}...")
    update_results = client.update_terms(
        project_id=project_id,
        file_path=file_path
    )

    terms = update_results['terms']
    print("Terms updated:")
    for k, v in terms.items():
        print(f"\t{k}: {v}")


project = client.view_project_details(project_id)
print(f"After update, {project['name']} (id: {project['id']}) has {project['terms']} terms.")
