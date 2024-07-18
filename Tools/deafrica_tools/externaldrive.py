import os.path
import google.auth

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive"]

def create_token():
    ''''
        Create the token json to connect the google drive.
        credential: provide the json creditials you would get from google service.
    '''
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow= InstalledAppFlow.from_client_secrets_file(credential, SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds

def list_gdrive():
    '''
    List the 10 recent files from the google drive
    '''
    creds = create_token()
    try:
        service = build("drive", "v3", credentials=creds)
      
        results = (service.files().list(pageSize=10, fields="nextPageToken, files(id, name)").execute())
        items = results.get("files", [])

        if not items:
            print("No files found.")
            return
        print("Files:")
        for item in items:
            print(f"{item['name']} ({item['id']})")
    except HttpError as error:
        print(f"An error occurred: {error}")


def upload_to_gdrive(file=None):
    '''
        Uploading files to google drive
    '''
    creds = create_token()
    try:
        # create drive api client
        service = build("drive", "v3", credentials=creds)

        file_metadata = {"name": file}
        media = MediaFileUpload(file)
        # pylint: disable=maybe-no-member
        file = (service.files().create(body=file_metadata, media_body=media).execute())
        print('File Uploaded successful')
    except HttpError as error:
        print(f"An error occurred: {error}")
        file = None
    return

