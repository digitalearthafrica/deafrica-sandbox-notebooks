import os.path
import google.auth

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

# If modifying these scopes, delete the file credentials.
SCOPES = ["https://www.googleapis.com/auth/drive"]
credential_path = '../Supplementary_data/DriveCredentials/credentials.json'


def create_token():
    ''''
        credential: provide the json creditials you would get from google service.
    '''
    creds = None
    creds = service_account.Credentials.from_service_account_file(credential_path, scopes=SCOPES)
    return creds

def list_gdrive():
    '''
    List the 10 recent files from the google drive
    '''
    creds = create_token()
    try:
        service = build("drive", "v3", credentials=creds)
      
        results = (service.files().list(pageSize=20, fields="nextPageToken, files(id, name)").execute())
        items = results.get("files", [])

        if not items:
            print("No files found.")
            return
        print("Files:")
        for item in items:
            print(f"{item['name']} ({item['id']})")
    except HttpError as error:
        print(f"An error occurred: {error}")


def upload_to_gdrive(file_path=None):
    '''
        Uploading files to google drive
    '''
    creds = create_token()
    try:
        # create drive api client
        service = build("drive", "v3", credentials=creds)
        folder_path = '../Supplementary_data/DriveCredentials/googledrive_id.txt'
        #read the first line of the file
        folder_id = open(folder_path, "r").readline()

        file_metadata = {"name": file_path, "parents": [folder_id]}
        media = MediaFileUpload(file_path, resumable=True)
        # pylint: disable=maybe-no-member
        file = (service.files().create(body=file_metadata, media_body=media).execute())
        print('File Uploaded successful')
    except HttpError as error:
        print(f"An error occurred: {error}")
        file = None
    return


def delete_to_gdrive(file_id=None):
    '''
        deleting file from google drive
    '''
    creds = create_token()
    try:
        # create drive api client
        service = build("drive", "v3", credentials=creds)

        # pylint: disable=maybe-no-member
        response = service.files().delete(fileId=file_id).execute()
        print('File deleted successful')
    except HttpError as error:
        print(f"An error occurred: {error}")
        file = None
    return