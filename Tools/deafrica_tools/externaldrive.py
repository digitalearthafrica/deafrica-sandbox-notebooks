import getpass
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# See https://developers.google.com/workspace/drive/api/guides/api-specific-auth#drive-scopes
# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def create_access_token(
    gdrive_credentials_dir: str = "/home/jovyan/Supplementary_data/DriveCredentials",
) -> Credentials:
    """
    Create the `token.json` file that stores the user's Google Drive
    access and refresh tokens.

    Parameters
    ----------
    gdrive_credentials_dir : str, optional
        Directory where Google Drive credentials are stored,
        by default "/home/jovyan/Supplementary_data/DriveCredentials".
        This directory should contain the `credentials.json` file.
        This is where the Google Drive API client will look for the user's
        credentials and where it will save the `token.json` file.

    Returns
    -------
    Credentials
        Google OAuth2 Credentials object for accessing Google Drive.
    """

    # Check if this was set before and inform the user
    prev_gdrive_credentials_dir = os.environ.get("GDRIVE_CREDENTIALS_DIR", None)
    if (
        prev_gdrive_credentials_dir is not None
        and gdrive_credentials_dir != prev_gdrive_credentials_dir
    ):
        print(
            f"Switching the gdrive credentials folder from {prev_gdrive_credentials_dir} "
            f"to {gdrive_credentials_dir}"
        )
    # Set the environment variable
    os.environ["GDRIVE_CREDENTIALS_DIR"] = gdrive_credentials_dir

    # Check if credentials have been uploaded
    credentials_json = os.path.join(gdrive_credentials_dir, "credentials.json")
    if not os.path.exists(credentials_json):
        raise FileNotFoundError(
            f"❌ Credentials file {credentials_json} not found:\n"
            "💡 Solution: Make sure you've uploaded the Google Drive credentials file "
            f"to the expected directory {gdrive_credentials_dir}.\n"
            f"🔍 Checked path: {os.path.abspath(credentials_json)}"
        )
    else:
        print(f"✅ Found credentials: {credentials_json}")

    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    token_json = os.path.join(gdrive_credentials_dir, "token.json")
    if os.path.exists(token_json):
        print(f"✅ Found existing access token: {token_json}")
        creds = Credentials.from_authorized_user_file(token_json, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_json, SCOPES)
            # Get the authorization URL
            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
            auth_url, _ = flow.authorization_url(prompt="consent")
            print("Please go to the following URL in your browser:")
            print(auth_url)
            # Prompt the user to enter the authorization code
            auth_code = getpass.getpass("Enter the authorization code here: ").strip()
            # Exchange the code for credentials
            flow.fetch_token(code=auth_code)
            creds = flow.credentials
            # Save the credentials for the next run
            with open(token_json, "w") as token:
                token.write(creds.to_json())
            print(f"✅ Saved access token to {token_json}")
    return creds


def get_folder_id(folder_name: str, creds: Credentials) -> str:
    """
    Get the unique, opaque ID for a folder in Google Drive.

    If multiple folders have the same name, returns the ID for the first
    folder.

    Parameters
    ----------
    folder_name : str
        Name of the folder to search for.

    creds : Credentials
        Google OAuth2 Credentials object for accessing Google Drive.

    Returns
    -------
    str
        Unique ID for the folder on Google Drive.
    """

    try:
        service = build("drive", "v3", credentials=creds)

        # Call the Drive v3 API
        results = (
            service.files()
            .list(
                q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
                fields="files(id, name, parents)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )

        items = results.get("files", [])

        if not items:
            print(f"No folder with the name {folder_name} found.")
            return None
        else:
            print(f"Found the following folder(s) matching the name '{folder_name}':")
            for item in items:
                print(f"Name: {item['name']} ID: {item['id']}")
            if len(items) > 1:
                print(
                    f"Found multiple folders matching the name {folder_name}. \n"
                    "Returning id for firs folder."
                )
            return items[0]["id"]

    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f"An error occurred: {error}")
