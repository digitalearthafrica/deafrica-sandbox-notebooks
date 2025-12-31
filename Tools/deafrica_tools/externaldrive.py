import json
import os

import yaml
from pydrive2.auth import GoogleAuth

# See https://developers.google.com/workspace/drive/api/guides/api-specific-auth#drive-scopes
# If modifying these scopes, delete the file token.json.
# View and edit Drive files created or opened by the app.
# SCOPES = ["https://www.googleapis.com/auth/drive.file"]
# View and manage all your Drive files.
SCOPES = ["https://www.googleapis.com/auth/drive"]


def write_settings_file(
    gdrive_credentials_dir: str = "/home/jovyan/Supplementary_data/DriveCredentials",
):
    """
    Write setting file for custom pydrive2 authentication

    Parameters
    ----------
    gdrive_credentials_dir : str, optional
        Directory where Google Drive credentials are stored,
        by default "/home/jovyan/Supplementary_data/DriveCredentials".
        This directory should contain the `credentials.json` file.
        This is where the pydrive2 client will look for the user's
        credentials and where it will save the `token.json` file.

    Returns
    -------
    str
        Path to the created settings.yaml file
    """
    settings_yaml = os.path.join(gdrive_credentials_dir, "settings.yaml")
    if not os.path.exists(settings_yaml):
        gdrive_config = {
            "client_config_backend": "file",
            "client_config_file": os.path.join(gdrive_credentials_dir, "credentials.json"),
            "save_credentials": True,
            "save_credentials_backend": "file",
            "save_credentials_file": os.path.join(gdrive_credentials_dir, "token.json"),
            "get_refresh_token": True,
            "oauth_scope": SCOPES,
        }
        with open(settings_yaml, "w") as file:
            # Use default_flow_style=False for block style (more readable)
            # Use sort_keys=False to preserve the dictionary's key order
            # (Python 3.7+ ensures this by default)
            yaml.dump(gdrive_config, file, default_flow_style=False, sort_keys=False)
        print(f"Wrote settings file for custom pydrive2 authentication to: {settings_yaml}")
    else:
        print(f"✅ Found existing settings file: {settings_yaml}")
    return settings_yaml


def update_redirect_uris(
    gdrive_credentials_dir: str = "/home/jovyan/Supplementary_data/DriveCredentials",
):
    """
    Update the redirect URI in the credentials.json file to
    "urn:ietf:wg:oauth:2.0:oob" for out-of-band authentication.

    Parameters
    ----------
    gdrive_credentials_dir : str, optional
        Directory where Google Drive credentials are stored,
        by default "/home/jovyan/Supplementary_data/DriveCredentials".
        This directory should contain the `credentials.json` file.
    """
    # Read settings file
    settings_yaml = os.path.join(gdrive_credentials_dir, "settings.yaml")
    with open(settings_yaml, "r") as f:
        settings = yaml.safe_load(f)

    credentials_json = settings["client_config_file"]
    if not os.path.exists(credentials_json):
        raise FileNotFoundError(
            f"❌ Credentials file {credentials_json} not found:\n"
            "💡 Solution: Make sure you've uploaded the Google Drive credentials file "
            f"to the expected directory {gdrive_credentials_dir}.\n"
            f"🔍 Checked path: {os.path.abspath(credentials_json)}"
        )
    else:
        print(f"✅ Found credentials: {credentials_json}")

    with open(credentials_json, "r") as file:
        creds_data = json.load(file)

    if creds_data["installed"]["redirect_uris"] == ["urn:ietf:wg:oauth:2.0:oob"]:
        return
    else:
        # Update redirect URIs
        creds_data["installed"]["redirect_uris"] = ["urn:ietf:wg:oauth:2.0:oob"]

        with open(credentials_json, "w") as file:
            json.dump(creds_data, file, indent=4)

        print(f"Updated redirect URIs in {credentials_json} to support out-of-band authentication.")


def authenticate(
    gdrive_credentials_dir: str = "/home/jovyan/Supplementary_data/DriveCredentials",
) -> GoogleAuth:
    """
    Authorize and authenticate and return Google Drive Authentication object.

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
    GoogleAuth
        Google OAuth2 object for accessing Google Drive.
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

    # Write pydrive2 settings file and update redirect URIs in credentials.json
    settings_yaml = write_settings_file(gdrive_credentials_dir)
    update_redirect_uris(gdrive_credentials_dir)

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    with open(settings_yaml, "r") as f:
        settings = yaml.safe_load(f)
    token_json = settings["save_credentials_file"]

    gauth = GoogleAuth(settings_file=settings_yaml)
    if os.path.exists(token_json):
        print(f"✅ Found existing access token: {token_json}")
        gauth.LoadCredentialsFile()
    else:
        gauth.CommandLineAuth()
    return gauth
