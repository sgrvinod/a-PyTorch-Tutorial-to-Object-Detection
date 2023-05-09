import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

def get_creds():
    creds = None
    token_path = 'token.json'
    creds_path = 'creds.json'
    SCOPES = ['https://www.googleapis.com/auth/devstorage.read_only']

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_path, SCOPES)
            creds = flow.run_local_server(port=8091)
        
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    return creds