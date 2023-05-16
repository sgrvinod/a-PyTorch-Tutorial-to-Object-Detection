from google.oauth2.service_account import Credentials
from google.cloud import storage
import pandas as pd
import os

from credentials import get_creds

def download_images(images):
    #try with blob storage
    creds = get_creds()
    print(creds)
    gcs_client = storage.Client(credentials=creds)
    bucket_name = 'public-datasets-lila'
    bucket = gcs_client.bucket(bucket_name)

    folder_name = 'SSDataset'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    downloaded_files = set()
    downloaded_files_path = os.path.join(folder_name, 'downloaded_files.txt')
    if os.path.exists(downloaded_files_path):
        with open(downloaded_files_path, 'r') as f:
            downloaded_files = set(f.read().splitlines())

    # Download each file that hasn't already been downloaded
    for path in images['image_path_rel']:
        if path in downloaded_files:
            continue

        # Download the file from GCS
        blob_name = f'snapshotserengeti-unzipped/{path}'
        blob = bucket.blob(blob_name)
        file_path = os.path.join(folder_name, images.loc[i, 'capture_id'] + '.jpg')
        blob.download_to_filename(file_path)

        # Record that the file has been downloaded
        downloaded_files.add(path)
        with open(downloaded_files_path, 'a') as f:
            f.write(f'{path}\n')
        print(f'Downloaded file {path}')
    

def main():
    print()
    print('Starting download')
    images = pd.read_csv('./SSDataset/images.csv')
    download_images(images)

if __name__ == '__main__':
    main()