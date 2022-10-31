__all__ = ('download_and_unzip', 'download_file_from_google_drive')

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import requests


def download_and_unzip(url, extract_to='.'):
  with urlopen(url) as http_response:
    with ZipFile(BytesIO(http_response.read())) as zipfile:
      zipfile.extractall(path=extract_to)


def download_file_from_google_drive(id, destination):
  def get_confirm_token(response):
    for key, value in response.cookies.items():
      if key.startswith('download_warning'):
        return value
    return None

  def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
      for chunk in response.iter_content(CHUNK_SIZE):
        if chunk:  # filter out keep-alive new chunks
          f.write(chunk)

  url = "https://docs.google.com/uc?export=download"
  session = requests.Session()
  response = session.get(url, params={'id': id}, stream=True)
  token = get_confirm_token(response)
  if token:
    params = {'id': id, 'confirm': token}
    response = session.get(url, params=params, stream=True)
  save_response_content(response, destination)
