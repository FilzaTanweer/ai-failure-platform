import pathlib
import requests

path = pathlib.Path('data/synthetic_project_data.csv')
with path.open('rb') as fh:
    files = {'file': ('synthetic_project_data.csv', fh, 'text/csv')}
    response = requests.post('http://127.0.0.1:8000/api/v1/predict/upload', files=files, timeout=600)

print(response.status_code)
print(response.text[:4000])
