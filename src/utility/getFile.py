import urllib.request
import os

def get_file(url: str, file_name: str) -> None:
    file_path = os.getcwd() + "\\data\\" + file_name
    print(file_path)
    try:
        with urllib.request.urlopen(url) as response:
            text_content = response.read().decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_content)
        print(f'{file_name} downloaded and saved successfully.')
    except Exception as e:
        print(f'An error occured: {e}')
