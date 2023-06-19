import requests
import os
import nbformat
import gpt_2_simple as gpt2

def fetch_user_repositories(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    repositories = response.json()
    print(repositories)  # Print the response content for debugging
    return repositories if isinstance(repositories, list) else []


def preprocess_code(repository):
    if 'has_jupyter_notebook' in repository and repository['has_jupyter_notebook']:
        notebook_files = get_notebook_files(repository['name'], repository['html_url'])
        
        for file_path in notebook_files:
            preprocess_jupyter_notebook(file_path)
    
    file_paths = get_file_paths(repository['name'], repository['html_url'])
    
    for file_path in file_paths:
        preprocess_file(file_path)

def get_notebook_files(repository_name, repository_url):
    notebook_files = []
    api_url = f"{repository_url}/contents"
    response = requests.get(api_url)
    files = response.json()
    
    for file in files:
        if file['type'] == 'file' and file['name'].endswith('.ipynb'):
            notebook_files.append(file['path'])
    
    return notebook_files

def preprocess_jupyter_notebook(file_path):
    notebook = nbformat.read(file_path, nbformat.NO_CONVERT)
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code = cell['source']
            preprocess_code(code)

def get_file_paths(repository_name, repository_url):
    file_paths = []
    api_url = f"{repository_url}/contents"
    response = requests.get(api_url)
    files = response.json()
    
    for file in files:
        if file['type'] == 'file' and not file['name'].endswith('.ipynb'):
            file_paths.append(file['path'])
    
    return file_paths

def preprocess_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    
    preprocess_code(code)

def evaluate_technical_complexity(code):
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess)

    prompt = "Provide relevant repository information and ask GPT to assess technical complexity."
    input_text = prompt + code

    response = gpt2.generate(sess, run_name="run1", length=100, prefix=input_text, return_as_list=True)[0]

    return response

def select_most_complex_repository(repositories):
    max_complexity_score = float("-inf")
    most_complex_repository = None

    for repository in repositories:
        preprocess_code(repository)
        complexity_score = evaluate_technical_complexity(repository['code'])
        
        if complexity_score > max_complexity_score:
            max_complexity_score = complexity_score
            most_complex_repository = repository
    
    return most_complex_repository

github_username = input("Enter the GitHub username: ")

repositories = fetch_user_repositories(github_username)
most_complex_repository = select_most_complex_repository(repositories)

if most_complex_repository:
    print("Most Complex Repository:")
    print(f"Name: {most_complex_repository['name']}")
    print(f"URL: {most_complex_repository['html_url']}")
else:
    print("No repositories found for the given GitHub username.")
