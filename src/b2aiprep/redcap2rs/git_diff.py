import subprocess
import json
from git import Repo

repo = Repo("./")


def get_staged_files():
    try:
        # Run the git diff command to get staged files
        result = subprocess.run(['git', 'diff', '--name-only'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True)
        # Return the output, which is a list of file names
        return result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        return [f"Error: {e.stderr.strip()}"]
    except FileNotFoundError:
        return ["Error: Git is not installed or not found in the PATH."]


def is_version_only_change(file_path, repo):
    """Check if the only change in the file is the 'version' field."""
    try:
        with open(file_path, 'r') as f:
            working_content = json.load(f)

        last_commit_content = repo.git.show(f'HEAD:{file_path}')
        last_commit_content = json.loads(last_commit_content)

        working_version = working_content.pop("version", None)
        last_commit_version = last_commit_content.pop("version", None)

        return working_content == last_commit_content and working_version != last_commit_version
    except FileNotFoundError:
        return False



def checkout(file_path):
    try:
        # Run the git checkout command to restore a specific file
        result = subprocess.run(['git', 'checkout', '--', file_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"
    except FileNotFoundError:
        return "Error: Git is not installed or not found in the PATH."

if __name__ == "__main__":
    staged_files = get_staged_files()
    print("Staged files:")
    for file in staged_files:
        print(file)
        if(is_version_only_change(file, repo)):
            checkout(file)

