import runpy
import sys
from pathlib import Path

def main():
    current_path = Path(__file__).parent

    streamlit_script_path = current_path.joinpath('app.py').resolve().as_posix()
    sys.argv = ["streamlit", "run", streamlit_script_path]
    runpy.run_module("streamlit", run_name="__main__")

if __name__ == "__main__":
    main()