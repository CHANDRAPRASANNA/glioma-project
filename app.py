import sys
import os

# Add the parent directory to the path so we can import glioma_project as a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from glioma_project.src import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
