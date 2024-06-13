import os
import subprocess

# Define the path to your notebooks and the path for the PDFs
notebooks_path = './'
pdfs_path = os.path.join(notebooks_path, 'pdfs')

# Create the 'pdfs' directory if it doesn't exist
if not os.path.exists(pdfs_path):
    os.makedirs(pdfs_path)

# List all Jupyter notebooks in the directory
notebooks = [f for f in os.listdir(notebooks_path) if f.endswith('.ipynb')]

# Convert each notebook to a PDF and save it in the 'pdfs' directory
for notebook in notebooks:
    notebook_path = os.path.join(notebooks_path, notebook)
    pdf_path = os.path.join(pdfs_path, notebook.replace('.ipynb', '.pdf'))
    
    print(f"Converting {notebook} to PDF...")
    
    # Use nbconvert to convert the notebook to PDF
    try:
        subprocess.run([
            'python', '-m', 'nbconvert', '--to', 'pdf', 
            notebook_path, '--output', pdf_path
        ], check=True)
        print(f"Successfully converted {notebook} to PDF.")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {notebook}: {e}")

print("Conversion complete.")