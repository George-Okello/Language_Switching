import nbformat

path = "Language_Switching.ipynb"  # replace with your file path
nb = nbformat.read(path, as_version=nbformat.NO_CONVERT)

if 'widgets' in nb['metadata']:
    del nb['metadata']['widgets']  # remove the problematic widgets metadata

nbformat.write(nb, path)
print("Notebook metadata cleaned.")
