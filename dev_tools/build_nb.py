import nbformat as nbf
import sys

def build(cells_spec, out_path):
    nb = nbf.v4.new_notebook()
    cells = []
    for kind, content in cells_spec:
        if kind == "md":
            cells.append(nbf.v4.new_markdown_cell(content))
        else:
            cells.append(nbf.v4.new_code_cell(content))
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python (churn)",
        "language": "python",
        "name": "churn_env",
    }
    with open(out_path, "w") as f:
        nbf.write(nb, f)
    print("written", out_path)
