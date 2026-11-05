import os
import json

def inspect_notebook():
    nb_path = r"c:\Users\Arogya Mary\Downloads\cse7thsem\cie3\forecasting-the-cell-growth-of-neuroblastoma-cells.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    code_cells = [c["source"] for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    
    with open("notebook_code_summary.txt", "w", encoding="utf-8") as out:
        for i, cell in enumerate(code_cells):
            out.write(f"=== CELL {i} ===\n")
            out.write("".join(cell))
            out.write("\n\n")

if __name__ == "__main__":
    inspect_notebook()
    print("Notebook extracted successfully to notebook_code_summary.txt")
