# Linear_Project

**Gen Alpha Roblox Skibidi rizz ranking tools right on our ways!**

---

## 🚀 Quick Start: JupyterLab Setup

Follow these steps to get up and running with JupyterLab for this project.

### Prerequisites

- **Python** 3.7 or higher
- **pip** (Python package manager)
- *(Recommended)* [virtualenv](https://virtualenv.pypa.io/) or [venv](https://docs.python.org/3/library/venv.html) for isolated Python environments

---

### Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/Nanach1ll/Linear_Project.git
   cd Linear_Project
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   # Using venv (cross-platform)
   python -m venv venv
   # On Unix/macOS
   source venv/bin/activate
   # On Windows
   venv\Scripts\activate
   ```

3. **Install all dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

### Run JupyterLab

After installing the requirements, start JupyterLab with:

```bash
jupyter lab
```

- This will open JupyterLab in your default web browser.
- If it does not open automatically, follow the link provided in the terminal (typically `http://localhost:8888`).

---

## Best Practices: Clear Notebook Output Before Committing

To keep this repository clean and lightweight, **always clear the output cells of Jupyter notebooks before pushing code**. This helps reduce file size, prevents accidental sharing of sensitive data, and keeps version history readable.

### How to Clear Output

- **From Jupyter Notebook/Lab:**  
  Click on `Cell` > `All Output` > `Clear` and then save the notebook.

- **From the Command Line:**  
  Run:
  ```bash
  jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace <your_notebook>.ipynb
  ```

- **From VS Code:**  
  Click the "Clear All Outputs" button in the notebook toolbar, then save.

### Why Do This?

- Prevents large diffs and merge conflicts caused by notebook outputs.
- Avoids pushing potentially sensitive or large data.
- Keeps repository size small and history clean.

> **Please clear outputs before every commit and push.**
