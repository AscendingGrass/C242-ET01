# C242-ET01

## Repository Setup Instructions

### 1. Clone the Repository
- Clone the repository to your local machine.

---

### 2. Open in VS Code
- Open the cloned repository in **Visual Studio Code (VS Code)**.

---

### 3. Set Up the Virtual Environment
1. Press **Ctrl + Shift + P** in VS Code.
2. Search for and select **Python: Create Environment**.
3. Choose **venv** as the environment type.
4. Select a Python interpreter (any version of your choice).
5. Use the `requirements-dev-win.txt` file to install the necessary dependencies if your os is windows. Use `requirements-dev.txt` if you're on linux.

---

### 4. Activate the Virtual Environment
1. Open the terminal in VS Code (**Ctrl + Shift + `**).
2. Activate the virtual environment by typing:  
   `.venv/Scripts/activate`
3. To deactivate the environment, type:  
   `deactivate`

---

### 5. Create a `.env` File
1. In the root folder of the repository, create a new file named `.env`.
2. Use the format from `C242-ET01/.env_format` as a template.
3. Fill in the `GENAI_API_KEY` with your API key from **Google AI Studio**.

---

### 6. Adding New Scripts
- Place any new scripts directly in the root folder of the repository.

---

### 7. Using the Wrapper
- Refer to the example script located at `C242-ET01/example_gemini.py` to learn how to use the wrapper.

---

## Our Webapps
[Dicoding Generative AI Web Application](https://github.com/RayaSatriatama/Dicoding-GenAI-WebApps/tree/main)
