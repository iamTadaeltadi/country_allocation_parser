# Running the Application

## Usage

```pip install fastapi uvicorn python-dotenv openai pandas pdf2image pillow jinja2 xlsxwriter```
If you're on Ubuntu/macOS and encounter issues with pdf2image, you might also need poppler:

```sh
# macOS
brew install poppler

# Ubuntu
sudo apt install poppler-utils
```
Run the application using the following commands:

```sh
uvicorn main-gemini-two:app --reload
```
Uses the first LLM.

```sh
uvicorn main-gemini-one:app --reload
```
Uses the second LLM.

## Model Configuration

The respective models used are:

- `model="google/gemini-2.0-flash-thinking-exp-01-21:free"`
- `model="google/gemini-2.0-pro-exp-02-05:free"`

