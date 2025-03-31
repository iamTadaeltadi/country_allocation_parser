# Running the Application

## Usage

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

