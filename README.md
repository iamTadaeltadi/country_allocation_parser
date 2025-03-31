Run it using:

uvicorn main-gemini-two:app --reload (uses the first LLM)

uvicorn main-gemini-one:app --reload (uses the second LLM)

The respective models are:

model="google/gemini-2.0-flash-thinking-exp-01-21:free"

model="google/gemini-2.0-pro-exp-02-05:free"
