from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the htmlcov directory as a static directory at the path /coverage/
app.mount("/coverage/", StaticFiles(directory="htmlcov"), name="coverage")

@app.get("/")
def read_root():
    return {"message": "Go to /coverage/ to see the coverage report!"}


