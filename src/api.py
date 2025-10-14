from fastapi import FastAPI 
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from main import search_games, df
from typing import Optional

app = FastAPI(title="Game Search API")

class SearchRequest(BaseModel):
   query: str
   limit: Optional[int] = None
   mode: Optional[bool]

@app.get("/")
def root():
    return {"message": "Game Search API is running"}

@app.post("/search")
def search(request: SearchRequest):
    results = search_games(df, request.query, request.limit if request.limit else 10, request.mode if request.mode else True)
    return results.to_dict(orient="records")
 
app.mount("/search-game", StaticFiles(directory="static", html=True), name="static")