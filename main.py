from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Подключаем статические файлы (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Подключаем шаблоны Jinja2
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)