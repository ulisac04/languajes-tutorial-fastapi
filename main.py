from fastapi import Body, FastAPI, Query, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union

app = FastAPI(
    title="My API",
    description="This is a sample API built with FastAPI.",
)

class Tag(BaseModel):
    name: str = Field(..., max_length=30, description="Nombre de la etiqueta")

class LanguageBase(BaseModel):
    title: str
    content: str
    tags: Optional[List[Tag]] = []

class LanguageCreate(BaseModel):
    title: str = Field(
        ...,
        min_length=3,
        max_length=20,
        description="Titulo minimo 3 maximo 20",
        examples=["C++", "Java"]
    )
    content: Optional[str] = Field(
        default="Contenido pendiente",
        min_length=10,
        description="Descripcion del lenguaje",
        examples=["C++ es de bajo nivel"]
    )
    tags: List[Tag] = []

    @field_validator("title")
    @classmethod
    def not_allowed_title(cls, value: str) -> str:
        if "xxx" in value.lower():
            raise ValueError("El titulo no puede ser 'xxx'")
        return value

class LanguageUpdate(LanguageBase):
    content: Optional[str] = None
    title: Optional[str] = None
    title: Optional[Tag] = None

class LanguagePublic(LanguageBase):
    id: int

class LanguageSummary(BaseModel):
    id: int
    title: str

LANGUAGES = [
    {"id": 1, "title": "Python", "content": "Python es un lenguaje dinámico y fácil de aprender, con una amplia biblioteca estándar. Muy usado en desarrollo web, scripting, ciencia de datos y automatización."},
    {"id": 2, "title": "JavaScript", "content": "JavaScript es el lenguaje de la web: orientado a eventos, se ejecuta en navegadores y en Node.js. Ideal para interfaces interactivas y aplicaciones full-stack."},
    {"id": 3, "title": "Java", "content": "Java es un lenguaje estáticamente tipado que corre sobre la JVM. Popular en aplicaciones empresariales por su rendimiento, portabilidad y robustez."},
    {"id": 4, "title": "C#", "content": "C# es un lenguaje moderno de Microsoft, orientado a objetos, con buen soporte para desarrollo de aplicaciones de escritorio, web y juegos (Unity). Tipado estático y potente ecosistema .NET."},
    {"id": 5, "title": "PHP", "content": "PHP es un lenguaje de scripting ampliamente usado para desarrollo web del lado del servidor. Fácil de desplegar, con muchas aplicaciones y frameworks como Laravel y Symfony."},
    {"id": 6, "title": "Go", "content": "Go (Golang) es un lenguaje compilado desarrollado por Google, conocido por su simplicidad, concurrencia nativa (goroutines) y alto rendimiento en servicios y microservicios."},
    {"id": 7, "title": "C++", "content": "C++ es un lenguaje de propósito general que combina programación de bajo nivel y abstracciones de alto nivel. Muy usado en sistemas, software de alto rendimiento y motores de juego."}
]



@app.get("/")
def home():
    return {"message": "Welcome to My API! 2025"}

@app.get("/language", response_model=List[LanguagePublic])
def get_languages(query: str | None = Query(default=None, description="Search query for blog posts")):
    if query:
        return [language for language in LANGUAGES if query.lower() in language["title"].lower() or query.lower() in language["content"].lower()]
    else:
        return LANGUAGES

@app.get("/language/{id}", response_model=Union[LanguagePublic, LanguageSummary])
def get_language(id: int, with_content: bool = Query(default=True, description="Include content in the response")):
    for language in LANGUAGES:
        if language["id"] == id:
            if not with_content:
                return {"id": language["id"], "title": language["title"]}
            return language
    raise HTTPException(status_code=404, detail="no se encontro el language")


@app.post("/language", response_model=LanguagePublic, response_description="Item creado(OK)")
def create_language(language: LanguageCreate):
    new_id = (LANGUAGES[-1]["id"]+1) if LANGUAGES else 1
    new_language = {"id": new_id, "title": language.title, "content": language.content, "tags": [tag.model_dump() for tag in language.tags]}
    LANGUAGES.append(new_language)

    return new_language


@app.put("/language/{id}", response_model=LanguagePublic, response_model_exclude_none=True)
def update_language(id: int, data: LanguageUpdate):
    for language in LANGUAGES:
        if language["id"] == id:
            payload = data.model_dump(exclude_unset=True)
            if "title" in payload:
                language["title"] = payload["title"]
            if "content" in payload:
                language["content"] = payload["content"]
            if "tags" in payload:
                if "tags" not in language:
                    language["tags"] = []
                if len(payload["tags"]) == 0: 
                    language["tags"] = []
                else:
                    language["tags"].extend(t for t in payload["tags"])
            return language
        
    raise HTTPException(status_code=404, detail="no se encontro el language")


@app.delete("/language/{id}", status_code=204)
def delete_language(id: int):
    for index, language in enumerate(LANGUAGES):
        if language["id"] == id:
            LANGUAGES.pop(index)
            return
    raise HTTPException(status_code=404, detail="no se encontro el language")