from datetime import datetime
from fastapi import Body, FastAPI, Query, HTTPException, Path
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Union
import math
import os
from sqlalchemy import Integer, create_engine, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase, Mapped, mapped_column

DATABASE_URL= os.getenv("DATABASE_URL", "sqlite:///./langs.db")
print("Conectado a ", DATABASE_URL)

engine_kwargs = {}

if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    **engine_kwargs
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(
    title="My API",
    description="This is a sample API built with FastAPI.",
)

class Tag(BaseModel):
    name: str = Field(..., max_length=30, description="Nombre de la etiqueta")

class LanguageBase(BaseModel):
    title: str
    content: str
    tags: Optional[List[Tag]] = Field(default_factory=list) # []

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
    tags: List[Tag] = Field(default_factory=list) # []

    @field_validator("title")
    @classmethod
    def not_allowed_title(cls, value: str) -> str:
        if "xxx" in value.lower():
            raise ValueError("El titulo no puede ser 'xxx'")
        return value

class LanguageUpdate(LanguageBase):
    content: Optional[str] = None
    title: Optional[str] = Field(None, min_length=3, max_length=100)

class LanguagePublic(LanguageBase):
    id: int

class LanguageSummary(BaseModel):
    id: int
    title: str

class PaginatedItem(BaseModel):
    page: int
    per_page: int
    total: int
    total_pages: int
    has_prev: bool
    has_next: bool
    order_by: Literal["id","titlle"]
    direction: Literal["asc", "desc"]
    search: Optional[str]
    items: List[LanguagePublic]

class LanguageORM(Base):
    __tablename__ = "languages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False),
    create_at = Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine) # just in dev

LANGUAGES = [
    {"id": 1, "title": "Python", "content": "Python es un lenguaje dinámico y fácil de aprender, con una amplia biblioteca estándar. Muy usado en desarrollo web, scripting, ciencia de datos y automatización."},
    {"id": 2, "title": "JavaScript", "content": "JavaScript es el lenguaje de la web: orientado a eventos, se ejecuta en navegadores y en Node.js. Ideal para interfaces interactivas y aplicaciones full-stack."},
    {"id": 3, "title": "Java", "content": "Java es un lenguaje estáticamente tipado que corre sobre la JVM. Popular en aplicaciones empresariales por su rendimiento, portabilidad y robustez."},
    {"id": 4, "title": "C#", "content": "C# es un lenguaje moderno de Microsoft, orientado a objetos, con buen soporte para desarrollo de aplicaciones de escritorio, web y juegos (Unity). Tipado estático y potente ecosistema .NET."},
    {"id": 5, "title": "PHP", "content": "PHP es un lenguaje de scripting ampliamente usado para desarrollo web del lado del servidor. Fácil de desplegar, con muchas aplicaciones y frameworks como Laravel y Symfony."},
    {"id": 6, "title": "Go", "content": "Go (Golang) es un lenguaje compilado desarrollado por Google, conocido por su simplicidad, concurrencia nativa (goroutines) y alto rendimiento en servicios y microservicios."},
    {"id": 7, "title": "C++", "content": "C++ es un lenguaje de propósito general que combina programación de bajo nivel y abstracciones de alto nivel. Muy usado en sistemas, software de alto rendimiento y motores de juego."},
    {"id": 8, "title": "Ruby", "content": "Ruby es un lenguaje dinámico y reflexivo, enfocado en la simplicidad y productividad. Famoso por su framework Ruby on Rails para desarrollo web rápido."},
    {"id": 9, "title": "Swift", "content": "Swift es un lenguaje potente e intuitivo creado por Apple para desarrollar apps de iOS, macOS, watchOS y tvOS. Es seguro, rápido y moderno."},
    {"id": 10, "title": "Kotlin", "content": "Kotlin es un lenguaje moderno que corre en la JVM, totalmente interoperable con Java. Es el lenguaje preferido por Google para el desarrollo de aplicaciones Android."},
    {"id": 11, "title": "Rust", "content": "Rust es un lenguaje de sistemas enfocado en la seguridad y el rendimiento. Garantiza la seguridad de memoria sin recolector de basura, ideal para software crítico."},
    {"id": 12, "title": "TypeScript", "content": "TypeScript es un superconjunto de JavaScript que añade tipado estático opcional. Mejora la mantenibilidad y escalabilidad de grandes proyectos web."},
    {"id": 13, "title": "Perl", "content": "Perl es un lenguaje de alto nivel, interpretado y dinámico. Históricamente fuerte en procesamiento de texto y administración de sistemas."},
    {"id": 14, "title": "Scala", "content": "Scala combina programación orientada a objetos y funcional. Corre en la JVM y es muy usado en procesamiento de datos masivos (Spark)."},
    {"id": 15, "title": "R", "content": "R es un lenguaje y entorno de software para computación estadística y gráficos. Es el estándar en análisis de datos, estadística y bioinformática."},
    {"id": 16, "title": "Dart", "content": "Dart es un lenguaje optimizado para clientes, desarrollado por Google. Es la base del framework Flutter para crear aplicaciones móviles nativas multiplataforma."},
    {"id": 17, "title": "Lua", "content": "Lua es un lenguaje de scripting ligero, rápido y embebible. Muy popular en la industria de videojuegos (como en Roblox o World of Warcraft) para lógica de scripts."},
    {"id": 18, "title": "Haskell", "content": "Haskell es un lenguaje puramente funcional con tipado estático fuerte. Es conocido por su elegancia matemática y uso en investigación académica e industrial."},
    {"id": 19, "title": "Elixir", "content": "Elixir es un lenguaje funcional y concurrente construido sobre la máquina virtual de Erlang (BEAM). Ideal para sistemas distribuidos, tolerantes a fallos y de baja latencia."},
    {"id": 20, "title": "Matlab", "content": "Matlab es un entorno de computación numérica y lenguaje de programación. Ampliamente utilizado en ingeniería y ciencia para cálculos matriciales y simulación."}
]


@app.get("/")
def home():
    return {"message": "Welcome to My API! 2025"}

@app.get("/language", response_model=PaginatedItem)
def get_languages(
        text: Optional[str] | None = Query(
            default=None,
            deprecated=True,
            description="Search query for language(deprecated)",
        ),
        query: Optional[str] | None = Query(
            default=None,
            description="Search query for language",
            alias="search",
            min_length=3,
            max_length=20,
            pattern="^[a-zA-Z0-9]+$"
        ),
        limit: int = Query(
            default=5,
            ge=1,
            le=50,
            description="Numero maximo de resultados (1-50)"
        ),
        page: int = Query(
            default=1,
            ge=1,
            description="Pagina"
        ),
        order_by: Literal["id","titlle"] = Query(
            default="id", 
            description="Ordenar por: id/title"
        ),
        direction: Literal["asc", "desc"] = Query(
            default="asc"
        )
    ):
    results = LANGUAGES
    query = query or text
    if query:
        results = [language for language in results if query.lower() in language["title"].lower() or query.lower() in language["content"].lower()]
    
    results = sorted(results, key=lambda l: l[order_by], reverse=(direction == "desc"))

    total = len(results)
    offset = (page-1) * limit

    items = results[offset : offset + limit]
    total_pages = math.ceil(total / limit)

    return PaginatedItem(
        page=page,
        total=total,
        items=items,
        per_page=limit,
        order_by=order_by,
        direction=direction,
        total_pages=total_pages,
        search=query,
        has_prev= page > 1,
        has_next= page < total_pages
        )

@app.get("/language/by-tags", response_model=List[LanguagePublic])
def filter_by_tags(
    tags: List[str] = Query(
        ...,
        min_length=1,
        description="Una o mas tags"
    )
):
    tags_power = [tag.lower() for tag in tags]
    return [
        lang for lang in LANGUAGES if any( tag["name"].lower() in tags_power for tag in lang.get("tags", []))
    ]

@app.get("/language/{id}", response_model=Union[LanguagePublic, LanguageSummary])
def get_language(id: int = Path(
    ...,
    gt=0,
    title="ID del language",
    description="ID del language"
), with_content: bool = Query(default=True, description="Include content in the response")):
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

