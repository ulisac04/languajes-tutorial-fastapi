from datetime import datetime
from fastapi import Body, Depends, FastAPI, Query, HTTPException, Path, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Literal, Optional, Union
import math
import os
from sqlalchemy import Integer, create_engine, String, Text, DateTime, func, select, UniqueConstraint
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase, Mapped, mapped_column
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

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
    model_config = ConfigDict(from_attributes=True)

class LanguageSummary(BaseModel):
    id: int
    title: str

    model_config = ConfigDict(from_attributes=True)

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
    __table_args__ = (UniqueConstraint("title", name="unique_lang_title"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine) # just in dev

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
        ),
        db: Session = Depends(get_db)
    ):
    results = select(LanguageORM)
    query = query or text
    if query:
        results = results.where(LanguageORM.title.ilike(f"%{query}%"))

    # ordenar resultados según order_by y direction
    if order_by == "id":
        order_column = LanguageORM.id
    else:  # "titlle" -> columna title
        order_column = LanguageORM.title

    if direction == "desc":
        order_column = order_column.desc()
    else:
        order_column = order_column.asc()

    results = results.order_by(order_column)

    # total de registros para paginación
    total = db.scalar(select(func.count()).select_from(results.subquery())) or 0
    offset = (page - 1) * limit
    total_pages = math.ceil(total / limit) if total > 0 else 0

    # ejecutar consulta paginada y obtener lista de items
    items = db.execute(
        results.offset(offset).limit(limit)
    ).scalars().all()

    return PaginatedItem(
        page=page,
        total=total,
        items=items,
        per_page=limit,
        order_by=order_by,
        direction=direction,
        total_pages=total_pages,
        search=query,
        has_prev=page > 1,
        has_next=page < total_pages
    )

@app.post("/language", response_model=LanguagePublic, status_code=status.HTTP_201_CREATED)
def create_language(data: LanguageCreate, db: Session = Depends(get_db)):
    new_lang = LanguageORM(
        title=data.title,
        content=data.content or "Contenido pendiente",
    )

    try:
        db.add(new_lang)
        db.commit()
        db.refresh(new_lang)
        return LanguagePublic.model_validate(new_lang, from_attributes=True)
    except IntegrityError:
        db.rollback()
        raise IntegrityError(status_code=409, detail="el titulo ya existe")
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Error al crear el language")


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
    ]

@app.get("/language/{id}", response_model=Union[LanguagePublic, LanguageSummary])
def get_language(id: int = Path(
    ...,
    gt=0,
    title="ID del language",
    description="ID del language"
), with_content: bool = Query(default=True, description="Include content in the response"), db: Session = Depends(get_db)):
    
    lang_find = select(LanguageORM).where(LanguageORM.id == id)
    lang = db.execute(lang_find).scalar_one_or_none()

    if not lang:
        raise HTTPException(status_code=404, detail="no se encontro el language")
    
    if with_content:
        return LanguagePublic.model_validate(lang, from_attributes=True)

    return LanguageSummary.model_validate(lang, from_attributes=True)


@app.put("/language/{id}", response_model=LanguagePublic, response_model_exclude_none=True)
def update_language(id: int, data: LanguageUpdate, db: Session = Depends(get_db)):
    lang_update = db.scalar(select(LanguageORM).where(LanguageORM.id == id))
    if not lang_update:
        raise HTTPException(status_code=404, detail="no se encontro el language")

    update_data = data.model_dump(exclude_unset=True)

    if "title" in update_data:
        lang_update.title = update_data["title"]
    if "content" in update_data and update_data["content"] is not None:
        lang_update.content = update_data["content"]

    try:
        db.commit()
        db.refresh(lang_update)
        return LanguagePublic.model_validate(lang_update, from_attributes=True)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Error al actualizar")

@app.delete("/language/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_language(id: int, db: Session = Depends(get_db)):
    lang = db.scalar(select(LanguageORM).where(LanguageORM.id == id))
    if not lang:
        raise HTTPException(status_code=404, detail="no se encontro el language")
    try:
        db.delete(lang)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Error al eliminar")

    raise HTTPException(status_code=404, detail="no se encontro el language")


