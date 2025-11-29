from fastapi import FastAPI, Query

app = FastAPI(
    title="My API",
    description="This is a sample API built with FastAPI.",
)
BLOG_POSTS = [
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

@app.get("/posts")
def get_posts(query: str | None = Query(default=None, description="Search query for blog posts")):
    if query:
        filtered_posts = [post for post in BLOG_POSTS if query.lower() in post["title"].lower() or query.lower() in post["content"].lower()]
        return {"data": filtered_posts, "query": query}
    else:
        return {"data": BLOG_POSTS, "query": query}

@app.get("/posts/{post_id}")
def get_post(post_id: int, with_content: bool = Query(default=True, description="Include content in the response")):
    for post in BLOG_POSTS:
        if post["id"] == post_id:
            if not with_content:
                return {"data": {"id": post["id"], "title": post["title"]}}
            return {"data": post}
    return {"error": "Post not found"}

