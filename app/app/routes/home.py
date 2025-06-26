from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from auth.authenticate import authenticate_cookie, authenticate
from auth.hash_password import HashPassword
from database.config import get_settings
from typing import Dict

settings = get_settings()
home_route = APIRouter()
hash_password = HashPassword()
templates = Jinja2Templates(directory="view")


@home_route.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Главная страница приложения.

    Args:
        request (Request): Объект запроса FastAPI

    Returns:
        HTMLResponse: HTML страница с контекстом пользователя
    """
    token = request.cookies.get(settings.COOKIE_NAME)
    if not token:
        user = None
    else:
        try:
            user = await authenticate_cookie(token)
        except HTTPException as e:
            # Если токен невалидный (просрочен или ошибка верификации),
            # просто считаем пользователя неавторизованным
            if e.detail == "Token expired!" or "Invalid token":
                user = None
            else:
                raise  # Если другая ошибка - пробрасываем её

    context = {"user": user, "request": request}
    return templates.TemplateResponse("index.html", context)


@home_route.get("/predict", response_class=HTMLResponse)
async def index_private(
    request: Request, username: str = Depends(authenticate_cookie)
):
    """
    Приватная страница, доступная только авторизованным пользователям через cookie.

    Args:
        request (Request): Объект запроса FastAPI
        user (str): Информация о пользователе из cookie-аутентификации

    Returns:
        HTMLResponse: Приватная HTML страница
    """
    context = {"user": username, "request": request}
    return templates.TemplateResponse("predict.html", context)


@home_route.get(
    "/health",
    response_model=Dict[str, str],
    summary="Проверка работоспособности",
    description="Возвращает статус работоспособности сервиса",
)
async def health_check() -> Dict[str, str]:
    """
    Эндпоинт проверки работоспособности сервиса.

    Returns:
        Dict[str, str]: Сообщение о статусе работоспособности

    Raises:
        HTTPException: Если сервис недоступен
    """
    try:
        # Add actual health checks here
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service unavailable")
