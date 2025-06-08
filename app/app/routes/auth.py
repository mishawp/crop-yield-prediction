from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from sqlmodel import Session
from auth.authenticate import authenticate_cookie, authenticate
from auth.hash_password import HashPassword
from auth.jwt_handler import create_access_token
from database.database import get_session
from services.auth.loginform import LoginForm
from models.user import UserCreate
from services.crud.user import UserService
from database.config import get_settings
from typing import Annotated

# Получаем настройки приложения
settings = get_settings()
# Создаем экземпляр роутера
auth_route = APIRouter()
# Создаем экземпляр для хеширования паролей
hash_password = HashPassword()
# Инициализируем шаблонизатор Jinja2
templates = Jinja2Templates(directory="view")


def get_user_service(session: Session = Depends(get_session)) -> UserService:
    return UserService(session)


@auth_route.post("/token")
async def login_for_access_token(
    response: Response,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> dict[str, str]:
    """
    Создает access token для аутентифицированного пользователя.

    Args:
        response (Response): Объект HTTP-ответа для установки cookie
        form_data (OAuth2PasswordRequestForm): Данные формы с email и паролем
        session (Session): Сессия базы данных

    Returns:
        dict[str, str]: Словарь с токеном и типом токена

    Raises:
        HTTPException: 404 если пользователь не найден
        HTTPException: 401 если неверные учетные данные
    """
    # Проверяем существование пользователя по email
    user_exist = user_service.get_user_by_email(form_data.username)
    if user_exist is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User does not exist"
        )

    # Проверяем правильность пароля
    if hash_password.verify_hash(form_data.password, user_exist.password):
        # Создаем JWT токен
        access_token = create_access_token(user_exist.email)
        # Устанавливаем токен в cookie
        response.set_cookie(
            key=settings.COOKIE_NAME,
            value=f"Bearer {access_token}",
            httponly=True,
        )

        # Возвращаем токен в ответе
        return {settings.COOKIE_NAME: access_token, "token_type": "bearer"}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid details passed.",
    )


@auth_route.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    """
    Отображает страницу входа.

    Args:
        request (Request): Объект HTTP-запроса

    Returns:
        TemplateResponse: HTML-страница с формой входа
    """
    # Передаем объект запроса в шаблон
    context = {
        "request": request,
    }
    return templates.TemplateResponse("login.html", context)


@auth_route.post("/login", response_model=None)
async def login_post(
    request: Request,
    user_service: Annotated[UserService, Depends(get_user_service)],
):
    """
    Обрабатывает отправку формы входа.

    Args:
        request (Request): Объект HTTP-запроса
        user_service (UserService): UserService

    Returns:
        Response: Перенаправление на главную страницу при успехе
        или HTML-страница с ошибками при неудаче
    """
    # Создаем и валидируем форму входа
    form = LoginForm(request)
    await form.load_data()
    if not await form.is_valid():
        return {"errors": form.errors}

    try:
        # При успешной валидации перенаправляем на главную
        response = RedirectResponse("/", status.HTTP_302_FOUND)
        await login_for_access_token(
            response=response, form_data=form, user_service=user_service
        )
        form.__dict__.update(msg="Login Successful!")
        print("[green]Login successful!!!!")
        return response
    except HTTPException:
        # При ошибке аутентификации показываем сообщение об ошибке
        form.__dict__.update(msg="")
        form.__dict__.get("errors").append("Incorrect Email or Password")
        return {"errors": form.errors}


@auth_route.post("/signup", response_model=None)
async def signup_post(
    request: Request,
    user_service: Annotated[UserService, Depends(get_user_service)],
):
    """
    Обрабатывает отправку формы регистрации.

    Args:
        request (Request): Объект HTTP-запроса
        user_service (UserService): UserService

    Returns:
        Response: Перенаправление на главную страницу при успехе
        или HTML-страница с ошибками при неудаче
    """
    form = LoginForm(request)
    await form.load_data()
    if not await form.is_valid():
        return {"errors": form.errors}
    if user_service.get_user_by_email(form.username):
        return {"errors": ["Пользователь с таким email уже существует"]}

    user_service.create_user(
        UserCreate(email=form.username, password=form.password)
    )
    response = RedirectResponse("/", status.HTTP_302_FOUND)
    await login_for_access_token(
        response=response,
        form_data=form,
        user_service=user_service,
    )
    return response


@auth_route.get("/logout")
async def login_get() -> RedirectResponse:
    """
    Выполняет выход пользователя из системы.

    Returns:
        RedirectResponse: Перенаправление на главную страницу
        с удалением cookie авторизации
    """
    # Создаем редирект и удаляем cookie с токеном
    response = RedirectResponse(url="/")
    response.delete_cookie(settings.COOKIE_NAME)
    return response
