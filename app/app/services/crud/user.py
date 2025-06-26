from auth.hash_password import HashPassword
from models.user import User, UserCreate
from models.mltask import MLTask
from sqlmodel import Session, select
from sqlalchemy.orm import selectinload
from typing import List, Optional


class UserService:
    hash_password = HashPassword()

    def __init__(self, session: Session):
        self.session = session

    def get_all_users(self) -> List[User]:
        """
        Получить всех пользователей с их событиями.

        Возвращает:
            List[User]: Список всех пользователей
        """
        try:
            statement = select(User).options(
                selectinload(User.ml_tasks).selectinload(MLTask.creator)
            )
            users = self.session.exec(statement).all()
            return users
        except Exception as e:
            raise

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Получить пользователя по ID.

        Аргументы:
            user_id: ID пользователя для поиска

        Возвращает:
            Optional[User]: Найденный пользователь или None
        """
        try:
            statement = (
                select(User)
                .where(User.id == user_id)
                .options(selectinload(User.ml_tasks))
            )
            user = self.session.exec(statement).first()
            return user
        except Exception as e:
            raise

    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Получить пользователя по email.

        Аргументы:
            email: Email для поиска

        Возвращает:
            Optional[User]: Найденный пользователь или None
        """
        try:
            statement = (
                select(User)
                .where(User.email == email)
                .options(selectinload(User.ml_tasks))
            )
            user = self.session.exec(statement).first()
            return user
        except Exception as e:
            raise

    def create_user(self, user: UserCreate) -> User:
        """
        Создать нового пользователя.

        Аргументы:
            user: DTO для создания пользователя

        Возвращает:
            User: Созданный пользователь с ID
        """
        user.password = self.hash_password.create_hash(user.password)
        db_user = User.from_orm(user)
        try:
            self.session.add(db_user)
            self.session.commit()
            self.session.refresh(db_user)
            return db_user
        except Exception as e:
            self.session.rollback()
            raise

    def delete_user(self, user_id: int) -> bool:
        """
        Удалить пользователя по ID.

        Аргументы:
            user_id: ID пользователя для удаления

        Возвращает:
            bool: True если удален, False если не найден
        """
        try:
            user = self.get_user_by_id(
                user_id,
            )
            if user:
                self.session.delete(user)
                self.session.commit()
                return True
            return False
        except Exception as e:
            self.session.rollback()
            raise
