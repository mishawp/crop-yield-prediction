from datetime import datetime, timezone
from enum import Enum
from typing import Optional, TYPE_CHECKING
from sqlmodel import SQLModel, Field, Relationship

if TYPE_CHECKING:
    from models.user import User


class TaskStatus(str, Enum):
    """Статусы выполнения ML задачи"""

    NEW = "new"  # Новая задача
    QUEUED = "queued"  # В очереди на выполнение
    PROCESSING = "processing"  # В процессе обработки
    COMPLETED = "completed"  # Выполнена
    FAILED = "failed"  # Ошибка выполнения


class MLTaskBase(SQLModel):
    """
    Базовая модель ML задачи.

    Атрибуты:
        status (TaskStatus): Текущий статус задачи
        result (Optional[str]): Результат обработки ML моделью
    """

    status: TaskStatus = Field(default=TaskStatus.NEW)
    result: Optional[str] = Field(default=None)
    file_path: Optional[str] = Field(default=None)


class MLTask(MLTaskBase, table=True):
    """
    Модель ML задачи для хранения в базе данных.

    Атрибуты:
        id (int): Уникальный идентификатор задачи
        event_id (int): ID связанного события
        user_id (int): ID пользователя, создавшего задачу
        created_at (datetime): Время создания задачи
        updated_at (datetime): Время последнего обновления
        creator (User): Связь с создателем
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    fips_code: int = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    creator: Optional["User"] = Relationship(
        back_populates="ml_tasks", sa_relationship_kwargs={"lazy": "selectin"}
    )

    def to_queue_message(self) -> dict:
        """Формирует сообщение для отправки в RabbitMQ"""
        return {
            "task_id": self.id,
            "question": self.file_path,
        }


class MLTaskCreate(MLTaskBase):
    """DTO для создания новой ML задачи"""

    user_id: int
    status: TaskStatus


class MLTaskUpdate(MLTaskBase):
    """DTO для обновления существующей ML задачи"""

    file_path: Optional[str] = None
    status: Optional[TaskStatus] = None
    result: Optional[str] = None
