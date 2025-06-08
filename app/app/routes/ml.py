import logging
from pathlib import Path
from typing import Dict, List
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from sqlmodel import Session
from database.database import get_session
from models.mltask import MLTask, TaskStatus, MLTaskCreate, MLTaskUpdate
from services.rm.rm import rabbit_client
from services.crud.user import UserService
from services.logging.logging import get_logger
from services.crud.mltask import MLTaskService
from auth.authenticate import authenticate_cookie

logging.getLogger("pika").setLevel(logging.INFO)

logger = get_logger(logger_name=__name__)

ml_route = APIRouter()
UPLOAD_DIR = Path("files")
shared_queue = {}


def get_mltask_service(
    session: Session = Depends(get_session),
) -> MLTaskService:
    return MLTaskService(session)


def get_user_service(session: Session = Depends(get_session)) -> UserService:
    return UserService(session)


@ml_route.post(
    "/send_task",
    response_model=Dict[str, str | int],
    summary="ML endpoint",
    description="Send ml request with file",
)
async def send_task(
    file: UploadFile = File(...),
    username: str = Depends(authenticate_cookie),
    mltask_service: MLTaskService = Depends(get_mltask_service),
    user_service: UserService = Depends(get_user_service),
) -> Dict[str, str | int]:
    """
    Endpoint for sending ML task with file.

    Args:
        file: Uploaded file for processing
        user_id: User identifier
        mltask_service: ML task service dependency

    Returns:
        Dict[str, str]: Status message
    """
    logger.info("Request accepted")
    user = user_service.get_user_by_email(username)
    created_task = None
    try:
        # Create uploads directory if it doesn't exist

        # Create task with file information
        mltask = MLTaskCreate(
            # Using filename as question or adjust as needed
            user_id=user.id,
            status=TaskStatus.NEW,
        )
        created_task = mltask_service.create(mltask)
        upload_file = UPLOAD_DIR / f"{created_task.id}-{file.filename}"
        created_task = mltask_service.set_file(
            created_task.id,
            upload_file.name,
        )
        # Save the uploaded file
        with open(upload_file, "wb") as buffer:
            buffer.write(await file.read())

        logger.info(f"Task has been created: {created_task}")

        logger.info(f"Sending task to RabbitMQ with file: {upload_file.name}")
        rabbit_client.send_task(created_task)
        mltask_service.set_status(created_task.id, TaskStatus.QUEUED)
        shared_queue[created_task.id] = TaskStatus.QUEUED

        return {
            "task_id": created_task.id,
            "message": f"Task with file {file.filename} sent successfully!",
        }

    except Exception as e:
        if created_task:
            mltask_service.set_status(created_task.id, TaskStatus.FAILED)
        logger.error(f"Unexpected error in sending task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@ml_route.get("/receive_task_result", response_model=Dict[str, str | int])
async def receive_task_result(
    task_id: int,
    mltask_service: MLTaskService = Depends(get_mltask_service),
):
    task_status = shared_queue[task_id]
    response = {"status": task_status}
    if task_status == TaskStatus.COMPLETED:
        completed_task = mltask_service.get(task_id)
        response["result"] = completed_task.result
    else:
        shared_queue[task_id] = TaskStatus.PROCESSING
    return response


@ml_route.post("/send_task_result", response_model=Dict[str, str | int])
def send_task_result(
    task_id: int,
    result: str,
    mltask_service: MLTaskService = Depends(get_mltask_service),
) -> Dict[str, str | int]:
    """
    Endpoint for sending ML task using Result.

    Args:
        message (str): The message to be sent.
        user_id (int): ID of the user creating the task.

    Returns:
        Dict[str, str]: Response message with original and processed text.
    """
    try:
        mltask_service.set_result(task_id, result)
        shared_queue[task_id] = TaskStatus.COMPLETED
        logger.info(f"!!!!!!!!Task result has been set: {result}")
        return {"message": "Task result sent successfully!"}
    except Exception as e:
        logger.error(f"Unexpected error in sending task result: {str(e)}")
        shared_queue[task_id] = TaskStatus.FAILED
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        )


@ml_route.get("/tasks", response_model=List[MLTask])
async def get_all_tasks(
    username: str = Depends(authenticate_cookie),
    mltask_service: MLTaskService = Depends(get_mltask_service),
    user_service: UserService = Depends(get_user_service),
):
    """Get all ML tasks for user"""
    return mltask_service.get_all()


@ml_route.get("/tasks/{task_id}", response_model=MLTask)
async def get_task(
    task_id: int, mltask_service: MLTaskService = Depends(get_mltask_service)
):
    """Get ML task by ID."""
    task = mltask_service.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
