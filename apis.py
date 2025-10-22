import io
import os
import time
import uuid
import asyncio
import base64
from boson_multimodal.data_types import AudioContent, Message
from boson_multimodal.serve.serve_engine import ChatMLSample, HiggsAudioResponse, HiggsAudioServeEngine
import torchaudio
from transformers import Pipeline
import uvicorn
import aiohttp
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import heapq
from dataclasses import dataclass, field
from huggingface_hub import snapshot_download

from fastapi import FastAPI, HTTPException, Request
import torch
from PIL import Image, ImageFile
from fastapi.responses import StreamingResponse

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from schema import (
    APIError, 
    APIErrorResponse,
    TaskStatus,
    TaskInfo,
    TaskStatusResponse,
    Priority,
    QueueStats,
    FailureReason,
    AudioGenerationRequest,
    Model,
    VoiceType,
    WebhookTaskStatus,
    ModelsResponse,
    WebhookAudioTaskRequest,
    WebhookAudioTaskResponse,
    WebhookAudioPayload,
    CDNUploadResult,
    CancelTaskResponse
)

load_dotenv()

# Configuration
@dataclass
class Config:
    """Application configuration"""
    hf_token: str = os.getenv("HF_TOKEN", "")
    tts_model_repo: str = os.getenv("TTS_MODEL_REPO", "bosonai/higgs-audio-v2-generation-3B-base")
    audio_tokenizer_repo: str = os.getenv("AUDIO_TOKENIZER_REPO", "bosonai/higgs-audio-v2-tokenizer")
    local_dir: str = os.getenv("LOCAL_DIR", "checkpoints")
    max_queue_size: int = int(os.getenv("MAX_QUEUE_SIZE", "100"))
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "1"))
    sync_timeout: int = int(os.getenv("SYNC_TIMEOUT", "1800"))
    cleanup_interval: int = int(os.getenv("CLEANUP_INTERVAL", "600"))
    task_ttl: int = int(os.getenv("TASK_TTL", "7200"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    default_processing_time: int = int(os.getenv("DEFAULT_PROCESSING_TIME", "900"))
    
    # Webhook Configuration
    webhook_timeout: int = int(os.getenv("WEBHOOK_TIMEOUT", "30"))
    webhook_retry_attempts: int = int(os.getenv("WEBHOOK_RETRY_ATTEMPTS", "3"))
    webhook_retry_delay: int = int(os.getenv("WEBHOOK_RETRY_DELAY", "2"))
    webhook_progress_interval: int = int(os.getenv("WEBHOOK_PROGRESS_INTERVAL", "10"))
    
    # CDN Configuration
    cdn_upload_timeout: int = int(os.getenv("CDN_UPLOAD_TIMEOUT", "300"))
    cdn_chunk_size: int = int(os.getenv("CDN_CHUNK_SIZE", "1048576"))  # 1MB chunks

config = Config()


VOICE_PATHS = {
    VoiceType.MALE: {
        "audio_path": "sample_voices/male/puck.wav",
        "text_path": "sample_voices/male/puck.txt",
    },
    VoiceType.FEMALE: {
        "audio_path": "sample_voices/female/belinda.wav",
        "text_path": "sample_voices/female/belinda.txt",
    },
    VoiceType.LUMIRA: {
        "audio_path": "sample_voices/lumira/lumira.mp3",
        "text_path": "sample_voices/lumira/lumira.txt",
    },
}


DEFAULT_AVERAGE_PROCESSING_TIME = config.default_processing_time
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_API_KEY = os.getenv("WEBHOOK_API_KEY", "")
BACKEND_UPLOAD_URL = os.getenv("BACKEND_UPLOAD_URL", "")
BACKEND_UPLOAD_ADMIN_KEY = os.getenv("BACKEND_UPLOAD_ADMIN_KEY", "no-need")

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level.upper()))
logger = logging.getLogger(__name__)


@dataclass
class QueuedTask:
    """Represents a task in the priority queue"""
    task_id: str
    request: Union[WebhookAudioTaskRequest, AudioGenerationRequest]
    created_at: datetime
    priority: int = field(default=1)  # Lower number = higher priority
    is_webhook_task: bool = field(default=False)  # Flag to identify webhook tasks
    
    def __lt__(self, other):
        # First compare by priority, then by creation time for FIFO within same priority
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at

class ProductionQueueManager:
    """Production-ready queue manager with concurrency control"""
    
    def __init__(self, max_queue_size: int = 100, max_concurrent: int = 1):
        self.max_queue_size = max_queue_size
        self.max_concurrent = max_concurrent
        
        # Semaphore to control concurrent processing
        self.processing_semaphore = asyncio.Semaphore(max_concurrent)
        
        # Priority queue for tasks
        self.task_queue: list[QueuedTask] = []
        self.queue_lock = asyncio.Lock()
        
        # Task tracking
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_errors: Dict[str, APIErrorResponse] = {}
        
        # Statistics
        self.stats = {
            'total_queued': 0,
            'total_processed': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_dropped': 0,
            'processing_times': []
        }
        
        # Failure tracking
        self.failure_reasons: Dict[str, FailureReason] = {}
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.processor_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the queue manager"""
        logger.info("Starting production queue manager...")
        self.processor_task = asyncio.create_task(self._process_queue())
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_tasks())
        
    async def stop(self):
        """Stop the queue manager"""
        logger.info("Stopping production queue manager...")
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
                
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
    
    def _get_priority_value(self, priority: Priority) -> int:
        """Convert priority enum to numeric value"""
        priority_map = {
            Priority.HIGH: 0,
            Priority.NORMAL: 1,
            Priority.LOW: 2
        }
        return priority_map.get(priority, 1)
    
    async def add_task(self, task_id: str, request: Union[AudioGenerationRequest, WebhookAudioTaskRequest]) -> int:
        """Add a task to the queue and return queue position"""
        async with self.queue_lock:
            # Check if queue is full
            if len(self.task_queue) >= self.max_queue_size:
                raise HTTPException(
                    status_code=503, 
                    detail=f"Queue is full. Maximum {self.max_queue_size} tasks allowed."
                )
            
            # Determine if this is a webhook task
            is_webhook_task = isinstance(request, WebhookAudioTaskRequest)
            
            # Create queued task
            queued_task = QueuedTask(
                task_id=task_id,
                request=request,
                created_at=datetime.now(),
                priority=self._get_priority_value(request.priority or Priority.NORMAL),
                is_webhook_task=is_webhook_task
            )
            
            # Add to priority queue
            heapq.heappush(self.task_queue, queued_task)
            
            # Invalidate queue position cache
            if hasattr(self, '_cached_sorted_queue'):
                del self._cached_sorted_queue
            
            # Create task info
            self.tasks[task_id] = TaskInfo(
                task_id=task_id,
                status=TaskStatus.QUEUED,
                progress=0.0,
                created_at=datetime.now(),
                priority=request.priority or Priority.NORMAL,
                queue_position=self._get_queue_position(task_id),
                total_queue_size=len(self.task_queue)
            )
            
            self.stats['total_queued'] += 1
            logger.info(f"Added {'webhook ' if is_webhook_task else ''}task {task_id} to queue (priority: {request.priority})")
            
            return len(self.task_queue)
    
    def _get_queue_position(self, task_id: str) -> int:
        """Get the current position of a task in the queue"""
        # Since task_queue is already a heap (priority queue), we need to sort to get accurate positions
        # Cache the sorted queue to avoid repeated sorting for multiple position queries
        if not hasattr(self, '_cached_sorted_queue') or len(self._cached_sorted_queue) != len(self.task_queue):
            self._cached_sorted_queue = sorted(self.task_queue)
        
        for i, task in enumerate(self._cached_sorted_queue):
            if task.task_id == task_id:
                return i + 1
        return -1
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatusResponse]:
        """Get the status of a task"""
        task_info = self.tasks.get(task_id)
        if not task_info:
            return None
            
        # Update queue position if still queued
        if task_info.status == TaskStatus.QUEUED:
            task_info.queue_position = self._get_queue_position(task_id)
            task_info.total_queue_size = len(self.task_queue)
            # Estimate completion time
            if task_info.queue_position > 0:
                avg_time = self._get_average_processing_time()
                if avg_time:
                    task_info.estimated_completion = datetime.now() + timedelta(
                        seconds=avg_time * task_info.queue_position
                    )
        
        # Calculate dynamic progress for processing tasks
        if task_info.status == TaskStatus.PROCESSING:
            dynamic_progress = self._calculate_task_progress(task_info)
        else:
            dynamic_progress = task_info.progress
            
        return TaskStatusResponse(
            task_id=task_info.task_id,
            status=task_info.status,
            progress=dynamic_progress,
            created_at=task_info.created_at,
            started_at=task_info.started_at,
            completed_at=task_info.completed_at,
            queue_position=task_info.queue_position,
            total_queue_size=task_info.total_queue_size,
            estimated_completion=task_info.estimated_completion,
            error=self.task_errors.get(task_id).error if task_id in self.task_errors else None
        )
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get the result of a completed task"""
        return self.task_results.get(task_id)
    
    def get_queue_stats(self) -> QueueStats:
        """Get current queue statistics"""
        # Use persistent stats for totals to avoid issues with task cleanup
        processing_count = sum(1 for task in self.tasks.values() if task.status == TaskStatus.PROCESSING)
        
        return QueueStats(
            total_queued=self.stats['total_queued'],
            total_processing=processing_count,
            total_completed=self.stats['total_completed'],
            total_failed=self.stats['total_failed'],
            total_dropped=self.stats['total_dropped'],
            queue_size=len(self.task_queue),
            max_queue_size=self.max_queue_size,
            average_processing_time=self._get_average_processing_time(),
            estimated_wait_time=self._get_estimated_wait_time(),
            failure_reasons=self.get_failure_reasons()
        )
    
    def _get_average_processing_time(self) -> Optional[float]:
        """Calculate average processing time"""
        if not self.stats['processing_times']:
            return DEFAULT_AVERAGE_PROCESSING_TIME
        return sum(self.stats['processing_times']) / len(self.stats['processing_times'])
    
    def _calculate_estimated_wait_time(self, queue_position: int) -> Optional[int]:
        """Calculate estimated wait time based on queue position and average processing time"""
        if not queue_position or queue_position <= 0:
            return None
        
        avg_processing_time = self._get_average_processing_time()
        if avg_processing_time:
            return int(queue_position * avg_processing_time)
        return None
    
    async def _send_queue_status_webhook(self, task_id: str, webhook_url: str, api_key: str, 
                                       queue_position: Optional[int] = None, 
                                       total_queue_size: Optional[int] = None) -> bool:
        """Send queue status webhook with consistent payload structure"""
        if not webhook_notifier:
            return False
        
        # Get current queue info if not provided
        if queue_position is None or total_queue_size is None:
            task_info = await self.get_task_status(task_id)
            if not task_info:
                return False
            queue_position = task_info.queue_position
            total_queue_size = task_info.total_queue_size
        
        # Calculate estimated wait time
        estimated_wait_time = self._calculate_estimated_wait_time(queue_position)
        
        try:
            success = await webhook_notifier.send_status_webhook(
                webhook_url,
                task_id,
                TaskStatus.QUEUED,
                progress=0.0,
                queue_position=queue_position,
                total_queue_size=total_queue_size,
                estimated_wait_time=estimated_wait_time,
                webhook_api_key=api_key
            )
            return success
        except Exception as e:
            logger.error(f"Failed to send queue status webhook for task {task_id}: {e}")
            return False
    
    def _calculate_task_progress(self, task_info: TaskInfo) -> float:
        """Calculate task progress based on processing time vs average processing time"""
        if task_info.status != TaskStatus.PROCESSING or not task_info.started_at:
            return task_info.progress if task_info.progress else 0.0
            
        # Calculate how long the task has been processing
        processing_time = (datetime.now() - task_info.started_at).total_seconds()
        
        # Get average processing time
        avg_time = self._get_average_processing_time()
        
        # If we have manual progress updates, use them as a baseline
        manual_progress = task_info.progress if task_info.progress else 0.0
        
        if not avg_time or avg_time <= 0:
            # Fallback to a simple time-based progress if no average available
            # Assume tasks typically take 60-120 seconds, cap at 95%
            time_based_progress = min(processing_time / 90.0, 0.95)
            # Use the higher of manual progress or time-based progress
            return max(manual_progress, time_based_progress)
        
        # Calculate time-based progress: processing_time / average_processing_time
        time_based_progress = min(processing_time / avg_time, 0.98)  # Cap at 98% until actual completion
        
        # Use the higher of manual progress or time-based progress for real-time updates
        final_progress = max(manual_progress, time_based_progress)
        
        # Log progress updates for debugging
        if abs(final_progress - manual_progress) > 0.05:  # Log significant changes
            logger.debug(f"Task {task_info.task_id}: Progress update - Manual: {manual_progress:.2f}, Time-based: {time_based_progress:.2f}, Final: {final_progress:.2f} (Processing: {processing_time:.1f}s / Avg: {avg_time:.1f}s)")
        
        return final_progress
    
    def _get_estimated_wait_time(self) -> Optional[float]:
        """Estimate wait time for new tasks"""
        avg_time = self._get_average_processing_time()
        if avg_time and len(self.task_queue) > 0:
            return avg_time * len(self.task_queue)
        return None
    
    def _categorize_error(self, error: Exception) -> tuple[str, str]:
        """Categorize error and return (error_type, error_message)"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Categorize based on error content and type
        if "out of memory" in error_str or "cuda out of memory" in error_str:
            return "memory_error", f"GPU/CPU memory exhausted: {str(error)}"
        elif "timeout" in error_str or isinstance(error, asyncio.TimeoutError):
            return "timeout_error", f"Operation timed out: {str(error)}"
        elif "connection" in error_str or "network" in error_str:
            return "network_error", f"Network connectivity issue: {str(error)}"
        elif "permission" in error_str or "access" in error_str:
            return "permission_error", f"File/resource access denied: {str(error)}"
        elif "invalid" in error_str or "bad" in error_str or isinstance(error, ValueError):
            return "validation_error", f"Invalid input or parameters: {str(error)}"
        elif "pipeline" in error_str or "model" in error_str:
            return "pipeline_error", f"AI model/pipeline error: {str(error)}"
        elif isinstance(error, FileNotFoundError):
            return "file_error", f"Required file not found: {str(error)}"
        elif isinstance(error, ImportError) or isinstance(error, ModuleNotFoundError):
            return "dependency_error", f"Missing dependency: {str(error)}"
        else:
            return "unknown_error", f"Unexpected error ({error_type}): {str(error)}"
    
    def _track_failure(self, error: Exception):
        """Track failure reason for monitoring"""
        error_type, error_message = self._categorize_error(error)
        current_time = datetime.now()
        
        # Create a key for grouping similar errors
        error_key = f"{error_type}:{error_message[:100]}"  # Limit message length for grouping
        
        if error_key in self.failure_reasons:
            # Update existing failure reason
            failure_reason = self.failure_reasons[error_key]
            failure_reason.count += 1
            failure_reason.last_occurrence = current_time
        else:
            # Create new failure reason
            self.failure_reasons[error_key] = FailureReason(
                error_type=error_type,
                error_message=error_message,
                count=1,
                last_occurrence=current_time
            )
        
        logger.error(f"Tracked failure - Type: {error_type}, Message: {error_message}")
    
    def get_failure_reasons(self) -> List[FailureReason]:
        """Get list of failure reasons sorted by count (descending)"""
        return sorted(self.failure_reasons.values(), key=lambda x: x.count, reverse=True)
    
    def clear_failure_statistics(self):
        """Clear failure statistics (useful for maintenance/reset)"""
        self.failure_reasons.clear()
        logger.info("Cleared failure statistics")
    
    async def _process_queue(self):
        """Main queue processing loop"""
        logger.info("Starting queue processor...")
        
        while True:
            try:
                # Get next task
                queued_task = None
                async with self.queue_lock:
                    if self.task_queue:
                        queued_task = heapq.heappop(self.task_queue)
                        
                        # Invalidate queue position cache when removing tasks
                        if hasattr(self, '_cached_sorted_queue'):
                            del self._cached_sorted_queue
                
                if queued_task:
                    # Acquire semaphore to limit concurrent processing
                    async with self.processing_semaphore:
                        await self._process_task(queued_task)
                else:
                    # No tasks to process, wait a bit
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(5)

    async def cancel_task(self, task_id: str):
        """If the task is still in the queue, cancel it and remove it"""
        async with self.queue_lock:
            if task_id in self.tasks:
                # Cancel periodic queue webhook updates if they exist
                if hasattr(self, 'queue_webhook_tasks') and task_id in self.queue_webhook_tasks:
                    queue_task = self.queue_webhook_tasks[task_id]
                    queue_task.cancel()
                    try:
                        await queue_task
                    except asyncio.CancelledError:
                        pass
                    del self.queue_webhook_tasks[task_id]
                    logger.info(f"Cancelled periodic queue webhook updates for cancelled task {task_id}")
                
                # Remove the task from the queue entirely
                cancelled_task = self.tasks.pop(task_id)
                logger.info(f"Cancelled and removed task {task_id} from queue")
                return cancelled_task
            else:
                logger.info(f"Task {task_id} not found in queue")
                return None
    
    async def _process_task(self, queued_task: QueuedTask):
        """Process a single task"""
        task_id = queued_task.task_id
        start_time = time.time()
        
        try:
            # Check if task still exists (might have been cleaned up)
            task_info = self.tasks.get(task_id)
            if not task_info:
                logger.info(f"Task {task_id} was already cleaned up, skipping processing")
                return
            
            # Update task status
            task_info.status = TaskStatus.PROCESSING
            task_info.started_at = datetime.now()
            # Progress will be calculated dynamically based on processing time
            
            logger.info(f"Processing task {task_id} (priority: {queued_task.priority})")
            
            # Generate audio
            if queued_task.is_webhook_task:
                # For webhook tasks, generate audio and handle CDN upload + webhook
                await self._process_webhook_task(queued_task, task_info)
            
            task_info.status = TaskStatus.COMPLETED
            task_info.completed_at = datetime.now()
            task_info.progress = 1.0
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            if len(self.stats['processing_times']) > 100:  # Keep only last 100 times
                self.stats['processing_times'] = self.stats['processing_times'][-100:]
            
            self.stats['total_processed'] += 1
            self.stats['total_completed'] += 1
            
            logger.info(f"Completed task {task_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            
            # Track failure for monitoring
            self._track_failure(e)
            
            # Categorize error for better error response
            error_type, error_message = self._categorize_error(e)
            
            # Create error response
            error_response = APIErrorResponse(
                created=int(time.time()),
                error=APIError(
                    code=error_type,
                    message=error_message,
                    type="processing_error"
                )
            )
            
            self.task_errors[task_id] = error_response
            task_info = self.tasks[task_id]
            task_info.status = TaskStatus.FAILED
            task_info.completed_at = datetime.now()
            
            self.stats['total_processed'] += 1
            self.stats['total_failed'] += 1
    
    async def _process_webhook_task(self, queued_task: QueuedTask, task_info: TaskInfo) -> None:
        """Process a webhook task - generate audio, upload to CDN, and send webhook"""
        task_id = queued_task.task_id
        webhook_request = queued_task.request
        
        # Get webhook URL and API key from environment variables
        webhook_url = WEBHOOK_URL
        api_key = WEBHOOK_API_KEY
        
        if not webhook_url:
            logger.error(f"No webhook_url environment variable set for task {task_id}")
            raise ValueError("webhook_url environment variable is required")
        
        # Cancel periodic queue webhook updates if they exist
        if hasattr(self, 'queue_webhook_tasks') and task_id in self.queue_webhook_tasks:
            queue_task = self.queue_webhook_tasks[task_id]
            queue_task.cancel()
            try:
                await queue_task
            except asyncio.CancelledError:
                pass
            del self.queue_webhook_tasks[task_id]
            logger.info(f"Cancelled periodic queue webhook updates for task {task_id}")

        # Send initial processing webhook
        if webhook_notifier:
            await webhook_notifier.send_status_webhook(
                webhook_url,
                task_id,
                TaskStatus.PROCESSING,
                progress=0.01 * 100,
                webhook_api_key=api_key
            )
            
        # Start background task for periodic progress updates
        progress_task = asyncio.create_task(self._send_periodic_progress_webhooks(task_id, webhook_url, api_key))
        
        try:
            # Generate audio first
            logger.info(f"Generating audio for webhook task {task_id}")
            tmp_audio_path = f"temp_audio_{task_id}.wav"
            
            # Send progress webhook - audio generation started
            if webhook_notifier:
                await webhook_notifier.send_status_webhook(
                    webhook_url,
                    task_id,
                    TaskStatus.PROCESSING,
                    progress=0.05 * 100,
                    webhook_api_key=api_key
                )
            
            # Generate audio and save to temporary file
            await generate_audio_for_webhook(webhook_request, task_id, task_info, tmp_audio_path)
            
            # Send progress webhook - audio generation complete
            if webhook_notifier:
                await webhook_notifier.send_status_webhook(
                    webhook_url,
                    task_id,
                    TaskStatus.PROCESSING,
                    progress=0.95 * 100,
                    webhook_api_key=api_key
                )
            
            cdn_result = None
            audio_url = None
            
            # Upload to CDN if requested
            if webhook_request.cdn_upload and cdn_uploader:
                logger.info(f"Uploading audio to CDN for task {task_id}")
                task_info.progress = 0.9
                cdn_result = await cdn_uploader.upload_audio(tmp_audio_path, task_id)
                
                if cdn_result.success:
                    audio_url = cdn_result.cdn_url
                    logger.info(f"Audio uploaded to CDN: {audio_url}")
                else:
                    logger.error(f"CDN upload failed for task {task_id}: {cdn_result.error_message}")
            
            # Clean up temporary file
            try:
                os.remove(tmp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary audio file {tmp_audio_path}: {e}")
            
            # Send webhook notification
            if webhook_notifier:
                
                webhook_payload = WebhookAudioPayload(
                    request_id=task_id,
                    cdn_url=audio_url or "",
                    status=WebhookTaskStatus(
                        status=TaskStatus.COMPLETED,
                        error=None,
                        progress=None,
                        queue_position=None,
                        total_queue_size=None,
                        estimated_wait_time=None
                    )
                )
                
                logger.info(f"Sending webhook notification for task {task_id}")
                webhook_success = await webhook_notifier.send_webhook(
                    webhook_url,
                    webhook_payload,
                    api_key
                )
                
                if webhook_success:
                    logger.info(f"Webhook sent successfully for task {task_id}")
                else:
                    logger.error(f"Webhook failed for task {task_id}")
            
            # Cancel the periodic progress task
            if 'progress_task' in locals():
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            logger.error(f"Error processing webhook task {task_id}: {e}")
            
            # Cancel the periodic progress task on error
            if 'progress_task' in locals():
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel periodic queue webhook updates if they still exist (shouldn't happen but safety check)
            if hasattr(self, 'queue_webhook_tasks') and task_id in self.queue_webhook_tasks:
                queue_task = self.queue_webhook_tasks[task_id]
                queue_task.cancel()
                try:
                    await queue_task
                except asyncio.CancelledError:
                    pass
                del self.queue_webhook_tasks[task_id]
                logger.info(f"Cancelled remaining queue webhook updates for failed task {task_id}")
            
            # Send failure webhook
            if webhook_notifier:
                
                webhook_payload = WebhookAudioPayload(
                    request_id=task_id,
                    cdn_url="",
                    status=WebhookTaskStatus(
                        status=TaskStatus.FAILED,
                        error=str(e),
                        progress=None,
                        queue_position=None,
                        total_queue_size=None,
                        estimated_wait_time=None
                    )
                )
                
                await webhook_notifier.send_webhook(
                    webhook_url,
                    webhook_payload,
                    api_key
                )
            
            raise e
    
    async def _send_periodic_queue_webhooks(self, task_id: str, webhook_url: str, api_key: str):
        """Send periodic queue status updates via webhook while task is queued"""
        last_sent_position = None
        last_sent_total = None
        start_time = time.time()
        max_duration = config.task_ttl  # Maximum duration for webhook task
        
        try:
            while True:
                await asyncio.sleep(config.webhook_progress_interval)  # Use same interval as progress updates
                
                # Check if task still exists
                task_info = self.tasks.get(task_id)
                if not task_info:
                    logger.debug(f"Task {task_id} no longer exists, stopping periodic webhooks")
                    break
                
                # Check if task is no longer queued
                if task_info.status != TaskStatus.QUEUED:
                    break  # Task started processing
                
                # Check if webhook task has been running too long
                if time.time() - start_time > max_duration:
                    logger.warning(f"Periodic webhook task for {task_id} exceeded maximum duration ({max_duration}s)")
                    break
                
                # Update queue position and total queue size
                current_position = self._get_queue_position(task_id)
                current_total = len(self.task_queue)
                
                # Update task info with current queue stats
                task_info.queue_position = current_position
                task_info.total_queue_size = current_total
                
                # Send webhook only if position or total changed
                if (current_position != last_sent_position or current_total != last_sent_total):
                    # Use the centralized webhook method
                    webhook_success = await self._send_queue_status_webhook(
                        task_id, webhook_url, api_key, current_position, current_total
                    )
                    
                    if webhook_success:
                        last_sent_position = current_position
                        last_sent_total = current_total
                        logger.debug(f"Sent periodic queue webhook for task {task_id}: position {current_position}/{current_total}")
                    else:
                        logger.warning(f"Failed to send periodic queue webhook for task {task_id}")
                        
        except asyncio.CancelledError:
            logger.debug(f"Periodic queue webhook task cancelled for {task_id}")
        except Exception as e:
            logger.error(f"Error in periodic queue webhook task for {task_id}: {e}")

    async def _send_periodic_progress_webhooks(self, task_id: str, webhook_url: str, api_key: str):
        """Send periodic progress updates via webhook with configurable interval"""
        last_sent_progress = 0.0
        min_progress_delta = 0.05  # Only send if progress changed by at least 5%
        start_time = time.time()
        max_duration = config.task_ttl  # Maximum duration for webhook task
        
        try:
            while True:
                await asyncio.sleep(config.webhook_progress_interval)
                
                # Check if task still exists
                task_info = self.tasks.get(task_id)
                if not task_info:
                    logger.debug(f"Task {task_id} no longer exists, stopping periodic progress webhooks")
                    break
                
                # Check if task is no longer processing
                if task_info.status != TaskStatus.PROCESSING:
                    break  # Task completed or not found
                
                # Check if webhook task has been running too long
                if time.time() - start_time > max_duration:
                    logger.warning(f"Periodic progress webhook task for {task_id} exceeded maximum duration ({max_duration}s)")
                    break
                
                # Calculate real-time progress
                current_progress = self._calculate_task_progress(task_info)

                # Send webhook only if progress changed significantly
                progress_delta = current_progress - last_sent_progress
                if (webhook_notifier and current_progress > 0.1 and 
                    progress_delta >= min_progress_delta):
                    try:
                        await webhook_notifier.send_status_webhook(
                            webhook_url,
                            task_id,
                            TaskStatus.PROCESSING,
                            progress=current_progress * 100,
                            webhook_api_key=api_key
                        )
                        last_sent_progress = current_progress
                        logger.debug(f"Sent periodic progress webhook for task {task_id}: {current_progress:.2f}")
                    except Exception as e:
                        logger.warning(f"Failed to send periodic progress webhook for task {task_id}: {e}")
                        
        except asyncio.CancelledError:
            logger.debug(f"Periodic progress webhook task cancelled for {task_id}")
        except Exception as e:
            logger.error(f"Error in periodic progress webhook task for {task_id}: {e}")
    
    async def _cleanup_expired_tasks(self):
        """Clean up expired tasks periodically"""
        while True:
            try:
                current_time = datetime.now()
                expired_tasks = []
                
                # Find expired tasks (older than configured TTL)
                for task_id, task_info in self.tasks.items():
                    if (current_time - task_info.created_at).total_seconds() > config.task_ttl:
                        expired_tasks.append(task_id)
                
                # Clean up expired tasks
                async with self.queue_lock:
                    # Remove expired tasks from task_queue
                    self.task_queue = [task for task in self.task_queue if task.task_id not in expired_tasks]
                    heapq.heapify(self.task_queue)  # Rebuild heap after filtering
                
                for task_id in expired_tasks:
                    task_info = self.tasks.get(task_id)
                    if task_info:
                        age_seconds = (current_time - task_info.created_at).total_seconds()
                        logger.warning(f"Dropping expired task {task_id} after {age_seconds:.1f} seconds (TTL: {config.task_ttl}s)")
                    else:
                        logger.info(f"Cleaning up expired task {task_id}")
                    
                    # Cancel periodic webhook tasks for expired tasks
                    if hasattr(self, 'queue_webhook_tasks') and task_id in self.queue_webhook_tasks:
                        queue_task = self.queue_webhook_tasks[task_id]
                        queue_task.cancel()
                        try:
                            await queue_task
                        except asyncio.CancelledError:
                            pass
                        del self.queue_webhook_tasks[task_id]
                        logger.info(f"Cancelled periodic webhook updates for expired task {task_id}")
                    
                    self.tasks.pop(task_id, None)
                    self.task_results.pop(task_id, None)
                    self.task_errors.pop(task_id, None)
                    self.stats['total_dropped'] += 1
                
                # Log summary if tasks were dropped
                if expired_tasks:
                    logger.info(f"Dropped {len(expired_tasks)} expired tasks (total dropped: {self.stats['total_dropped']})")
                
                # Sleep for configured cleanup interval
                await asyncio.sleep(config.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)

# CDN Upload functionality
class CDNUploader:
    def __init__(self, config: Config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            # Create session with optimized connector settings
            connector = aiohttp.TCPConnector(
                limit=10,  # Connection pool size
                limit_per_host=5,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            timeout = aiohttp.ClientTimeout(
                total=self.config.cdn_upload_timeout,
                connect=30,
                sock_read=60
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
    
    async def close_session(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _stream_file_reader(self, file_path: str):
        """Stream file in chunks for memory-efficient upload"""
        chunk_size = self.config.cdn_chunk_size
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def upload_audio(self, audio_path: str, task_id: str) -> CDNUploadResult:
        """Upload audio to backend CDN service with streaming and optimized settings"""
        await self._ensure_session()
        
        try:
            # Get backend upload URL and admin key from environment
            backend_url = BACKEND_UPLOAD_URL
            admin_key = BACKEND_UPLOAD_ADMIN_KEY
            
            if not backend_url:
                logger.error("BACKEND_UPLOAD_URL not configured")
                return CDNUploadResult(success=False, cdn_url=None)
            
            # Get file size for progress tracking
            file_size = os.path.getsize(audio_path)
            logger.info(f"Starting CDN upload for task {task_id}, file size: {file_size} bytes")
            
            # Create streaming multipart form data
            data = aiohttp.FormData()
            data.add_field(
                'file', 
                self._stream_file_reader(audio_path),
                filename=os.path.basename(audio_path), 
                content_type='audio/wav'
            )
            
            async with self.session.post(
                f"{backend_url}?admin_key={admin_key}",
                data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("status") == 1:
                        logger.info(f"CDN upload successful for task {task_id}")
                        return CDNUploadResult(
                            success=True,
                            cdn_url=result.get("data")
                        )
                    else:
                        logger.error(f"Backend upload failed: {result.get('data')}")
                        return CDNUploadResult(
                            success=False,
                            cdn_url=None,
                            error_message=result.get('data', 'Unknown error')
                        )
                else:
                    response_text = await response.text()
                    logger.error(f"Backend upload HTTP error {response.status}: {response_text}")
                    return CDNUploadResult(
                        success=False,
                        cdn_url=None,
                        error_message=f"HTTP {response.status}: {response_text}"
                    )
                    
        except asyncio.TimeoutError:
            logger.error(f"CDN upload timeout for task {task_id}")
            return CDNUploadResult(
                success=False,
                cdn_url=None,
                error_message="Upload timeout"
            )
        except Exception as e:
            logger.error(f"Failed to upload audio to backend {audio_path}: {str(e)}")
            return CDNUploadResult(
                success=False,
                cdn_url=None,
                error_message=str(e)
            )


# Webhook functionality
class WebhookNotifier:
    """Handles webhook notifications with session reuse and exponential backoff"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def send_status_webhook(self, webhook_url: str, task_id: str, 
                                 status: TaskStatus, progress: Optional[float] = None,
                                 queue_position: Optional[int] = None, 
                                 total_queue_size: Optional[int] = None,
                                 estimated_wait_time: Optional[int] = None,
                                 error: Optional[str] = None,
                                 webhook_api_key: Optional[str] = None) -> bool:
        """Send status update webhook notification"""
        webhook_payload = WebhookAudioPayload(
            request_id=task_id,
            cdn_url="",  # No CDN URL for status updates
            status=WebhookTaskStatus(
                status=status,
                error=error,
                progress=progress,
                queue_position=queue_position,
                total_queue_size=total_queue_size,
                estimated_wait_time=estimated_wait_time
            )
        )
        
        return await self.send_webhook(webhook_url, webhook_payload, webhook_api_key)
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def send_webhook(self, webhook_url: str, payload: WebhookAudioPayload, 
                          webhook_api_key: Optional[str] = None) -> bool:
        """Send webhook notification with retries and exponential backoff"""
        await self._ensure_session()
        
        for attempt in range(self.config.webhook_retry_attempts):
            try:
                payload_json = payload.model_dump_json()                
                # Generate signature if secret provided
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {webhook_api_key}"
                }
                
                # Send HTTP request using reused session
                async with self.session.post(
                    webhook_url,
                    data=payload_json,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook sent successfully to {webhook_url} (attempt {attempt + 1})")
                        return True
                    else:
                        response_text = await response.text()
                        logger.warning(f"Webhook failed with status {response.status}: {response_text} (attempt {attempt + 1})")
                        
            except asyncio.TimeoutError:
                logger.error(f"Webhook timeout to {webhook_url} (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Webhook attempt {attempt + 1} to {webhook_url} failed: {e}")
                
            # Exponential backoff with jitter (except on last attempt)
            if attempt < self.config.webhook_retry_attempts - 1:
                backoff_delay = self.config.webhook_retry_delay * (2 ** attempt)
                jitter = backoff_delay * 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)
                total_delay = min(backoff_delay + jitter, 60)  # Cap at 60 seconds
                logger.debug(f"Webhook retry backoff: {total_delay:.2f}s")
                await asyncio.sleep(total_delay)
        
        logger.error(f"All webhook attempts failed to {webhook_url}")
        return False

# Global variables
tts_serve_engine: HiggsAudioServeEngine | None = None

queue_manager: Optional[ProductionQueueManager] = None
cdn_uploader: Optional[CDNUploader] = None
webhook_notifier: Optional[WebhookNotifier] = None

async def initialize_pipeline():
    """Initialize the audio generation pipeline"""
    global tts_serve_engine
    try:
        logger.info("Initializing Higgs Audio generation pipeline...")

        local_tts_model_path = os.path.join(config.local_dir, config.tts_model_repo)
        local_audio_tokenizer_path = os.path.join(config.local_dir, config.audio_tokenizer_repo)

        snapshot_download(config.tts_model_repo, local_dir=local_tts_model_path)
        snapshot_download(config.audio_tokenizer_repo, local_dir=local_audio_tokenizer_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_serve_engine = HiggsAudioServeEngine(local_tts_model_path, local_audio_tokenizer_path, device=device)

        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise e


async def generate_audio_for_webhook(request_data: WebhookAudioTaskRequest, task_id: str, task_info: TaskInfo, output_path: str) -> None:
    """Generate audio for webhook task and save directly to file - optimized version"""
    logger.info(f"Starting audio generation for webhook task {task_id}")

    def encode_base64_content_from_file(file_path: str) -> str:
        """Encode a content from a local file to base64 format."""
        # Read the file as binary and encode it directly to Base64
        with open(file_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
        return audio_base64

    def get_voice_clone_input_sample(
        voice_audio_path: str,
        voice_text_path: str,
        prompt: str
    ):
        with open(os.path.join(os.path.dirname(__file__), voice_text_path), "r") as f:
            reference_text = f.read()
        reference_audio = encode_base64_content_from_file(
            os.path.join(os.path.dirname(__file__), voice_audio_path)
        )
        messages = [
            Message(
                role="user",
                content=reference_text,
            ),
            Message(
                role="assistant",
                content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
            ),
            Message(
                role="user",
                content=prompt,
            ),
        ]
        return ChatMLSample(messages=messages)

    voice_path = VOICE_PATHS[request_data.voice_type]

    sample = get_voice_clone_input_sample(
        voice_audio_path=voice_path["audio_path"],
        voice_text_path=voice_path["text_path"],
        prompt=request_data.input_text
    )

    # Generate audio in thread pool to avoid blocking the event loop
    def _run_pipeline():
        try:
            output: HiggsAudioResponse = tts_serve_engine.generate(
                chat_ml_sample=sample,
                max_new_tokens=2048,
                temperature=0.3,
                top_p=0.95,
                top_k=50,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )

            return output
        except Exception as e:
            logger.error(f"Pipeline execution failed for task {task_id}: {e}")
            raise
    
    try:
        # Run the pipeline in a thread pool to keep the event loop responsive
        loop = asyncio.get_event_loop()
        output: HiggsAudioResponse = await loop.run_in_executor(None, _run_pipeline)

        torchaudio.save(output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        
        logger.info(f"Audio generation completed for webhook task {task_id}")
        
    finally:
        # Clean up resources in finally block to ensure cleanup even on exceptions
        try:
            if 'output' in locals():
                del output
            del sample
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup for task {task_id}: {cleanup_error}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global queue_manager, cdn_uploader, webhook_notifier
    
    # Startup
    logger.info("Starting up the audio generation API...")
    
    # Initialize pipeline
    await initialize_pipeline()
    
    # Initialize CDN uploader
    cdn_uploader = CDNUploader(config)
    logger.info("CDN uploader initialized")
    
    # Initialize webhook notifier
    webhook_notifier = WebhookNotifier(config)
    logger.info("Webhook notifier initialized")
    
    # Initialize queue manager
    queue_manager = ProductionQueueManager(
        max_queue_size=config.max_queue_size, 
        max_concurrent=config.max_concurrent
    )
    await queue_manager.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down the audio generation API...")
    if queue_manager:
        await queue_manager.stop()
    if webhook_notifier:
        await webhook_notifier.close_session()
    if cdn_uploader:
        await cdn_uploader.close_session()

# Create FastAPI app
app = FastAPI(
    title="Audio Generation API",
    description="A FastAPI-based audio generation API using Higgs Audio pipeline",
    version="1.0.0",
    lifespan=lifespan
)



@app.get("/v1/audios/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a audio generation task"""
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")
    
    task_status = await queue_manager.get_task_status(task_id)
    
    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_status

@app.get("/health", response_model=dict)
async def health_check() -> dict:
    return {"status": "ok"}

@app.get("/v1/queue/stats", response_model=QueueStats)
async def get_queue_stats() -> QueueStats:
    """Get the stats of the queue"""
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")
    return queue_manager.get_queue_stats()

@app.get("/v1/cancel/{task_id}", response_model=CancelTaskResponse)
async def cancel_task(task_id: str) -> CancelTaskResponse:
    """Cancel a task"""
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")
    queue_manager.cancel_task(task_id)
    return CancelTaskResponse(status="success", message="Task cancelled")

@app.get("/v1/queue/failures", response_model=List[FailureReason])
async def get_failure_details() -> List[FailureReason]:
    """Get detailed failure information for monitoring"""
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")
    return queue_manager.get_failure_reasons()

@app.post("/v1/queue/failures/clear", response_model=dict)
async def clear_failure_statistics() -> dict:
    """Clear failure statistics (maintenance endpoint)"""
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")
    queue_manager.clear_failure_statistics()
    return {"status": "success", "message": "Failure statistics cleared"}


@app.post("/v1/audios/generate", response_model=WebhookAudioPayload)
async def generate_audio(request: AudioGenerationRequest) -> WebhookAudioPayload:
    """Generate audio for a given text"""

    logger.info(f"Starting audio generation for task {request.request_id}")

    def encode_base64_content_from_file(file_path: str) -> str:
        """Encode a content from a local file to base64 format."""
        # Read the file as binary and encode it directly to Base64
        with open(file_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
        return audio_base64

    def get_voice_clone_input_sample(
        voice_audio_path: str,
        voice_text_path: str,
        prompt: str
    ):
        with open(os.path.join(os.path.dirname(__file__), voice_text_path), "r") as f:
            reference_text = f.read()
        reference_audio = encode_base64_content_from_file(
            os.path.join(os.path.dirname(__file__), voice_audio_path)
        )
        messages = [
            Message(
                role="user",
                content=reference_text,
            ),
            Message(
                role="assistant",
                content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
            ),
            Message(
                role="user",
                content=prompt,
            ),
        ]
        return ChatMLSample(messages=messages)

    voice_path = VOICE_PATHS[request.voice_type]

    sample = get_voice_clone_input_sample(
        voice_audio_path=voice_path["audio_path"],
        voice_text_path=voice_path["text_path"],
        prompt=request.input_text
    )

    # Generate audio in thread pool to avoid blocking the event loop
    def _run_pipeline():
        try:
            output: HiggsAudioResponse = tts_serve_engine.generate(
                chat_ml_sample=sample,
                max_new_tokens=2048,
                temperature=0.3,
                top_p=0.95,
                top_k=50,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )

            return output
        except Exception as e:
            logger.error(f"Pipeline execution failed for task {request.request_id}: {e}")
            raise
    
    try:
        # Run the pipeline in a thread pool to keep the event loop responsive
        loop = asyncio.get_event_loop()
        output: HiggsAudioResponse = await loop.run_in_executor(None, _run_pipeline)

        os.makedirs("output", exist_ok=True)
        torchaudio.save(os.path.join("output", f"temp_audio_{request.request_id}.wav"), torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        
        logger.info(f"Audio generation completed for task {request.request_id}")
        
    finally:
        # Clean up resources in finally block to ensure cleanup even on exceptions
        try:
            if 'output' in locals():
                del output
            del sample
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup for task {request.request_id}: {cleanup_error}")

    return WebhookAudioPayload(
        request_id=request.request_id,
        cdn_url=f"temp_audio_{request.request_id}.wav",
        status=WebhookTaskStatus(
            status=TaskStatus.COMPLETED,
            error=None,
            progress=None,
            queue_position=None,
            total_queue_size=None,
            estimated_wait_time=None
        )
    )

@app.post("/v1/audios/webhook", response_model=WebhookAudioTaskResponse)
async def create_webhook_audio_task(request: WebhookAudioTaskRequest) -> WebhookAudioTaskResponse:
    """Create a audio generation task with webhook notification"""

    if not queue_manager:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")
    
    # Generate task ID
    task_id = request.request_id if request.request_id else str(uuid.uuid4())

    # check if task id already exists
    if task_id in queue_manager.tasks:
        raise HTTPException(status_code=400, detail="Task ID already exists")
    
    try:
        # Add task to queue
        queue_position = await queue_manager.add_task(task_id, request)

        # Send initial queue status webhook notification and start periodic updates
        if webhook_notifier and WEBHOOK_URL:
            # Send initial queue status webhook using centralized method
            webhook_success = await queue_manager._send_queue_status_webhook(
                task_id, WEBHOOK_URL, WEBHOOK_API_KEY
            )
            
            if webhook_success:
                
                logger.info(f"Added webhook audio generation task {task_id} to queue at position {queue_position}")
                
                # Get task info for logging
                task_info = await queue_manager.get_task_status(task_id)
                if task_info:
                    logger.info(f"Queue status webhook sent successfully for task {task_id} (position: {task_info.queue_position}/{task_info.total_queue_size})")
                
                # Start periodic queue status updates
                queue_webhook_task = asyncio.create_task(
                    queue_manager._send_periodic_queue_webhooks(task_id, WEBHOOK_URL, WEBHOOK_API_KEY)
                )
                # Store the task reference so it can be cancelled later
                if not hasattr(queue_manager, 'queue_webhook_tasks'):
                    queue_manager.queue_webhook_tasks = {}
                queue_manager.queue_webhook_tasks[task_id] = queue_webhook_task
                logger.info(f"Started periodic queue webhook updates for task {task_id}")
            else:
                logger.warning(f"Failed to send queue status webhook for task {task_id}")
                
                return WebhookAudioTaskResponse(
                    task_id=task_id,
                    status=TaskStatus.FAILED
                )
        
        return WebhookAudioTaskResponse(
            task_id=task_id,
            status=TaskStatus.QUEUED
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating webhook task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """List available models (OpenAI compatible)"""
    model_card = Model(
        id=config.model_id,
        object="model",
        created=int(time.time()),
        owned_by="eternalai"
    )
    return ModelsResponse(object="list", data=[model_card])

if __name__ == "__main__":
    uvicorn.run("apis:app", host=config.host, port=config.port)
