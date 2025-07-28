"""State tracking system for managing long-running operations in ABBA."""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class OperationStatus(str, Enum):
    """Status of an operation."""
    NEVER_STARTED = "never_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"  # Set when detecting incomplete in_progress on startup


@dataclass
class JobState:
    """State of a single job within an operation."""
    status: OperationStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    last_update: Optional[str] = None
    progress: Dict[str, Any] = None
    error_message: Optional[str] = None
    validation: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.progress is None:
            self.progress = {}
        if self.metadata is None:
            self.metadata = {}
        if self.validation is None:
            self.validation = {}


@dataclass
class OperationState:
    """State of an operation containing multiple jobs."""
    status: OperationStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_update: Optional[str] = None
    jobs: Dict[str, JobState] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.jobs is None:
            self.jobs = {}
        if self.metadata is None:
            self.metadata = {}


class StateTracker:
    """Tracks state of operations with automatic recovery from interruptions."""
    
    def __init__(self, state_file: Union[str, Path]):
        """Initialize state tracker.
        
        Args:
            state_file: Path to JSON file for storing state
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self._check_interrupted_operations()
    
    def _load_state(self) -> Dict[str, Dict[str, Any]]:
        """Load state from file."""
        if not self.state_file.exists():
            return {}
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                # Convert dict back to dataclass objects
                state = {}
                
                if 'operations' in data:
                    # New hierarchical format
                    state = {'operations': {}}
                    for op_name, op_data in data['operations'].items():
                        if isinstance(op_data, dict) and 'status' in op_data:
                            op_data['status'] = OperationStatus(op_data['status'])
                            
                            # Convert jobs
                            if 'jobs' in op_data:
                                jobs = {}
                                for job_name, job_data in op_data['jobs'].items():
                                    if isinstance(job_data, dict) and 'status' in job_data:
                                        job_data['status'] = OperationStatus(job_data['status'])
                                        jobs[job_name] = JobState(**job_data)
                                op_data['jobs'] = jobs
                            
                            state['operations'][op_name] = OperationState(**op_data)
                else:
                    # Old format - convert to new
                    state = self._migrate_old_format(data)
                
                return state
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading state file: {e}")
            return {'operations': {}}
    
    def _save_state(self):
        """Save state to file."""
        try:
            # Convert dataclass objects to dicts
            data = {'operations': {}}
            
            for op_name, op_state in self.state.get('operations', {}).items():
                if isinstance(op_state, OperationState):
                    op_dict = asdict(op_state)
                    # Convert nested JobState objects
                    if 'jobs' in op_dict:
                        for job_name, job_state in op_dict['jobs'].items():
                            if isinstance(job_state, JobState):
                                op_dict['jobs'][job_name] = asdict(job_state)
                    data['operations'][op_name] = op_dict
                else:
                    data['operations'][op_name] = op_state
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Error saving state file: {e}")
    
    def _check_interrupted_operations(self):
        """Check for operations and jobs that were in progress and mark as interrupted."""
        modified = False
        
        if 'operations' not in self.state:
            self.state = {'operations': {}}
            return
        
        for op_name, op_state in self.state['operations'].items():
            if isinstance(op_state, OperationState):
                # Check operation status
                if op_state.status == OperationStatus.IN_PROGRESS:
                    logger.warning(f"Found interrupted operation: {op_name}")
                    op_state.status = OperationStatus.INTERRUPTED
                    op_state.last_update = datetime.now().isoformat()
                    modified = True
                
                # Check individual jobs
                for job_name, job_state in op_state.jobs.items():
                    if isinstance(job_state, JobState):
                        if job_state.status == OperationStatus.IN_PROGRESS:
                            logger.warning(f"Found interrupted job: {op_name}/{job_name}")
                            job_state.status = OperationStatus.INTERRUPTED
                            job_state.last_update = datetime.now().isoformat()
                            modified = True
        
        if modified:
            self._save_state()
    
    def _migrate_old_format(self, old_data: Dict) -> Dict:
        """Migrate old format to new hierarchical format."""
        new_state = {'operations': {}}
        
        # Convert old category/operation structure to new format
        for category, operations in old_data.items():
            op_name = f"{category}_operations"
            new_state['operations'][op_name] = OperationState(
                status=OperationStatus.IN_PROGRESS,
                started_at=datetime.now().isoformat(),
                jobs={}
            )
            
            for job_name, job_data in operations.items():
                if isinstance(job_data, dict):
                    new_state['operations'][op_name].jobs[job_name] = JobState(
                        status=OperationStatus(job_data.get('status', 'never_started')),
                        progress=job_data
                    )
        
        return new_state
    
    def start_operation(
        self, 
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OperationState:
        """Start a new operation.
        
        Args:
            operation_name: Name of operation (e.g., 'import_translations')
            metadata: Optional metadata about the operation
            
        Returns:
            OperationState object
        """
        if 'operations' not in self.state:
            self.state = {'operations': {}}
        
        # Check if operation exists
        existing = self.get_operation_state(operation_name)
        if existing and existing.status == OperationStatus.INTERRUPTED:
            logger.info(f"Resuming interrupted operation: {operation_name}")
            existing.status = OperationStatus.IN_PROGRESS
            existing.last_update = datetime.now().isoformat()
            self._save_state()
            return existing
        
        op_state = OperationState(
            status=OperationStatus.IN_PROGRESS,
            started_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.state['operations'][operation_name] = op_state
        self._save_state()
        return op_state
    
    def start_job(
        self,
        operation_name: str,
        job_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> JobState:
        """Start a job within an operation.
        
        Args:
            operation_name: Name of parent operation
            job_name: Name of specific job (e.g., 'KJV', 'ESV')
            metadata: Optional metadata about the job
            
        Returns:
            JobState object
        """
        # Ensure operation exists
        op_state = self.get_operation_state(operation_name)
        if not op_state:
            self.start_operation(operation_name)
            op_state = self.get_operation_state(operation_name)
        
        # Check if job exists and is interrupted
        existing = op_state.jobs.get(job_name)
        if existing and existing.status == OperationStatus.INTERRUPTED:
            logger.info(f"Resuming interrupted job: {operation_name}/{job_name}")
            existing.status = OperationStatus.IN_PROGRESS
            existing.last_update = datetime.now().isoformat()
            self._save_state()
            return existing
        
        job_state = JobState(
            status=OperationStatus.IN_PROGRESS,
            started_at=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        op_state.jobs[job_name] = job_state
        op_state.last_update = datetime.now().isoformat()
        self._save_state()
        return job_state
    
    def update_job_progress(
        self,
        operation_name: str,
        job_name: str,
        progress: Dict[str, Any]
    ):
        """Update progress for a job.
        
        Args:
            operation_name: Name of operation
            job_name: Name of job
            progress: Progress data to store
        """
        op_state = self.get_operation_state(operation_name)
        if not op_state:
            logger.error(f"Cannot update progress for non-existent operation: {operation_name}")
            return
        
        job_state = op_state.jobs.get(job_name)
        if not job_state:
            logger.error(f"Cannot update progress for non-existent job: {operation_name}/{job_name}")
            return
        
        job_state.progress.update(progress)
        job_state.last_update = datetime.now().isoformat()
        op_state.last_update = datetime.now().isoformat()
        self._save_state()
    
    def complete_job(
        self,
        operation_name: str,
        job_name: str,
        validation: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Mark a job as completed.
        
        Args:
            operation_name: Name of operation
            job_name: Name of job
            validation: Validation results
            metadata: Optional completion metadata
        """
        op_state = self.get_operation_state(operation_name)
        if not op_state:
            logger.error(f"Cannot complete job for non-existent operation: {operation_name}")
            return
        
        job_state = op_state.jobs.get(job_name)
        if not job_state:
            logger.error(f"Cannot complete non-existent job: {operation_name}/{job_name}")
            return
        
        job_state.status = OperationStatus.COMPLETED
        job_state.completed_at = datetime.now().isoformat()
        job_state.last_update = datetime.now().isoformat()
        
        if validation:
            job_state.validation.update(validation)
        if metadata:
            job_state.metadata.update(metadata)
        
        # Check if all jobs are complete
        all_complete = all(
            j.status == OperationStatus.COMPLETED 
            for j in op_state.jobs.values()
        )
        
        if all_complete and op_state.jobs:
            op_state.status = OperationStatus.COMPLETED
            op_state.completed_at = datetime.now().isoformat()
        
        op_state.last_update = datetime.now().isoformat()
        self._save_state()
        logger.info(f"Job completed: {operation_name}/{job_name}")
    
    def fail_job(
        self,
        operation_name: str,
        job_name: str,
        error_message: str
    ):
        """Mark a job as failed.
        
        Args:
            operation_name: Name of operation
            job_name: Name of job
            error_message: Error message describing the failure
        """
        op_state = self.get_operation_state(operation_name)
        if not op_state:
            logger.error(f"Cannot fail job for non-existent operation: {operation_name}")
            return
        
        job_state = op_state.jobs.get(job_name)
        if not job_state:
            logger.error(f"Cannot fail non-existent job: {operation_name}/{job_name}")
            return
        
        job_state.status = OperationStatus.FAILED
        job_state.failed_at = datetime.now().isoformat()
        job_state.last_update = datetime.now().isoformat()
        job_state.error_message = error_message
        
        op_state.last_update = datetime.now().isoformat()
        self._save_state()
        logger.error(f"Job failed: {operation_name}/{job_name} - {error_message}")
    
    def get_operation_state(
        self,
        operation_name: str
    ) -> Optional[OperationState]:
        """Get the state of an operation.
        
        Args:
            operation_name: Name of operation
            
        Returns:
            OperationState if exists, None otherwise
        """
        return self.state.get('operations', {}).get(operation_name)
    
    def get_job_state(
        self,
        operation_name: str,
        job_name: str
    ) -> Optional[JobState]:
        """Get the state of a job.
        
        Args:
            operation_name: Name of operation
            job_name: Name of job
            
        Returns:
            JobState if exists, None otherwise
        """
        op_state = self.get_operation_state(operation_name)
        if not op_state:
            return None
        
        return op_state.jobs.get(job_name)
    
    def should_cleanup_job(
        self,
        operation_name: str,
        job_name: str
    ) -> bool:
        """Check if a job needs cleanup before starting.
        
        Args:
            operation_name: Name of operation
            job_name: Name of job
            
        Returns:
            True if cleanup is needed
        """
        job_state = self.get_job_state(operation_name, job_name)
        if not job_state:
            return False
        
        return job_state.status in [
            OperationStatus.INTERRUPTED,
            OperationStatus.FAILED
        ]
    
    def get_interrupted_jobs(self) -> List[tuple[str, str]]:
        """Get list of all interrupted jobs.
        
        Returns:
            List of (operation_name, job_name) tuples
        """
        interrupted = []
        for op_name, op_state in self.state.get('operations', {}).items():
            if isinstance(op_state, OperationState):
                for job_name, job_state in op_state.jobs.items():
                    if isinstance(job_state, JobState):
                        if job_state.status == OperationStatus.INTERRUPTED:
                            interrupted.append((op_name, job_name))
        return interrupted
    
    def reset_job(
        self,
        operation_name: str,
        job_name: str
    ):
        """Reset a job to never_started state.
        
        Args:
            operation_name: Name of operation
            job_name: Name of job
        """
        op_state = self.get_operation_state(operation_name)
        if op_state and job_name in op_state.jobs:
            del op_state.jobs[job_name]
            op_state.last_update = datetime.now().isoformat()
            self._save_state()
            logger.info(f"Reset job: {operation_name}/{job_name}")
    
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all operations and their jobs.
        
        Returns:
            Dictionary with operation and job statuses
        """
        summary = {'operations': {}}
        
        for op_name, op_state in self.state.get('operations', {}).items():
            if isinstance(op_state, OperationState):
                op_summary = {
                    'status': op_state.status.value,
                    'jobs': {}
                }
                
                for job_name, job_state in op_state.jobs.items():
                    if isinstance(job_state, JobState):
                        op_summary['jobs'][job_name] = {
                            'status': job_state.status.value
                        }
                        if job_state.validation:
                            op_summary['jobs'][job_name]['validation'] = job_state.validation
                
                summary['operations'][op_name] = op_summary
        
        return summary