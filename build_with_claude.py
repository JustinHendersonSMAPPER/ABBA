#!/usr/bin/env python3
"""
Robust Claude Code Project Monitor
Monitors project completion based on comprehensive criteria including test coverage,
functionality completeness, and implementation status.
"""
import os
import subprocess
import time
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import signal
import sys
from dataclasses import dataclass
from enum import Enum


class CompletionStatus(Enum):
    PASS = "✅"
    FAIL = "❌"
    PARTIAL = "⚠️"
    UNKNOWN = "❓"


@dataclass
class ProjectGoal:
    """Represents a project goal with completion status."""
    id: str
    description: str
    completed: bool = False
    evidence: Optional[str] = None


@dataclass
class Recommendation:
    """Represents a recommendation with implementation status."""
    id: str
    description: str
    implemented: bool = False
    evidence: Optional[str] = None


@dataclass
class CompletionCriteria:
    """Represents all completion criteria and their status."""
    test_coverage_above_90: bool = False
    unit_tests_cover_key_functionality: bool = False
    all_functionality_exists: bool = False
    integration_tests_exist: bool = False
    all_recommendations_implemented: bool = False
    
    coverage_percentage: float = 0.0
    missing_functionality: List[str] = None
    missing_tests: List[str] = None
    pending_recommendations: List[str] = None
    
    def __post_init__(self):
        if self.missing_functionality is None:
            self.missing_functionality = []
        if self.missing_tests is None:
            self.missing_tests = []
        if self.pending_recommendations is None:
            self.pending_recommendations = []
    
    def is_complete(self) -> bool:
        """Check if all criteria are met."""
        return all([
            self.test_coverage_above_90,
            self.unit_tests_cover_key_functionality,
            self.all_functionality_exists,
            self.integration_tests_exist,
            self.all_recommendations_implemented
        ])
    
    def get_completion_percentage(self) -> float:
        """Get overall completion percentage."""
        criteria = [
            self.test_coverage_above_90,
            self.unit_tests_cover_key_functionality,
            self.all_functionality_exists,
            self.integration_tests_exist,
            self.all_recommendations_implemented
        ]
        return (sum(criteria) / len(criteria)) * 100


class RobustClaudeCodeManager:
    def __init__(self, project_path: str = ".", check_interval: int = 300, 
                 auto_restart: bool = True, max_restarts: int = 10):
        self.project_path = Path(project_path).resolve()
        self.check_interval = check_interval
        self.claude_process: Optional[subprocess.Popen] = None
        self.running = True
        self.auto_restart = auto_restart
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.crash_timestamps = []
        self.project_config = self._load_project_config()
        
    def _load_project_config(self) -> Dict:
        """Load project configuration with goals and requirements."""
        config_files = [
            self.project_path / "project_completion.json",
            self.project_path / "project_completion.yml",
            self.project_path / "project_completion.yaml",
            self.project_path / ".project_config.json"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        if config_file.suffix == '.json':
                            return json.load(f)
                        else:
                            return yaml.safe_load(f)
                except Exception as e:
                    print(f"Warning: Could not load {config_file}: {e}")
        
        # Return default configuration
        return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """Create a default project configuration."""
        default_config = {
            "project_goals": [
                {
                    "id": "core_functionality",
                    "description": "Core functionality implemented according to requirements",
                    "validation_method": "manual_review"
                },
                {
                    "id": "error_handling",
                    "description": "Proper error handling implemented",
                    "validation_method": "code_analysis"
                }
            ],
            "key_functionality": [
                "main application logic",
                "configuration management",
                "error handling",
                "logging"
            ],
            "recommendations": [
                {
                    "id": "code_quality",
                    "description": "Code quality tools configured (linting, formatting)",
                    "validation_method": "config_check"
                },
                {
                    "id": "documentation",
                    "description": "README and API documentation complete",
                    "validation_method": "file_check"
                }
            ],
            "test_patterns": {
                "unit_test_dirs": ["tests/", "test/", "src/tests/"],
                "integration_test_dirs": ["tests/integration/", "integration_tests/"],
                "test_file_patterns": ["test_*.py", "*_test.py"]
            },
            "coverage_config": {
                "min_coverage": 90.0,
                "coverage_command": "coverage report --show-missing",
                "pytest_cov_command": "pytest --cov=src --cov-report=term-missing"
            }
        }
        
        # Save default config for user to customize
        config_path = self.project_path / "project_completion.json"
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created default configuration at {config_path}")
            print("Customize this file to define your project's specific completion criteria.")
        except Exception as e:
            print(f"Warning: Could not save default config: {e}")
            
        return default_config
    
    def check_test_coverage(self) -> Tuple[bool, float]:
        """Check if test coverage is above 90%."""
        coverage_commands = [
            self.project_config.get("coverage_config", {}).get("pytest_cov_command", "pytest --cov=src --cov-report=term-missing"),
            "poetry run pytest --cov=src --cov-report=term-missing",
            "python -m pytest --cov=src --cov-report=term-missing",
            "coverage report",
            "poetry run coverage report"
        ]
        
        for cmd in coverage_commands:
            try:
                result = subprocess.run(
                    cmd.split(),
                    cwd=self.project_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    # Parse coverage percentage from output
                    coverage_match = re.search(r'TOTAL.*?(\d+)%', result.stdout)
                    if coverage_match:
                        coverage = float(coverage_match.group(1))
                        min_coverage = self.project_config.get("coverage_config", {}).get("min_coverage", 90.0)
                        return coverage >= min_coverage, coverage
                        
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                print(f"Coverage command failed: {cmd} - {e}")
                continue
                
        return False, 0.0
    
    def check_unit_tests_exist(self) -> Tuple[bool, List[str]]:
        """Check if unit tests exist for key functionality."""
        test_patterns = self.project_config.get("test_patterns", {})
        test_dirs = test_patterns.get("unit_test_dirs", ["tests/", "test/"])
        test_file_patterns = test_patterns.get("test_file_patterns", ["test_*.py", "*_test.py"])
        
        key_functionality = self.project_config.get("key_functionality", [])
        missing_tests = []
        
        # Find all test files
        test_files = []
        for test_dir in test_dirs:
            test_path = self.project_path / test_dir
            if test_path.exists():
                for pattern in test_file_patterns:
                    test_files.extend(test_path.glob(f"**/{pattern}"))
        
        if not test_files:
            return False, key_functionality.copy()
        
        # Check if key functionality is tested
        all_test_content = ""
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    all_test_content += f.read().lower()
            except Exception:
                continue
        
        for functionality in key_functionality:
            # Simple heuristic: check if functionality name appears in test files
            if functionality.lower().replace(' ', '_') not in all_test_content:
                missing_tests.append(functionality)
        
        return len(missing_tests) == 0, missing_tests
    
    def check_functionality_exists(self) -> Tuple[bool, List[str]]:
        """Check if all required functionality exists."""
        project_goals = self.project_config.get("project_goals", [])
        missing_functionality = []
        
        # Get all Python files in the project
        python_files = list(self.project_path.glob("**/*.py"))
        if not python_files:
            return False, [goal.get("description", goal.get("id", "Unknown")) for goal in project_goals]
        
        all_code_content = ""
        for py_file in python_files:
            if "test" not in str(py_file).lower():  # Skip test files
                try:
                    with open(py_file, 'r') as f:
                        all_code_content += f.read()
                except Exception:
                    continue
        
        # Check each goal
        for goal in project_goals:
            goal_id = goal.get("id", "")
            description = goal.get("description", goal_id)
            validation_method = goal.get("validation_method", "manual_review")
            
            if validation_method == "code_analysis":
                # Simple heuristic: check if goal-related code exists
                keywords = goal_id.lower().split('_') + description.lower().split()
                if not any(keyword in all_code_content.lower() for keyword in keywords if len(keyword) > 3):
                    missing_functionality.append(description)
            elif validation_method == "file_check":
                # Check if required files exist
                required_files = goal.get("required_files", [])
                for required_file in required_files:
                    if not (self.project_path / required_file).exists():
                        missing_functionality.append(description)
                        break
            # For manual_review, assume it needs manual verification
            
        return len(missing_functionality) == 0, missing_functionality
    
    def check_integration_tests_exist(self) -> bool:
        """Check if integration tests exist."""
        test_patterns = self.project_config.get("test_patterns", {})
        integration_dirs = test_patterns.get("integration_test_dirs", ["tests/integration/", "integration_tests/"])
        test_file_patterns = test_patterns.get("test_file_patterns", ["test_*.py", "*_test.py"])
        
        for int_dir in integration_dirs:
            int_path = self.project_path / int_dir
            if int_path.exists():
                for pattern in test_file_patterns:
                    if list(int_path.glob(f"**/{pattern}")):
                        return True
        
        # Also check for integration test markers in regular test files
        test_dirs = test_patterns.get("unit_test_dirs", ["tests/", "test/"])
        for test_dir in test_dirs:
            test_path = self.project_path / test_dir
            if test_path.exists():
                for pattern in test_file_patterns:
                    for test_file in test_path.glob(f"**/{pattern}"):
                        try:
                            with open(test_file, 'r') as f:
                                content = f.read().lower()
                                if any(marker in content for marker in ["integration", "end_to_end", "e2e", "@pytest.mark.integration"]):
                                    return True
                        except Exception:
                            continue
        
        return False
    
    def check_recommendations_implemented(self) -> Tuple[bool, List[str]]:
        """Check if all recommendations are implemented."""
        recommendations = self.project_config.get("recommendations", [])
        pending_recommendations = []
        
        for rec in recommendations:
            rec_id = rec.get("id", "")
            description = rec.get("description", rec_id)
            validation_method = rec.get("validation_method", "manual_review")
            
            implemented = False
            
            if validation_method == "config_check":
                # Check if configuration files exist
                config_files = rec.get("config_files", [])
                if config_files:
                    implemented = all((self.project_path / cf).exists() for cf in config_files)
                else:
                    # Default config file checks
                    common_configs = [
                        "pyproject.toml", ".pre-commit-config.yaml", 
                        ".github/workflows/", "Makefile", "tox.ini"
                    ]
                    implemented = any((self.project_path / cf).exists() for cf in common_configs)
            elif validation_method == "file_check":
                # Check if required files exist
                required_files = rec.get("required_files", ["README.md", "docs/"])
                implemented = any((self.project_path / rf).exists() for rf in required_files)
            elif validation_method == "command_check":
                # Run a command to check implementation
                check_command = rec.get("check_command", "")
                if check_command:
                    try:
                        result = subprocess.run(
                            check_command.split(),
                            cwd=self.project_path,
                            capture_output=True,
                            timeout=30
                        )
                        implemented = result.returncode == 0
                    except Exception:
                        implemented = False
            
            if not implemented:
                pending_recommendations.append(description)
        
        return len(pending_recommendations) == 0, pending_recommendations
    
    def assess_completion_criteria(self) -> CompletionCriteria:
        """Assess all completion criteria."""
        print("🔍 Assessing project completion criteria...")
        
        criteria = CompletionCriteria()
        
        # Check test coverage
        print("📊 Checking test coverage...")
        criteria.test_coverage_above_90, criteria.coverage_percentage = self.check_test_coverage()
        
        # Check unit tests
        print("🧪 Checking unit test coverage...")
        criteria.unit_tests_cover_key_functionality, criteria.missing_tests = self.check_unit_tests_exist()
        
        # Check functionality
        print("⚙️ Checking functionality completeness...")
        criteria.all_functionality_exists, criteria.missing_functionality = self.check_functionality_exists()
        
        # Check integration tests
        print("🔗 Checking integration tests...")
        criteria.integration_tests_exist = self.check_integration_tests_exist()
        
        # Check recommendations
        print("✅ Checking recommendations implementation...")
        criteria.all_recommendations_implemented, criteria.pending_recommendations = self.check_recommendations_implemented()
        
        return criteria
    
    def display_status_report(self, criteria: CompletionCriteria):
        """Display a detailed status report."""
        print("\n" + "="*80)
        print("📋 PROJECT COMPLETION STATUS REPORT")
        print("="*80)
        
        # Overall status
        completion_pct = criteria.get_completion_percentage()
        print(f"🎯 Overall Completion: {completion_pct:.1f}%")
        print(f"🏁 Ready for Production: {'YES' if criteria.is_complete() else 'NO'}")
        print()
        
        # Individual criteria
        status_icon = lambda x: CompletionStatus.PASS.value if x else CompletionStatus.FAIL.value
        
        print("📋 COMPLETION CRITERIA:")
        print(f"  {status_icon(criteria.test_coverage_above_90)} Test Coverage > 90% ({criteria.coverage_percentage:.1f}%)")
        print(f"  {status_icon(criteria.unit_tests_cover_key_functionality)} Unit Tests Cover Key Functionality")
        print(f"  {status_icon(criteria.all_functionality_exists)} All Required Functionality Exists")
        print(f"  {status_icon(criteria.integration_tests_exist)} Integration Tests Exist")
        print(f"  {status_icon(criteria.all_recommendations_implemented)} All Recommendations Implemented")
        
        # Details for failed criteria
        if criteria.missing_tests:
            print(f"\n❌ Missing Unit Tests for:")
            for test in criteria.missing_tests:
                print(f"   • {test}")
        
        if criteria.missing_functionality:
            print(f"\n❌ Missing Functionality:")
            for func in criteria.missing_functionality:
                print(f"   • {func}")
        
        if criteria.pending_recommendations:
            print(f"\n❌ Pending Recommendations:")
            for rec in criteria.pending_recommendations:
                print(f"   • {rec}")
        
        print("\n" + "="*80)
    
    def start_claude_code(self) -> bool:
        """Start Claude Code process with safety flags."""
        try:
            print(f"🚀 Starting Claude Code in {self.project_path}")
            
            # Claude Code command with safety and continuation flags
            cmd = [
                "claude", "code",
                "--continue",  # Continue from where it left off
                "--skip-dangerous-permissions",  # Skip dangerous permission requests
                "--non-interactive",  # Reduce interactive prompts
                "--timeout", "3600"  # 1 hour timeout per operation
            ]
            
            # Add environment variables for additional safety
            env = os.environ.copy()
            env.update({
                "CLAUDE_AUTO_APPROVE_SAFE": "true",
                "CLAUDE_SKIP_CONFIRMATION": "true",
                "CLAUDE_MAX_FILE_SIZE": "10MB",
                "CLAUDE_WORKSPACE_ONLY": "true"
            })
            
            self.claude_process = subprocess.Popen(
                cmd,
                cwd=self.project_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Log the start
            self.restart_count += 1
            print(f"✅ Claude Code started (attempt #{self.restart_count})")
            
            return True
            
        except FileNotFoundError:
            print("❌ Error: 'claude code' command not found. Is Claude Code installed?")
            return False
        except Exception as e:
            print(f"❌ Error starting Claude Code: {e}")
            return False
    
    def is_claude_running(self) -> bool:
        """Check if Claude Code is still running."""
        if not self.claude_process:
            return False
        return self.claude_process.poll() is None
    
    def check_claude_health(self) -> Tuple[bool, Optional[str]]:
        """Check Claude Code health and get any error information."""
        if not self.claude_process:
            return False, "No process running"
        
        poll_result = self.claude_process.poll()
        if poll_result is None:
            return True, None  # Still running
        
        # Process has ended, check exit code
        if poll_result == 0:
            return False, "Normal exit"
        else:
            # Try to get error output
            try:
                stderr_output = self.claude_process.stderr.read() if self.claude_process.stderr else ""
                return False, f"Exit code {poll_result}: {stderr_output[:200]}"
            except Exception:
                return False, f"Exit code {poll_result}"
    
    def should_restart_claude(self) -> Tuple[bool, str]:
        """Determine if Claude Code should be restarted."""
        current_time = time.time()
        
        # Clean old crash timestamps (older than 1 hour)
        self.crash_timestamps = [ts for ts in self.crash_timestamps if current_time - ts < 3600]
        
        # Check restart limits
        if self.restart_count >= self.max_restarts:
            return False, f"Maximum restart limit reached ({self.max_restarts})"
        
        # Check crash frequency (no more than 3 crashes in 10 minutes)
        recent_crashes = [ts for ts in self.crash_timestamps if current_time - ts < 600]
        if len(recent_crashes) >= 3:
            return False, "Too many recent crashes (3 in 10 minutes)"
        
        return True, "Safe to restart"
    
    def restart_claude_code(self) -> bool:
        """Restart Claude Code with backoff strategy."""
        should_restart, reason = self.should_restart_claude()
        
        if not should_restart:
            print(f"❌ Not restarting Claude Code: {reason}")
            return False
        
        # Record this crash
        self.crash_timestamps.append(time.time())
        
        # Cleanup existing process
        if self.claude_process:
            try:
                print("🛑 Terminating crashed Claude Code process...")
                self.claude_process.terminate()
                try:
                    self.claude_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("⚠️ Force killing Claude Code process...")
                    self.claude_process.kill()
            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")
        
        # Exponential backoff before restart
        backoff_time = min(30, 2 ** min(len(self.crash_timestamps), 5))
        print(f"⏳ Waiting {backoff_time} seconds before restart...")
        time.sleep(backoff_time)
        
        # Save current state before restart
        self.save_monitoring_state()
        
        print(f"🔄 Restarting Claude Code (attempt #{self.restart_count + 1}/{self.max_restarts})...")
        return self.start_claude_code()
    
    def save_monitoring_state(self):
        """Save current monitoring state for recovery."""
        state = {
            "timestamp": time.time(),
            "restart_count": self.restart_count,
            "crash_timestamps": self.crash_timestamps,
            "project_path": str(self.project_path),
            "last_assessment": getattr(self, 'last_criteria', None)
        }
        
        state_file = self.project_path / ".claude_monitor_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save monitoring state: {e}")
    
    def load_monitoring_state(self):
        """Load previous monitoring state for recovery."""
        state_file = self.project_path / ".claude_monitor_state.json"
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Only restore if state is recent (within last 24 hours)
            if time.time() - state.get("timestamp", 0) < 86400:
                self.restart_count = state.get("restart_count", 0)
                self.crash_timestamps = state.get("crash_timestamps", [])
                print(f"📂 Restored monitoring state: {self.restart_count} previous restarts")
                
        except Exception as e:
            print(f"⚠️ Could not load monitoring state: {e}")
    
    def monitor_claude_output(self):
        """Monitor Claude Code output for issues (non-blocking)."""
        if not self.claude_process or not self.claude_process.stdout:
            return
        
        try:
            # Non-blocking read
            import select
            if select.select([self.claude_process.stdout], [], [], 0) == ([self.claude_process.stdout], [], []):
                line = self.claude_process.stdout.readline()
                if line:
                    # Check for error patterns
                    line_lower = line.lower()
                    if any(error in line_lower for error in [
                        "permission denied", "access denied", "connection refused",
                        "timeout", "rate limit", "authentication failed"
                    ]):
                        print(f"⚠️ Claude Code warning: {line.strip()}")
        except Exception:
            pass  # Non-critical monitoring
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
        self.running = False
        if self.claude_process:
            self.claude_process.terminate()
        sys.exit(0)
    
    def run_monitoring_loop(self):
        """Main monitoring loop."""
        self.setup_signal_handlers()
        
        if not self.start_claude_code():
            return False
        
        print(f"⏱️ Check interval: {self.check_interval} seconds")
        print("🔄 Monitoring project completion criteria...")
        print("Press Ctrl+C to stop monitoring\n")
        
        iteration = 0
        while self.running:
            try:
                iteration += 1
                print(f"\n🔄 Monitoring Iteration #{iteration}")
                print("-" * 40)
                
                # Check if Claude Code is still running
                if not self.is_claude_running():
                    print("⚠️ Claude Code process ended.")
                    break
                
                # Assess completion criteria
                criteria = self.assess_completion_criteria()
                self.display_status_report(criteria)
                
                # Check if project is complete
                if criteria.is_complete():
                    print("🎉 ALL COMPLETION CRITERIA MET! Project is ready for production.")
                    print("🛑 Stopping monitoring.")
                    break
                else:
                    print(f"⏳ Project not yet complete. Next check in {self.check_interval} seconds...")
                    time.sleep(self.check_interval)
                    
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user.")
                break
        
        # Cleanup
        if self.claude_process and self.is_claude_running():
            print("🛑 Stopping Claude Code...")
            self.claude_process.terminate()
            try:
                self.claude_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.claude_process.kill()
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Robust Claude Code Project Monitor")
    parser.add_argument("--project-path", default=".", help="Path to project directory")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds (default: 5 minutes)")
    parser.add_argument("--assess-only", action="store_true", help="Only assess criteria, don't start Claude Code")
    
    args = parser.parse_args()
    
    manager = RobustClaudeCodeManager(args.project_path, args.interval)
    
    print("🤖 Robust Claude Code Project Monitor")
    print("=" * 50)
    print("Monitoring comprehensive project completion criteria:")
    print("✅ Test coverage > 90%")
    print("✅ Unit tests for key functionality")
    print("✅ All required functionality implemented")
    print("✅ Integration tests exist")
    print("✅ All recommendations implemented")
    print("=" * 50)
    
    if args.assess_only:
        criteria = manager.assess_completion_criteria()
        manager.display_status_report(criteria)
        if criteria.is_complete():
            print("🎉 Project is complete!")
            sys.exit(0)
        else:
            print("⏳ Project is not yet complete.")
            sys.exit(1)
    else:
        success = manager.run_monitoring_loop()
        if not success:
            print("❌ Monitoring failed to start.")
            sys.exit(1)


if __name__ == "__main__":
    main()