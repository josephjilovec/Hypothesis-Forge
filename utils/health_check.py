"""
Health check utilities for production monitoring.
"""
import time
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config.config import BASE_DIR, RESULTS_DIR, HYPOTHESIS_DIR, PROCESSED_DATA_DIR


class HealthChecker:
    """System health checker for monitoring."""

    @staticmethod
    def check_disk_space() -> Dict[str, Any]:
        """Check available disk space."""
        if not PSUTIL_AVAILABLE:
            return {"status": "unknown", "message": "psutil not available"}
        try:
            disk = psutil.disk_usage(BASE_DIR)
            return {
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "percent_used": disk.percent,
                "status": "healthy" if disk.percent < 90 else "warning",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def check_memory() -> Dict[str, Any]:
        """Check system memory usage."""
        if not PSUTIL_AVAILABLE:
            return {"status": "unknown", "message": "psutil not available"}
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "used_gb": round(memory.used / (1024 ** 3), 2),
                "percent_used": memory.percent,
                "status": "healthy" if memory.percent < 85 else "warning",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def check_cpu() -> Dict[str, Any]:
        """Check CPU usage."""
        if not PSUTIL_AVAILABLE:
            return {"status": "unknown", "message": "psutil not available"}
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            return {
                "usage_percent": cpu_percent,
                "cpu_count": cpu_count,
                "status": "healthy" if cpu_percent < 80 else "warning",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def check_directories() -> Dict[str, Any]:
        """Check if required directories exist and are writable."""
        directories = {
            "results": RESULTS_DIR,
            "hypothesis": HYPOTHESIS_DIR,
            "processed": PROCESSED_DATA_DIR,
        }
        status = {}
        all_healthy = True

        for name, path in directories.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                # Test write
                test_file = path / ".health_check"
                test_file.write_text("test")
                test_file.unlink()
                status[name] = {"exists": True, "writable": True, "status": "healthy"}
            except Exception as e:
                status[name] = {"exists": False, "writable": False, "status": "error", "error": str(e)}
                all_healthy = False

        return {
            "directories": status,
            "overall_status": "healthy" if all_healthy else "error",
        }

    @staticmethod
    def check_data_files() -> Dict[str, Any]:
        """Check if data files exist."""
        files = {
            "simulation_results": RESULTS_DIR / "simulation_results.parquet",
            "ranked_hypotheses": HYPOTHESIS_DIR / "ranked_hypotheses.json",
        }
        status = {}
        for name, path in files.items():
            status[name] = {
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        return status

    @classmethod
    def get_full_health_report(cls) -> Dict[str, Any]:
        """Get complete health report."""
        return {
            "timestamp": time.time(),
            "disk": cls.check_disk_space(),
            "memory": cls.check_memory(),
            "cpu": cls.check_cpu(),
            "directories": cls.check_directories(),
            "data_files": cls.check_data_files(),
            "overall_status": "healthy",
        }


def get_health_status() -> Dict[str, Any]:
    """Get health status for API endpoint."""
    try:
        report = HealthChecker.get_full_health_report()
        # Determine overall status
        statuses = [
            report["disk"].get("status"),
            report["memory"].get("status"),
            report["cpu"].get("status"),
            report["directories"].get("overall_status"),
        ]
        if "error" in statuses:
            report["overall_status"] = "error"
        elif "warning" in statuses:
            report["overall_status"] = "warning"
        else:
            report["overall_status"] = "healthy"

        return report
    except Exception as e:
        return {
            "timestamp": time.time(),
            "overall_status": "error",
            "error": str(e),
        }

