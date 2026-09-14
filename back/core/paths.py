"""项目内稳定路径定义，避免模块移动后相对路径失效。"""

from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESOURCES_DIR = PROJECT_ROOT / "resources"
KNOWLEDGE_RESOURCES_DIR = RESOURCES_DIR / "knowledge"
