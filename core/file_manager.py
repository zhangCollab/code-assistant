"""
文件管理器 - 安全地读写文件
"""
import os
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class FileOperation(Enum):
    """文件操作类型"""
    READ = "read"
    WRITE = "write"
    EDIT = "edit"


@dataclass
class FileChange:
    """文件变更记录"""
    operation: FileOperation
    file_path: str
    timestamp: str
    original_content: str = ""
    new_content: str = ""
    success: bool = False
    error_message: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


@dataclass
class FileInfo:
    """文件信息"""
    path: str
    name: str
    extension: str
    size: int
    created_at: str
    modified_at: str
    content: str = ""
    encoding: str = "utf-8"
    language: str = "unknown"

    @property
    def exists(self) -> bool:
        return Path(self.path).exists()


class FileManager:
    """文件管理器"""

    def __init__(self, workdir: str = ".", backup_dir: str | None = None):
        self.workdir = Path(workdir).resolve()
        self.backup_dir = Path(backup_dir).resolve() if backup_dir else self.workdir / "backups"
        self.change_history: List[FileChange] = []

    def _resolve_path(self, file_path: str) -> Path:
        """将相对或绝对路径解析为workdir下的绝对路径 - 强制使用workdir"""
        path = Path(file_path)
        
        # 如果是绝对路径，强制转换为相对于workdir的路径
        if path.is_absolute():
            try:
                rel_path = path.relative_to(self.workdir)
                return (self.workdir / rel_path).resolve()
            except ValueError:
                # 路径不在workdir下，强制使用文件名部分
                return (self.workdir / path.name).resolve()
        
        return (self.workdir / path).resolve()

    def _detect_language(self, file_path: Path) -> str:
        """检测文件语言"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.yaml': 'yaml',
            '.md': 'markdown',
            '.sql': 'sql',
        }
        ext = file_path.suffix.lower() if file_path.suffix else ""
        return extension_map.get(ext, 'unknown')

    def read_file(self, file_path: str, encoding: str = "utf-8") -> Tuple[str, FileInfo]:
        """读取文件"""
        path = self._resolve_path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {file_path}")

        content = path.read_text(encoding=encoding)

        info = FileInfo(
            path=str(path.absolute()),
            name=path.name,
            extension=path.suffix.lower(),
            size=len(content.encode(encoding)),
            created_at=datetime.datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
            modified_at=datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            content=content,
            encoding=encoding,
            language=self._detect_language(path),
        )

        change = FileChange(
            operation=FileOperation.READ,
            file_path=str(path.absolute()),
            timestamp=datetime.datetime.now().isoformat(),
            success=True,
        )
        self.change_history.append(change)

        return content, info

    def write_file(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        auto_create_dir: bool = True
    ) -> FileInfo:
        """写入文件"""
        path = self._resolve_path(file_path)

        if auto_create_dir:
            path.parent.mkdir(parents=True, exist_ok=True)

        original_content = ""
        if path.exists():
            original_content = path.read_text(encoding=encoding)

        path.write_text(content, encoding=encoding)

        info = FileInfo(
            path=str(path.absolute()),
            name=path.name,
            extension=path.suffix.lower(),
            size=len(content.encode(encoding)),
            created_at=datetime.datetime.now().isoformat(),
            modified_at=datetime.datetime.now().isoformat(),
            content=content,
            encoding=encoding,
            language=self._detect_language(path),
        )

        change = FileChange(
            operation=FileOperation.WRITE if not original_content else FileOperation.EDIT,
            file_path=str(path.absolute()),
            timestamp=datetime.datetime.now().isoformat(),
            original_content=original_content,
            new_content=content,
            success=True,
        )
        self.change_history.append(change)

        return info

    def edit_file(
        self,
        file_path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False
    ) -> FileInfo:
        """编辑文件内容"""
        path = self._resolve_path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text(encoding='utf-8')
        original_content = content

        if replace_all:
            new_content = content.replace(old_text, new_text)
        else:
            if old_text not in content:
                raise ValueError(f"Text not found in file: {old_text}")
            new_content = content.replace(old_text, new_text, 1)

        path.write_text(new_content, encoding='utf-8')

        info = FileInfo(
            path=str(path.absolute()),
            name=path.name,
            extension=path.suffix.lower(),
            size=len(new_content.encode('utf-8')),
            created_at=datetime.datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
            modified_at=datetime.datetime.now().isoformat(),
            content=new_content,
            encoding="utf-8",
            language=self._detect_language(path),
        )

        change = FileChange(
            operation=FileOperation.EDIT,
            file_path=str(path.absolute()),
            timestamp=datetime.datetime.now().isoformat(),
            original_content=original_content,
            new_content=new_content,
            success=True,
        )
        self.change_history.append(change)

        return info
