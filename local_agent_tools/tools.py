"""
Tool definitions for the local agent.

This module defines a collection of simple tools that can be
invoked by a language model via structured tool calls.  Each tool
accepts JSON‑serialisable parameters and returns a result that can
be fed back into the conversation.  The tools are designed to run
locally without any network access, supporting offline workflows.

Available tools:

* ``list_dir`` – list the contents of a directory.
* ``read_file`` – read the contents of a text file.
* ``write_file`` – write content to a text file, creating it if
  necessary.
* ``search_file`` – recursively search for a keyword in files under
  a directory and return paths of matching files.
* ``run_python`` – execute a snippet of Python code and return its
  stdout or exception.

The ``TOOLS`` dictionary maps tool names to callables and their
descriptions.  When adding new tools, update this dictionary so
that the agent can advertise them to the model.
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
import traceback
from typing import List, Dict, Any

def list_dir(path: str) -> Dict[str, Any]:
    """List the contents of a directory.

    Parameters
    ----------
    path : str
        Path to the directory to list.  If omitted or empty, defaults
        to the current working directory.

    Returns
    -------
    dict
        A dictionary containing the directory path and a list of entries.
        Each entry is a dict with ``name`` and ``type`` (``"file"`` or
        ``"dir"``).
    """
    if not path:
        path = os.getcwd()
    try:
        entries = []
        for entry in os.scandir(path):
            entry_type = "dir" if entry.is_dir() else "file"
            entries.append({"name": entry.name, "type": entry_type})
        return {"path": path, "entries": entries}
    except Exception as exc:
        return {"error": str(exc)}


def read_file(path: str, max_bytes: int = 100_000) -> Dict[str, Any]:
    """Read the contents of a file.

    Parameters
    ----------
    path : str
        Path to the file to read.
    max_bytes : int, optional
        Maximum number of bytes to read.  Defaults to 100k.  This
        prevents loading extremely large files into memory.  If the file
        exceeds this limit, only the first ``max_bytes`` bytes are
        returned.

    Returns
    -------
    dict
        A dictionary containing the file path and its content as a
        string, or an ``error`` field if reading fails.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read(max_bytes)
        return {"path": path, "content": data}
    except Exception as exc:
        return {"error": str(exc)}


def write_file(path: str, content: str) -> Dict[str, Any]:
    """Write content to a file.

    Parameters
    ----------
    path : str
        Path to the file to write.  Parent directories will be
        created if they do not exist.
    content : str
        Content to write to the file.

    Returns
    -------
    dict
        A dictionary indicating success or error.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"path": path, "status": "written"}
    except Exception as exc:
        return {"error": str(exc)}


def search_file(query: str, directory: str = ".", max_results: int = 10) -> Dict[str, Any]:
    """Search for files containing a query string.

    Parameters
    ----------
    query : str
        Text to search for within files.
    directory : str, optional
        Root directory to search recursively.  Defaults to the current
        working directory.
    max_results : int, optional
        Maximum number of matching file paths to return.  Defaults to 10.

    Returns
    -------
    dict
        A dictionary containing a list of file paths that match the
        query.  If no files match, the list will be empty.
    """
    matches: List[str] = []
    try:
        for root, _, files in os.walk(directory):
            for fname in files:
                try:
                    fpath = os.path.join(root, fname)
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        if query in f.read():
                            matches.append(fpath)
                            if len(matches) >= max_results:
                                return {"matches": matches}
                except Exception:
                    # Skip files that cannot be read
                    continue
        return {"matches": matches}
    except Exception as exc:
        return {"error": str(exc)}


def run_python(code: str, timeout: int = 10) -> Dict[str, Any]:
    """Execute a Python code snippet and return its output.

    This function spawns a separate Python process to execute the
    provided code.  Standard output and standard error are captured
    and returned.  A timeout (in seconds) can be specified to avoid
    hanging processes.

    Parameters
    ----------
    code : str
        The Python code to execute.  It should be self‑contained and
        not require any external input.
    timeout : int, optional
        Maximum number of seconds to allow the code to run.  Defaults
        to 10 seconds.

    Returns
    -------
    dict
        A dictionary containing ``stdout`` and ``stderr`` fields.  If
        execution fails, an ``error`` field will contain the
        exception message.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout expired"}
    except Exception as exc:
        return {"error": str(exc)}


TOOLS: Dict[str, Dict[str, Any]] = {
    "list_dir": {
        "function": list_dir,
        "description": "List the contents of a directory. Parameters: path (optional). Returns a list of entries with name and type.",
    },
    "read_file": {
        "function": read_file,
        "description": "Read the contents of a text file. Parameters: path. Returns the file content or an error.",
    },
    "write_file": {
        "function": write_file,
        "description": "Write text to a file, creating it if needed. Parameters: path, content. Returns status or error.",
    },
    "search_file": {
        "function": search_file,
        "description": "Recursively search for a query string in files under a directory. Parameters: query, directory (optional). Returns matching file paths.",
    },
    "run_python": {
        "function": run_python,
        "description": "Execute a Python code snippet in a sandboxed process. Parameters: code. Returns stdout, stderr and returncode.",
    },
}

__all__ = [
    "list_dir",
    "read_file",
    "write_file",
    "search_file",
    "run_python",
    "TOOLS",
]