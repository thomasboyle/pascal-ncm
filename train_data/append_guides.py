"""Append licensed C++ and Python guides to code_corpus.txt."""
import os
from pathlib import Path

CORPUS = Path(r"D:\C++\LocalAI\code_corpus.txt")
SRC = Path(r"D:\C++\LocalAI\train_data\src")
MARKER = "# FILE: guides/ATTRIBUTION.txt\n"
TEXT_EXT = {".md", ".rst", ".txt", ".html", ".py", ".h", ".cc", ".inc", ".hpp", ".cpp"}


def want(path: Path, root: Path) -> bool:
    if path.suffix.lower() not in TEXT_EXT:
        return False
    rel = path.relative_to(root).as_posix().lower()
    parts = rel.split("/")
    if any(p in {"test", "tests", "idle_test", "testing", "benchmark", "__pycache__"} for p in parts):
        return False
    name = path.name.lower()
    if name.endswith(("_test.cc", "_test.h", "_benchmark.cc", "_benchmark.h")):
        return False
    return True


def collect(root: Path, exts: set[str] | None = None) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "test", "tests", "benchmark"}]
        for name in filenames:
            path = Path(dirpath) / name
            if exts is not None and path.suffix.lower() not in exts:
                continue
            if want(path, root):
                files.append(path)
    return sorted(files)


def main() -> None:
    existing = CORPUS.read_text(encoding="utf-8", errors="replace")
    if MARKER in existing:
        raise SystemExit("guides already appended")

    groups: list[tuple[str, list[Path]]] = [
        ("cpp/CppCoreGuidelines.md", [SRC / "CppCoreGuidelines" / "CppCoreGuidelines.md"]),
        ("cpp/CppCoreGuidelines-LICENSE", [SRC / "CppCoreGuidelines" / "LICENSE"]),
        ("google/cppguide.html", [SRC / "styleguide" / "cppguide.html"]),
        ("google/pyguide.md", [SRC / "styleguide" / "pyguide.md"]),
        ("google/LICENSE", [SRC / "styleguide" / "LICENSE"]),
        ("python/pep-0008.rst", [SRC / "extra" / "pep-0008.rst"]),
        ("python/CC-BY-4.0.txt", [SRC / "extra" / "CC-BY-4.0.txt"]),
        ("python/MIT.txt", [SRC / "extra" / "MIT.txt"]),
        ("owasp", collect(SRC / "extra" / "owasp", {".md"})),
        ("optimizing", collect(SRC / "extra" / "optimizing")),
        ("cert-cpp", collect(SRC / "secure-coding-standards" / "content" / "5.sei-cert-cpp-coding-standard", {".md"})),
        ("cert-c", collect(SRC / "secure-coding-standards" / "content" / "4.sei-cert-c-coding-standard", {".md"})),
        ("openssf-python", collect(SRC / "ossf" / "docs" / "Secure-Coding-Guide-for-Python")),
        ("cpython-LICENSE", [SRC / "cpython" / "LICENSE"]),
        ("cpython", collect(SRC / "cpython" / "Lib", {".py"})),
        ("abseil-LICENSE", [SRC / "abseil-cpp" / "LICENSE"]),
        ("abseil", collect(SRC / "abseil-cpp" / "absl", {".h", ".cc", ".inc", ".hpp"})),
    ]

    attribution = """Local training appendix. Each following file keeps the license named here.

C++ Core Guidelines, Standard C++ Foundation. Copy is limited to personal or internal business use. Notice is in guides/cpp/CppCoreGuidelines-LICENSE.
Google C++ Style Guide and Google Python Style Guide. CC BY 3.0. Notice is in guides/google/LICENSE.
SEI CERT C Coding Standard and SEI CERT C++ Coding Standard. Copyright Carnegie Mellon University. Standards text CC BY 4.0. Code examples MIT. Source: https://github.com/cmu-sei/secure-coding-standards
OpenSSF Secure Coding Guide for Python. Documentation CC BY 4.0. Code examples MIT. Educational examples, not production code.
PEP 8. This document has been placed in the public domain.
OWASP Cheat Sheet Series excerpts (Django, Django REST Framework, FastAPI). CC BY-SA 4.0. https://github.com/OWASP/CheatSheetSeries
Scientific Python Lectures, advanced/optimizing, by Gael Varoquaux and the project authors. CC BY 4.0.
CPython standard library sources under Lib, test directories omitted. Python Software Foundation license, guides/cpython/LICENSE.
Abseil C++ library sources, test and benchmark files omitted. Apache 2.0. Notice is in guides/abseil/LICENSE.
"""

    added = 0
    nbytes = 0
    with CORPUS.open("a", encoding="utf-8", newline="\n") as out:
        out.write("\n" + MARKER)
        out.write(attribution)
        if not attribution.endswith("\n"):
            out.write("\n")
        for label, paths in groups:
            for path in paths:
                if not path.is_file():
                    raise SystemExit(f"missing {path}")
                raw = path.read_bytes()
                if b"\x00" in raw[:8192]:
                    continue
                text = raw.decode("utf-8", errors="replace")
                if label in {"owasp", "optimizing", "cert-cpp", "cert-c", "openssf-python", "cpython", "abseil"}:
                    rel = Path("guides") / label / path.relative_to(
                        {
                            "owasp": SRC / "extra" / "owasp",
                            "optimizing": SRC / "extra" / "optimizing",
                            "cert-cpp": SRC / "secure-coding-standards" / "content" / "5.sei-cert-cpp-coding-standard",
                            "cert-c": SRC / "secure-coding-standards" / "content" / "4.sei-cert-c-coding-standard",
                            "openssf-python": SRC / "ossf" / "docs" / "Secure-Coding-Guide-for-Python",
                            "cpython": SRC / "cpython" / "Lib",
                            "abseil": SRC / "abseil-cpp" / "absl",
                        }[label]
                    )
                else:
                    rel = Path("guides") / label
                out.write(f"\n# FILE: {rel.as_posix()}\n")
                out.write(text)
                if not text.endswith("\n"):
                    out.write("\n")
                added += 1
                nbytes += len(text.encode("utf-8"))
    print(f"appended {added} files, {nbytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
