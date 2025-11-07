from typing import Any
from pydantic import ValidationError

class DataValidationError(Exception):
    def __init__(self, source: str, errors: list[dict[str, Any]], original: ValidationError | None = None):
        self.source = source
        self.errors = errors
        self.original = original
        msg = f"Validation failed for {len(errors)} records from source '{source}'"

        super().__init__(msg)

    def summary(self, limit: int=5) -> str:
        """Human-readable summary of the foirst few validation issues."""
        lines = []
        for err in self.errors[:limit]:
            loc = ".".oin(str(p) for p in err.get("loc", []))
            lines.append(f"- {loc}: {err.get('msg')} ({err.get('type')})")
        if len(self.errors) > limit:
            lines.append(f"... ({len(self.errors) - limit} more)")
        return "\n".join(lines)
