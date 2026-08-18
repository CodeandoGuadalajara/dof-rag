"""Validated, framework-neutral HTTP contracts for human evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

RATINGS = frozenset({"helpful", "partially_helpful", "not_helpful"})
PROBLEM_TYPES = frozenset(
    {
        "incorrect_answer",
        "missing_evidence",
        "bad_citation",
        "incomplete_coverage",
        "cutoff_error",
        "hard_to_understand",
        "other",
    }
)
CLIENT_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class ContractError(ValueError):
    """An API request does not match the public contract."""

    def __init__(self, message: str, *, field: str | None = None):
        super().__init__(message)
        self.field = field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_unknown(data: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ContractError(f"unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class RunRequest:
    question: str
    as_of: str | None = None
    required_hops: int = 1
    client_request_id: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> "RunRequest":
        if not isinstance(data, dict):
            raise ContractError("request body must be a JSON object")
        _reject_unknown(
            data, {"question", "as_of", "required_hops", "client_request_id"}
        )
        question = data.get("question")
        if not isinstance(question, str):
            raise ContractError("question must be a string", field="question")
        question = question.strip()
        if not 3 <= len(question) <= 2000:
            raise ContractError(
                "question must contain between 3 and 2000 characters",
                field="question",
            )
        as_of = data.get("as_of")
        if as_of is not None:
            if not isinstance(as_of, str):
                raise ContractError("as_of must be an ISO date", field="as_of")
            try:
                date.fromisoformat(as_of)
            except ValueError as exc:
                raise ContractError(
                    "as_of must be an ISO date (YYYY-MM-DD)", field="as_of"
                ) from exc
        required_hops = data.get("required_hops", 1)
        if (
            not isinstance(required_hops, int)
            or isinstance(required_hops, bool)
            or not 1 <= required_hops <= 5
        ):
            raise ContractError(
                "required_hops must be an integer between 1 and 5",
                field="required_hops",
            )
        client_request_id = data.get("client_request_id")
        if client_request_id is not None and (
            not isinstance(client_request_id, str)
            or not CLIENT_REQUEST_ID_RE.fullmatch(client_request_id)
        ):
            raise ContractError(
                "client_request_id must use 1-128 letters, digits, '.', '_', ':' or '-'",
                field="client_request_id",
            )
        return cls(question, as_of, required_hops, client_request_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "as_of": self.as_of,
            "required_hops": self.required_hops,
            "client_request_id": self.client_request_id,
        }


@dataclass(frozen=True)
class FeedbackRequest:
    rating: str
    problem_types: tuple[str, ...] = ()
    comment: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "FeedbackRequest":
        if not isinstance(data, dict):
            raise ContractError("request body must be a JSON object")
        _reject_unknown(data, {"rating", "problem_types", "comment"})
        rating = data.get("rating")
        if rating not in RATINGS:
            raise ContractError(
                f"rating must be one of: {', '.join(sorted(RATINGS))}",
                field="rating",
            )
        raw_problems = data.get("problem_types", [])
        if not isinstance(raw_problems, list) or not all(
            isinstance(item, str) for item in raw_problems
        ):
            raise ContractError(
                "problem_types must be an array of strings", field="problem_types"
            )
        problems = tuple(dict.fromkeys(raw_problems))
        unknown = sorted(set(problems) - PROBLEM_TYPES)
        if unknown:
            raise ContractError(
                f"unknown problem_types: {', '.join(unknown)}",
                field="problem_types",
            )
        comment = data.get("comment", "")
        if not isinstance(comment, str) or len(comment.strip()) > 2000:
            raise ContractError(
                "comment must be a string of at most 2000 characters",
                field="comment",
            )
        return cls(rating, problems, comment.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "rating": self.rating,
            "problem_types": list(self.problem_types),
            "comment": self.comment,
        }
