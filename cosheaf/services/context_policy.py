"""Context exposure policy checks for provider and agent send previews."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from cosheaf.core.artifact import BaseArtifact
from cosheaf.memory import ArtifactCard, MemoryRootScope, search_artifact_cards
from cosheaf.services.models import (
    ContextBuildRequest,
    ContextPolicyMode,
    ErrorResult,
    ProviderContextPreview,
    ProviderContextPreviewItem,
)
from cosheaf.storage.loader import IssueRecord, LoadedRecord, LoadError, load_artifacts
from cosheaf.storage.repo import RepoContext

PUBLIC_PROVIDER_SCOPES = (MemoryRootScope.PUBLIC,)
PRIVATE_RESEARCH_PROVIDER_SCOPES = (
    MemoryRootScope.PRIVATE,
    MemoryRootScope.PUBLIC,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class _PolicyDecision:
    allowed_scopes: tuple[MemoryRootScope, ...]
    private_context_requested: bool


class ContextSendPolicyService:
    """Policy boundary for context shown to agents or sent to providers."""

    def __init__(self, context: RepoContext) -> None:
        self.context = context

    def provider_preview(
        self,
        request: ContextBuildRequest,
    ) -> ProviderContextPreview | ErrorResult:
        """Return a safe provider-send preview or a structured denial."""
        decision_or_error = _provider_scope_decision(request)
        if isinstance(decision_or_error, ErrorResult):
            return decision_or_error

        decision = decision_or_error
        try:
            records = tuple(load_artifacts(self.context))
            issue = _find_issue(records, request.issue_id)
            allowed_related_artifacts = _allowed_related_artifact_ids(
                records,
                issue=issue,
                allowed_scopes=decision.allowed_scopes,
            )
            retrieval = search_artifact_cards(
                self.context,
                query=_issue_query(
                    issue,
                    allowed_related_artifacts=allowed_related_artifacts,
                ),
                issue_id=issue.id,
                max_cards=request.max_cards,
                allowed_scopes=decision.allowed_scopes,
                include_refuted=True,
                include_obsolete=True,
                role=request.role,
                max_full_artifacts=0,
            )
        except (LoadError, ValueError) as exc:
            return ErrorResult(
                code="provider_context_preview_failed",
                message=str(exc),
                remediation="Fix repository records or choose an existing issue id.",
                blocking=True,
            )

        hits = tuple(retrieval.cards)
        leaked = sorted(
            hit.card.id
            for hit in hits
            if hit.card.root_scope not in decision.allowed_scopes
        )
        if leaked:
            return ErrorResult(
                code="provider_context_scope_violation",
                message="Context retrieval returned cards outside the allowed scope.",
                remediation=(
                    "Do not send this context. Rebuild the preview with the "
                    "intended policy scope before any provider call."
                ),
                blocking=True,
                details={"artifact_ids": ",".join(leaked)},
            )

        items = tuple(_preview_item(hit.card) for hit in hits)
        root_scopes = _ordered_root_scopes(items)
        risk_flags = _preview_risk_flags(items)
        private_included = MemoryRootScope.PRIVATE in root_scopes
        if private_included:
            risk_flags = tuple(dict.fromkeys((*risk_flags, "private_context")))

        return ProviderContextPreview(
            issue_id=issue.id,
            policy_mode=request.policy_mode,
            public_only=request.public_only,
            private_context_requested=decision.private_context_requested,
            private_context_included=private_included,
            artifact_ids=[item.artifact_id for item in items],
            root_scopes=list(root_scopes),
            estimated_tokens=sum(item.estimated_tokens for item in items),
            card_count=len(items),
            full_artifact_count=len(retrieval.full_artifact_pulls),
            content_mode=_preview_content_mode(
                full_artifact_count=len(retrieval.full_artifact_pulls)
            ),
            risk_flags=list(risk_flags),
            items=list(items),
        )


def _provider_scope_decision(
    request: ContextBuildRequest,
) -> _PolicyDecision | ErrorResult:
    private_context_requested = not request.public_only
    if request.policy_mode is ContextPolicyMode.PUBLIC:
        if private_context_requested:
            return ErrorResult(
                code="private_context_requires_policy",
                message="Private context cannot be previewed in public policy mode.",
                remediation=(
                    "Use policy_mode=private_research with explicit private-context "
                    "consent before previewing private KB context."
                ),
                blocking=True,
                details={
                    "policy_mode": request.policy_mode.value,
                    "public_only": str(request.public_only).lower(),
                    "allow_private_context": str(
                        request.allow_private_context
                    ).lower(),
                },
            )
        return _PolicyDecision(
            allowed_scopes=PUBLIC_PROVIDER_SCOPES,
            private_context_requested=False,
        )

    if request.policy_mode is ContextPolicyMode.PRIVATE_RESEARCH:
        if private_context_requested and not request.allow_private_context:
            return ErrorResult(
                code="private_context_requires_consent",
                message="Private research context requires explicit consent.",
                remediation=(
                    "Set allow_private_context=true only when the operator has "
                    "approved sending private KB context to the provider."
                ),
                blocking=True,
                details={
                    "policy_mode": request.policy_mode.value,
                    "public_only": str(request.public_only).lower(),
                    "allow_private_context": str(
                        request.allow_private_context
                    ).lower(),
                },
            )
        return _PolicyDecision(
            allowed_scopes=PRIVATE_RESEARCH_PROVIDER_SCOPES
            if private_context_requested
            else PUBLIC_PROVIDER_SCOPES,
            private_context_requested=private_context_requested,
        )

    return ErrorResult(
        code="unknown_context_policy_mode",
        message=f"Unsupported context policy mode: {request.policy_mode.value}",
        remediation="Use policy_mode=public or policy_mode=private_research.",
        blocking=True,
    )


def _find_issue(records: tuple[LoadedRecord, ...], issue_id: str) -> IssueRecord:
    issues = [
        record.record
        for record in records
        if isinstance(record.record, IssueRecord) and record.record.id == issue_id
    ]
    if not issues:
        raise ValueError(f"issue not found: {issue_id}")
    if len(issues) > 1:
        raise ValueError(f"duplicate issue id: {issue_id}")
    return issues[0]


def _allowed_related_artifact_ids(
    records: tuple[LoadedRecord, ...],
    *,
    issue: IssueRecord,
    allowed_scopes: tuple[MemoryRootScope, ...],
) -> tuple[str, ...]:
    scope_by_id = {
        record.id: _root_scope_for_loaded_record(record)
        for record in records
        if isinstance(record.record, BaseArtifact)
    }
    return tuple(
        artifact_id
        for artifact_id in issue.related_artifacts
        if scope_by_id.get(artifact_id) in allowed_scopes
    )


def _issue_query(
    issue: IssueRecord,
    *,
    allowed_related_artifacts: tuple[str, ...],
) -> str:
    parts = [
        issue.title,
        issue.description,
        *issue.tags,
        *allowed_related_artifacts,
    ]
    return " ".join(part for part in parts if part.strip()).strip() or issue.id


def _preview_item(card: ArtifactCard) -> ProviderContextPreviewItem:
    return ProviderContextPreviewItem(
        artifact_id=card.id,
        root_scope=card.root_scope,
        status=card.status,
        estimated_tokens=_estimated_card_tokens(card),
        risk_flags=card.risk_flags,
    )


def _estimated_card_tokens(card: ArtifactCard) -> int:
    text = " ".join(
        [
            card.id,
            card.title,
            card.summary,
            *card.domain,
            *card.tags,
            *card.depends_on,
            *card.sources,
            *card.risk_flags,
            card.review_state,
            card.verifier_state,
            card.formalization_state,
        ]
    )
    return max(1, len(TOKEN_RE.findall(text)))


def _ordered_root_scopes(
    items: tuple[ProviderContextPreviewItem, ...],
) -> tuple[MemoryRootScope, ...]:
    return tuple(dict.fromkeys(item.root_scope for item in items))


def _preview_risk_flags(
    items: tuple[ProviderContextPreviewItem, ...],
) -> tuple[str, ...]:
    flags: list[str] = []
    for item in items:
        flags.extend(item.risk_flags)
    return tuple(dict.fromkeys(flag for flag in flags if flag))


def _preview_content_mode(
    *,
    full_artifact_count: int,
) -> Literal["cards_only", "cards_with_full_artifacts"]:
    if full_artifact_count > 0:
        return "cards_with_full_artifacts"
    return "cards_only"


def _root_scope_for_loaded_record(loaded: LoadedRecord) -> MemoryRootScope:
    if loaded.is_private_scope:
        return MemoryRootScope.PRIVATE
    kb_root_name = (loaded.kb_root_name or "").lower()
    if kb_root_name == "framework":
        return MemoryRootScope.FRAMEWORK
    if loaded.is_public_kb_root or kb_root_name == "public":
        return MemoryRootScope.PUBLIC
    return MemoryRootScope.WORKSPACE


__all__ = [
    "ContextSendPolicyService",
    "PRIVATE_RESEARCH_PROVIDER_SCOPES",
    "PUBLIC_PROVIDER_SCOPES",
]
