import pytest

from src.indexing import FaqDocument
from src.retrieval import SearchResult


def test_faq_document_builds_embedding_text_by_mode() -> None:
    document = FaqDocument(row_id=12, title="응시수수료 반환", body="접수 취소 시 반환 기준입니다.")

    assert document.embedding_text("title") == "응시수수료 반환"
    assert document.embedding_text("title_body") == (
        "제목: 응시수수료 반환\n본문: 접수 취소 시 반환 기준입니다."
    )


def test_faq_document_rejects_unknown_embedding_mode() -> None:
    document = FaqDocument(row_id=12, title="제목", body="본문")

    with pytest.raises(ValueError, match="Unsupported embedding mode"):
        document.embedding_text("body")


def test_search_result_exposes_document_and_score() -> None:
    document = FaqDocument(row_id=12, title="제목", body="본문")

    result = SearchResult(document=document, score=0.81)

    assert result.document.row_id == 12
    assert result.score == 0.81
