"""Chunking unit tests: sections, labels, sizes, bilingual coverage."""

from support_agent.chunking import (
    chunk_file,
    chunk_markdown,
    chunk_text,
    detect_lang,
)


def test_markdown_sections_keep_doc_and_section_labels():
    text = ("# Billing\n\n## Refund policy\n\nYou can refund within 30 days.\n\n"
            "## Plan changes\n\nUpgrades apply immediately.\n")
    chunks = chunk_markdown(text, doc="billing", path="billing.md")
    assert len(chunks) == 2
    assert all(c.doc == "Billing" for c in chunks)
    sections = {c.section for c in chunks}
    assert sections == {"Refund policy", "Plan changes"}


def test_nested_sections_form_breadcrumbs():
    text = "# Doc\n\n## Parent\n\nBody one.\n\n### Child\n\nBody two.\n"
    chunks = chunk_markdown(text, doc="d", path="d.md")
    by_section = {c.section: c.text for c in chunks}
    assert "Parent" in by_section
    assert "Parent > Child" in by_section
    assert all(c.doc == "Doc" for c in chunks)


def test_long_section_splits_within_max_chars():
    para = "Sentence about refunds and billing cycles. " * 60  # ~2500 chars
    text = f"# Billing\n\n## Refunds\n\n{para}\n"
    chunks = chunk_markdown(text, doc="b", path="b.md",
                            max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_short_docs_stay_in_one_chunk():
    chunks = chunk_markdown("# T\n\n## S\n\nShort body.\n",
                            doc="t", path="t.md")
    assert len(chunks) == 1


def test_arabic_chunks_detected_and_labeled():
    text = ("# الفوترة\n\n## سياسة الاسترداد\n\nيمكنك طلب استرداد المبلغ "
            "خلال 30 يوماً من الشراء عبر صفحة الفوترة.\n")
    chunks = chunk_markdown(text, doc="billing-ar", path="billing-ar.md")
    assert chunks and all(c.lang == "ar" for c in chunks)
    assert chunks[0].doc == "الفوترة"
    assert chunks[0].section == "سياسة الاسترداد"


def test_plain_text_chunking_covers_arabic():
    text = ("الحسابات\n\nلإعادة تعيين كلمة المرور اضغط نسيت كلمة المرور "
            "في صفحة تسجيل الدخول.\n")
    chunks = chunk_text(text, doc="accounts-ar", path="a.txt")
    assert chunks and chunks[0].lang == "ar"


def test_detect_lang_english_vs_arabic():
    assert detect_lang("How do I reset my password?") == "en"
    assert detect_lang("كيف أعيد تعيين كلمة المرور؟") == "ar"
    assert detect_lang("12345 !!!") == "en"


def test_chunk_file_supports_md_and_txt(sample_doc_paths):
    counts = {p: len(chunk_file(p)) for p in sample_doc_paths}
    assert all(n >= 1 for n in counts.values())


def test_chunk_file_rejects_unknown_suffix(tmp_path):
    bad = tmp_path / "doc.pdfx"
    bad.write_text("hi", encoding="utf-8")
    try:
        chunk_file(str(bad))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown suffix")
