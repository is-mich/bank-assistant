from langchain_core.documents import Document

from document_qa_bot.ingestion.structure_splitter import split_by_structure


def page(number, text):
    """A loader page: 0-based page metadata, first line is the printed page number."""
    return Document(page_content=f"{number + 1}\n{text}", metadata={"page": number, "source": "doc.pdf"})


TOC = page(1, "ΠΙΝΑΚΑΣ ΠΕΡΙΕΧΟΜΕΝΩΝ\nΚΕΦΑΛΑΙΟ 1: ΟΡΙΣΜΟΙ\x08 4\nΚΕΦΑΛΑΙΟ 2: ΔΕΔΟΜΕΝΑ\x08 8\nΚΕΦΑΛΑΙΟ 3: ΠΛΗΡΟΦΟΡΙΕΣ\x08 9")

DEFS = page(3, (
    "ΚΕΦΑΛΑΙΟ 1: ΟΡΙΣΜΟΙ\n"
    "Οι παρακάτω όροι έχουν την έννοια που τους αποδίδεται:\n"
    "1. Κάρτα: Είναι το μέσο πληρωμής.\n"
    "2. Αντιπρόσωπος: Είναι ο νόμιμος αντιπρόσωπος.\n"
    "3. Τράπεζα: Είναι η Eurobank."
))


def split(pages, size=1000, overlap=100):
    return split_by_structure(pages, size, overlap)


def test_drops_table_of_contents_and_page_number_lines():
    chunks = split([TOC, DEFS])
    text = "\n".join(c.page_content for c in chunks)
    assert "ΠΙΝΑΚΑΣ ΠΕΡΙΕΧΟΜΕΝΩΝ" not in text
    assert "\x08" not in text
    assert not any(line.strip() == "4" for line in text.splitlines())


def test_small_clauses_are_packed_with_chapter_header():
    chunks = split([DEFS])
    assert len(chunks) == 1
    body = chunks[0].page_content
    assert body.startswith("ΚΕΦΑΛΑΙΟ 1: ΟΡΙΣΜΟΙ\n")
    assert "1. Κάρτα" in body and "3. Τράπεζα" in body
    assert chunks[0].metadata["page"] == 3
    assert chunks[0].metadata["chapter"] == "ΚΕΦΑΛΑΙΟ 1: ΟΡΙΣΜΟΙ"


def test_chunk_boundaries_fall_on_clause_starts():
    clauses = "\n".join(f"{i}. Όρος {i}: " + "λέξη " * 40 for i in range(1, 9))
    chunks = split([page(5, clauses)], size=500)
    assert len(chunks) > 1
    for chunk in chunks:
        first_body_line = chunk.page_content.splitlines()[0]
        assert first_body_line[:2].strip(". ").isdigit()  # no header here: text starts at a clause


def test_chunks_never_cross_pages():
    chunks = split([page(5, "1. Πρώτος όρος."), page(6, "2. Δεύτερος όρος.")])
    assert [c.metadata["page"] for c in chunks] == [5, 6]


def test_page_without_leading_boundary_is_marked_as_continuation():
    first = page(5, "ΚΕΦΑΛΑΙΟ 6: ΚΑΡΤΕΣ\nΕΝΟΤΗΤΑ ΙΙΙ: ΚΑΡΤΕΣ\n2. Χρήση Κάρτας\nΠρώτο μέρος του όρου.")
    cont = page(6, "συνέχεια του όρου.")
    chunks = split([first, cont])
    assert chunks[1].page_content.startswith("ΚΕΦΑΛΑΙΟ 6: ΚΑΡΤΕΣ | ΕΝΟΤΗΤΑ ΙΙΙ: ΚΑΡΤΕΣ | 2. Χρήση Κάρτας (συνέχεια)\n")
    assert chunks[1].metadata["page"] == 6


def test_section_changes_start_a_new_chunk_with_its_header():
    pages = [page(5, "ΚΕΦΑΛΑΙΟ 6: ΚΑΡΤΕΣ\nΕΝΟΤΗΤΑ Ι: ΚΑΤΑΘΕΣΕΙΣ\n1. Όρος α.\nΕΝΟΤΗΤΑ ΙΙ: ΕΝΤΟΛΕΣ\n1. Όρος β.")]
    chunks = split(pages)
    assert len(chunks) == 2
    assert "ΕΝΟΤΗΤΑ Ι: ΚΑΤΑΣΤΗΜΑΤΑ" not in chunks[0].page_content
    assert chunks[0].page_content.startswith("ΚΕΦΑΛΑΙΟ 6: ΚΑΡΤΕΣ | ΕΝΟΤΗΤΑ Ι: ΚΑΤΑΘΕΣΕΙΣ\n")
    assert chunks[1].page_content.startswith("ΚΕΦΑΛΑΙΟ 6: ΚΑΡΤΕΣ | ΕΝΟΤΗΤΑ ΙΙ: ΕΝΤΟΛΕΣ\n")
    assert chunks[1].metadata["section"] == "ΕΝΟΤΗΤΑ ΙΙ: ΕΝΤΟΛΕΣ"


def test_oversized_clause_is_split_within_limit_and_marked_as_continued():
    long_clause = "3. Μεγάλος όρος: " + " ".join(f"λέξη{i}." for i in range(400))
    chunks = split([page(7, "ΚΕΦΑΛΑΙΟ 6: ΚΑΡΤΕΣ\n" + long_clause)], size=400, overlap=40)
    assert len(chunks) > 3
    assert all(len(c.page_content) <= 400 + 10 for c in chunks)
    assert "(συνέχεια)" in chunks[1].page_content.splitlines()[0]
    assert all(c.metadata["page"] == 7 for c in chunks)


def test_broken_hyphen_in_heading_is_repaired():
    text = "7\nΚΕΦΑΛΑΙΟ 6: ΚΑΤΑΘΕΣΕΙΣ\ufffeΕΝΤΟΛΕΣ\n1. Όρος."
    [chunk] = split([Document(page_content=text, metadata={"page": 6, "source": "doc.pdf"})])
    assert chunk.page_content.startswith("ΚΕΦΑΛΑΙΟ 6: ΚΑΤΑΘΕΣΕΙΣ-ΕΝΤΟΛΕΣ\n")


def test_plain_text_without_structure_still_splits():
    chunks = split([Document(page_content="Απλό κείμενο χωρίς δομή. " * 100, metadata={"source": "a.txt"})], size=300)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 300 + 10 for c in chunks)
