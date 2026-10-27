"""The quote check keeps the 13 lab's semantics, on chunk ids."""

from p2 import verify

CHUNKS = {
    "d1#0": "# Rock dust\n\nThe incombustible content shall be not less than 80 percent.",
    "d2#0": "# Seat belts\n\nSeat belts shall be provided and worn in haulage trucks.",
}


def record(qid, kind, retrieved, claims, not_found=False):
    return {"qid": qid, "kind": kind, "retrieved": retrieved, "claims": claims, "not_found": not_found}


def test_lab_semantics():
    data = {
        "answers": [
            record("a01", "in", ["d1#0"], [{"text": "80%", "chunk_id": "d1#0", "quote": "not  less than 80\nPERCENT"}]),
            record("a02", "in", ["d1#0"], [{"text": "belts", "chunk_id": "d2#0", "quote": "Seat belts shall be provided"}]),
            record("a03", "in", ["d2#0"], [{"text": "belts", "chunk_id": "d2#0", "quote": "Seat belts are optional"}]),
            record("a04", "in", ["d2#0"], [], not_found=True),
            record("a05", "out", ["d1#0"], [], not_found=True),
            record("a06", "out", ["d1#0"], [{"text": "x", "chunk_id": "d1#0", "quote": "80 percent"}], not_found=False),
        ]
    }
    totals = verify.evaluate(data, CHUNKS, label="t")
    assert (totals.n_in, totals.n_out) == (4, 2)
    assert totals.verified == 1  # a01 only: whitespace and case are normalized
    assert totals.not_found_in == 1  # a04
    assert totals.declined_out == 1  # a05
    assert (totals.claims, totals.claims_verified) == (4, 2)
    lines = "\n".join(totals.lines)
    assert "not retrieved" in lines and "quote not found" in lines
    assert totals.summary() == "Answers t | verified 1 of 4 (25%) | not_found in 1 of 4 (25%) | declined out 1 of 2 (50%)"


def test_mechanics_flag_unknown_chunks_and_malformed_claims():
    data = {"answers": [record("a01", "in", ["d9#0"], [{"text": "x", "chunk_id": "d1#0"}])]}
    problems = verify.mechanics(data, CHUNKS)
    assert any("d9#0" in p for p in problems)
    assert any("without text, chunk_id and quote" in p for p in problems)
