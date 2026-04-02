from training.passage_filter import PassageFilter


def test_deduplicate_preserves_first_seen_order():
    passage_filter = PassageFilter(min_passage_length=0)

    passages = ["alpha", "beta", "alpha", "gamma", "beta"]

    assert passage_filter.deduplicate(passages) == ["alpha", "beta", "gamma"]


def test_remove_stubs_respects_minimum_length():
    passage_filter = PassageFilter(min_passage_length=5)

    assert passage_filter.remove_stubs(["tiny", "large", "lengthy"]) == ["large", "lengthy"]


def test_filter_combines_deduplication_and_stub_removal():
    passage_filter = PassageFilter(min_passage_length=6)

    passages = ["short", "longer!", "longer!", "wide ok", "small"]

    assert passage_filter.filter(passages) == ["longer!", "wide ok"]
