from video_audio_transcriber.evaluate import (
    ErrorRate,
    char_error_rate,
    compare,
    error_rate,
    fold,
    format_table,
    read_reference,
    tokenize,
    word_error_rate,
)


def test_identical_text_is_zero():
    assert word_error_rate("سلام دنیا", "سلام دنیا").rate == 0.0
    assert char_error_rate("سلام دنیا", "سلام دنیا").rate == 0.0


def test_counts_a_substitution():
    rate = error_rate(["a", "b", "c"], ["a", "x", "c"])
    assert (rate.hits, rate.substitutions, rate.deletions, rate.insertions) == (2, 1, 0, 0)
    assert rate.reference_length == 3


def test_counts_a_deletion():
    rate = error_rate(["a", "b", "c"], ["a", "c"])
    assert (rate.hits, rate.substitutions, rate.deletions, rate.insertions) == (2, 0, 1, 0)
    assert rate.reference_length == 3


def test_counts_an_insertion():
    rate = error_rate(["a", "b"], ["a", "x", "b"])
    assert (rate.hits, rate.substitutions, rate.deletions, rate.insertions) == (2, 0, 0, 1)
    assert rate.reference_length == 2
    assert rate.rate == 0.5


def test_total_errors_are_minimal_however_they_decompose():
    # Several alignments of this pair cost 3; only the total is well defined.
    assert error_rate(["a", "b", "c", "d"], ["a", "x", "d", "e"]).errors == 3


def test_all_deleted_and_all_inserted():
    assert error_rate(["a", "b"], []) == ErrorRate(deletions=2)
    assert error_rate([], ["a", "b"]) == ErrorRate(insertions=2)


def test_empty_reference_is_zero_not_a_crash():
    assert ErrorRate().rate == 0.0
    assert word_error_rate("", "").rate == 0.0
    assert word_error_rate("", "سلام").rate == 0.0


def test_rates_add_up_across_files():
    total = ErrorRate(8, 1, 1, 0) + ErrorRate(8, 2, 0, 0)
    assert (total.hits, total.substitutions, total.deletions) == (16, 3, 1)
    assert total.reference_length == 20
    assert total.rate == 0.2


def test_the_headline_case_heard_right_written_differently():
    # Every word differs orthographically and none differs acoustically.
    reference = "من می‌روم و کتاب‌ها را می‌خوانم؟"
    other_convention = "من می روم و كتاب ها را مي خوانم?"
    rates = compare(reference, other_convention)
    assert rates["wer"].rate == 1.0
    assert rates["wer_scoring"].rate == 0.0
    assert rates["cer_scoring"].rate == 0.0


def test_arabic_letters_are_errors_only_before_folding():
    assert word_error_rate("کتاب", "كتاب").rate == 1.0
    assert word_error_rate("کتاب", "كتاب", scoring=True).rate == 0.0


def test_zwnj_and_space_agree_in_scoring_mode():
    # Raw, one reference token became two, so the rate exceeds 100%: that is
    # normal for word error rate and worth pinning down.
    assert word_error_rate("می‌رود", "می رود").rate == 2.0
    assert word_error_rate("می‌رود", "می رود", scoring=True).rate == 0.0


def test_only_the_character_rate_can_ignore_joined_spellings():
    # Persian word boundaries are the thing the spellings disagree about, so
    # no word-level comparison can call these equal; a character one can.
    assert char_error_rate("می‌رود", "میرود", scoring=True).rate == 0.0
    assert word_error_rate("می‌رود", "میرود", scoring=True).rate > 0.0


def test_digits_and_punctuation_fold_away():
    assert word_error_rate("سال ۱۴۰۳ بود، بله", "سال 1403 بود بله", scoring=True).rate == 0.0
    assert word_error_rate("سال ۱۴۰۳", "سال ١٤٠٣", scoring=True).rate == 0.0


def test_real_errors_still_count_after_folding():
    # نقمه for نغمه is a genuine mishearing, not a spelling convention.
    assert word_error_rate("نغمه‌ی تنهایی", "نقمه‌ی تنهایی", scoring=True).rate > 0.0


def test_fold_drops_separators_only_when_asked():
    assert fold("می‌رود") == "می رود"
    assert fold("می‌رود", drop_separators=True) == "میرود"


def test_tokenize_scoring_splits_on_zwnj():
    assert tokenize("می‌رود") == ["می‌رود"]
    assert tokenize("می‌رود", scoring=True) == ["می", "رود"]


def test_character_rate_counts_characters():
    rate = char_error_rate("abcd", "abxd")
    assert rate.substitutions == 1
    assert rate.reference_length == 4


def test_too_long_to_align_is_refused_clearly():
    huge = ["x"] * 3001
    try:
        error_rate(huge, huge)
    except ValueError as exc:
        assert "3000" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_read_reference_from_a_file_and_a_directory(tmp_path):
    (tmp_path / "clip.txt").write_text("سلام", encoding="utf-8")
    assert read_reference(tmp_path / "clip.mp3", tmp_path / "clip.txt") == "سلام"
    assert read_reference(tmp_path / "clip.mp3", tmp_path) == "سلام"

    (tmp_path / "clip.reference.txt").write_text("درود", encoding="utf-8")
    assert read_reference(tmp_path / "clip.mp3", tmp_path) == "درود"  # more specific name wins
    assert read_reference(tmp_path / "other.mp3", tmp_path) is None
    assert read_reference(tmp_path / "clip.mp3", tmp_path / "missing") is None


def test_table_renders_a_total_for_several_files():
    rows = [("a.mp3", compare("سلام دنیا", "سلام دنیا")),
            ("b.mp3", compare("سلام دنیا", "سلام جهان"))]
    table = format_table(rows)
    assert "a.mp3" in table and "b.mp3" in table
    assert "total" in table
    assert "25.0%" in table  # one substitution across four reference words


def test_table_for_one_file_has_no_total():
    assert "total" not in format_table([("a.mp3", compare("سلام", "سلام"))])
