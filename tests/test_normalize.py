from video_audio_transcriber.normalize import ZWNJ, normalize, normalize_word


def test_arabic_letters_become_persian():
    assert normalize("كتاب يك") == "کتاب یک"


def test_arabic_indic_digits_become_persian_even_in_keep_mode():
    assert normalize("سال ١٤٠٣") == "سال ۱۴۰۳"


def test_digit_modes():
    assert normalize("سال 1403", digits="persian") == "سال ۱۴۰۳"
    assert normalize("سال ۱۴۰۳", digits="western") == "سال 1403"
    assert normalize("سال 1403") == "سال 1403"


def test_mi_prefix_joined():
    assert normalize("من می روم و نمی دانم") == f"من می{ZWNJ}روم و نمی{ZWNJ}دانم"


def test_mi_not_joined_before_particles():
    assert normalize("می و مطرب") == "می و مطرب"


def test_mi_at_end_of_a_word_untouched():
    assert normalize("تقدیمی شد") == "تقدیمی شد"


def test_plural_suffix_joined():
    assert normalize("کتاب ها را خواندم") == f"کتاب{ZWNJ}ها را خواندم"
    assert normalize("بچه هایشان") == f"بچه{ZWNJ}هایشان"


def test_superlative_joined():
    assert normalize("بزرگ ترین شهر") == f"بزرگ{ZWNJ}ترین شهر"


def test_affix_joining_can_be_disabled():
    assert normalize("می روم", zwnj=False) == "می روم"


def test_latin_punctuation_after_persian():
    assert normalize("خوبی ?") == "خوبی؟"
    assert normalize("سلام, خوبی") == "سلام، خوبی"
    assert normalize("سلام,خوبی") == "سلام، خوبی"
    assert normalize("اول; دوم") == "اول؛ دوم"


def test_spacing_around_punctuation():
    assert normalize("سلام .خوبی ؟") == "سلام. خوبی؟"
    assert normalize("یک  ،  دو") == "یک، دو"
    assert normalize("« سلام »") == "«سلام»"


def test_numbers_keep_their_separators():
    assert normalize("ساعت 12:30 و 3.5 درصد") == "ساعت 12:30 و 3.5 درصد"
    assert normalize("۱،۰۰۰ تومان") == "۱،۰۰۰ تومان"


def test_stray_zwnj_removed():
    assert normalize(f"کتاب{ZWNJ} ها{ZWNJ}") == f"کتاب{ZWNJ}ها"
    assert normalize(f"می{ZWNJ}{ZWNJ}روم") == f"می{ZWNJ}روم"


def test_english_untouched():
    assert normalize("Hello, world? Yes; 1,000.") == "Hello, world? Yes; 1,000."


def test_idempotent():
    text = "سلام، من می‌روم و کتاب‌ها را می‌خوانم؟"
    assert normalize(normalize(text)) == normalize(text)


def test_normalize_word_is_character_level_only():
    assert normalize_word(" كتاب ها") == " کتاب ها"


def test_empty():
    assert normalize("") == ""
    assert normalize("   ") == ""
