"""The fast guard holds tests/README.md's writing rules on what a change adds."""

from check_test_shape import problems

PATH = "tests/owner/test_example.py"


def test_the_shape_check_refuses_what_a_change_adds_and_spares_what_it_keeps() -> None:
    """Each rule fires on added content; kept lines, shrinking files and tests/structural pass."""
    big = "x = 1\n" * 20_000
    assert problems(PATH, big, big + "y = 2\n") and not problems(PATH, big + "y = 2\n", big)
    source = "import inspect\ninspect.getsource(owner)\n"
    assert problems(PATH, "", source) and not problems(PATH, source, source)
    assert not problems("tests/structural/test_example.py", "", source)
    pin = "assert.ok(html.includes('The update waits for the provider'));\n"
    cjs = "tests/owner/workbench_example.cjs"
    assert problems(cjs, "", pin) and not problems(cjs, pin, pin)
    assert not problems(cjs, "", "assert.ok(html.includes('data-key=\"wait\"'));\n")
    assert problems(PATH, "", 'assert "the update waits for the provider" in text\n')
    assert problems(PATH, "", "time.sleep" + "(2)\n") and not problems(
        PATH, "", "time.sleep(0.05)\n"
    )
    long = 'def test_a():\n    """One.\n\n    Two.\n    Three.\n    """\n'
    assert problems(PATH, None, long) and not problems(PATH, long, long)
