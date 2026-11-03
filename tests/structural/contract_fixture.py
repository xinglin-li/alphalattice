"""Static contract reader regression examples."""

from pathlib import Path

from tests.structural.contract_support import contracts


def check_cfg(root: Path) -> None:
    """An annotated override is read, including unsupported values, not inherited away."""
    src = root / "contract-fixtures"
    src.mkdir()
    spaced = src / "spaced.py"
    spaced.write_text(
        "from pydantic import BaseModel\nclass Loose (BaseModel):\n    pass\n",
        encoding="utf-8",
    )
    unicode_name = src / "unicode_name.py"
    unicode_name.write_text(
        "from pydantic import BaseModel\nclass \u5408\u540c(BaseModel):\n    pass\n",
        encoding="utf-8",
    )
    rows = contracts(src)
    assert "spaced:Loose:mutable:None" in rows
    assert "unicode_name:\u5408\u540c:mutable:None" in rows
    contracts.cache_clear()
    spaced.unlink()
    unicode_name.unlink()
    module = src / "specimen.py"
    module.write_text(
        "from pydantic import BaseModel, ConfigDict\n"
        'class Parent(BaseModel):\n    model_config = ConfigDict(frozen=True, extra="forbid")\n'
        "class Child(Parent):\n"
        '    model_config: ConfigDict = ConfigDict(frozen=False, extra="allow")\n'
        "class Strict(Parent):\n"
        '    model_config: ConfigDict = ConfigDict(frozen=True, extra="forbid")\n',
        encoding="utf-8",
    )
    rows = contracts(src)
    assert "specimen:Child:mutable:allow" in rows
    assert "specimen:Strict:frozen:forbid" in rows
    module.write_text(
        module.read_text("utf-8").replace(
            'ConfigDict(frozen=False, extra="allow")', "dict(frozen=False)"
        ),
        encoding="utf-8",
    )
    contracts.cache_clear()
    try:
        contracts(src)
    except AssertionError as error:
        assert "specimen.Child" in str(error)
    else:
        raise AssertionError("unsupported annotated config must be refused")
    finally:
        contracts.cache_clear()
    module.write_text(
        "from pydantic import BaseModel, ConfigDict\n"
        "class Child(BaseModel):\n    model_config: ConfigDict = ConfigDict(frozen=unknown)\n",
        encoding="utf-8",
    )
    try:
        contracts(src)
    except (AssertionError, ValueError) as error:
        assert "specimen.Child" in str(error)
    else:
        raise AssertionError("a nonliteral config must be refused by class name")
    finally:
        contracts.cache_clear()
    for change in (
        "model_config.update(frozen=False)",
        'model_config["frozen"] = False',
        'model_config |= {"frozen": False}',
        'del model_config["frozen"]',
        "del model_config",
    ):
        module.write_text(
            "from pydantic import BaseModel, ConfigDict\n"
            'class Parent(BaseModel):\n    model_config = ConfigDict(frozen=True, extra="forbid")\n'
            'class Child(Parent):\n    model_config = ConfigDict(frozen=True, extra="forbid")\n'
            f"    {change}\n",
            encoding="utf-8",
        )
        try:
            contracts(src)
        except AssertionError as error:
            assert "specimen.Child" in str(error)
        else:
            raise AssertionError("mutated config must be refused")
        finally:
            contracts.cache_clear()
