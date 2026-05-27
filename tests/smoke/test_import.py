def test_import_top_level():
    import nichelens_st

    assert hasattr(nichelens_st, "__version__")
    assert isinstance(nichelens_st.__version__, str)
    assert nichelens_st.__version__ != ""


def test_no_module_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    import nichelens_st as m

    importlib.reload(m)
    assert m.__version__ != ""
