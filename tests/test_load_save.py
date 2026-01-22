"""Tests for load_save module."""

import pandas as pd
import pytest

from liss_cleaning.helper_modules.load_save import load_data, save_data


class TestSaveData:
    def test_saves_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.csv"
        save_data(df, path)
        assert path.exists()

    def test_saves_pickle(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.pickle"
        save_data(df, path)
        assert path.exists()

    def test_saves_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.parquet"
        save_data(df, path)
        assert path.exists()

    def test_saves_arrow(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.arrow"
        save_data(df, path)
        assert path.exists()

    def test_raises_for_unsupported_format(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        path = tmp_path / "test.xyz"
        with pytest.raises(ValueError, match="Format .* not supported"):
            save_data(df, path)


class TestLoadData:
    def test_loads_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        result = load_data(path)
        pd.testing.assert_frame_equal(result, df)

    def test_loads_pickle(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.pickle"
        df.to_pickle(path)
        result = load_data(path)
        pd.testing.assert_frame_equal(result, df)

    def test_loads_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)
        result = load_data(path)
        pd.testing.assert_frame_equal(result, df)

    def test_loads_arrow(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.arrow"
        df.to_feather(path)
        result = load_data(path)
        pd.testing.assert_frame_equal(result, df)

    def test_raises_for_unsupported_format(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_text("dummy")
        with pytest.raises(ValueError, match="Format .* not supported"):
            load_data(path)


class TestRoundTrip:
    @pytest.mark.parametrize("extension", [".csv", ".pickle", ".parquet", ".arrow"])
    def test_save_load_roundtrip(self, tmp_path, extension):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        path = tmp_path / f"test{extension}"
        save_data(df, path)
        result = load_data(path)
        pd.testing.assert_frame_equal(result, df)
