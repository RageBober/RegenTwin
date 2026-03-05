"""Тесты для AnalysisService и AnalysisTaskManager."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.api.services.analysis_service import AnalysisTaskManager


class TestAnalysisTaskManager:
    def test_register_and_progress(self) -> None:
        tm = AnalysisTaskManager()
        mock_task = MagicMock()
        tm.register("ana-1", mock_task)
        assert tm.get_progress("ana-1") == 0.0

        tm.update_progress("ana-1", 50, 100)
        assert tm.get_progress("ana-1") == 50.0

    def test_cleanup(self) -> None:
        tm = AnalysisTaskManager()
        mock_task = MagicMock()
        tm.register("ana-2", mock_task)
        tm.update_progress("ana-2", 75, 100)

        tm.cleanup("ana-2")
        assert tm.get_progress("ana-2") == 0.0

    def test_nonexistent_progress(self) -> None:
        tm = AnalysisTaskManager()
        assert tm.get_progress("nonexistent") == 0.0
