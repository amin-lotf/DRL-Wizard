import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drl_wizard.backend.services.logging.json_logger import SegmentedJsonlLogger
from drl_wizard.backend.services.logging.log_manifest import Manifest
from drl_wizard.common.types import ResultType
from drl_wizard.configs.app_cfg import AppConfig
from drl_wizard.configs.log_cfg import LogConfig


class DummyTrainingService:
    async def add_job_results(self, job_id: int, result_type: ResultType, segment_steps: int, manifest_path: str):
        return None


def test_logger_writes_first_segment_under_value_named_directory(tmp_path: Path):
    app_cfg = AppConfig(
        run_dir=tmp_path,
        log_cfg=LogConfig(buffer_rows=1, segment_steps=100, compress=False, tb_writer=False),
    )
    logger = SegmentedJsonlLogger(
        svc=DummyTrainingService(),
        app_cfg=app_cfg,
        save_dir=app_cfg.run_dir,
        job_id=3,
    )

    assert (tmp_path / "3" / "log" / "train").is_dir()
    assert not (tmp_path / "3" / "log" / "ResultType.TRAIN").exists()

    logger.log_data({"reward": 1.0}, step=0, log_type=ResultType.TRAIN)
    logger.close()

    segment_path = tmp_path / "3" / "log" / "train" / "part-00001.jsonl"
    assert segment_path.exists()

    manifest = Manifest.load(tmp_path / "3" / "manifest.json")
    assert manifest.segments[ResultType.TRAIN][0].path == "train/part-00001.jsonl"
