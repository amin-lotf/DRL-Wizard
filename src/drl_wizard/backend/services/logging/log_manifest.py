from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import json, tempfile

from drl_wizard.backend.services.utils import json_default
from drl_wizard.common.types import ResultType, ConfigType  # e.g., Enum('ResultType', 'TRAIN EVALUATE')





@dataclass
class Segment:
    path: str  # "train/part-00001.jsonl[.zst]"
    start: int  # inclusive
    end: int  # inclusive

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError("segment.end must be >= segment.start")



@dataclass
class Manifest:
    job_id: int
    path: Path  # absolute path to manifest.json
    log_path: Path
    configs_path: Path
    checkpoints_path: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int = 1
    segments: Dict[ResultType, List[Segment]] = field(
        default_factory=lambda: {ResultType.TRAIN: [], ResultType.EVALUATE: []}
    )


    # -------- IO --------
    @classmethod
    def load(cls, manifest_path: Path) -> "Manifest":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        # normalize created_at
        ca = data.get("created_at")
        if isinstance(ca, str):
            data["created_at"] = datetime.fromisoformat(ca.replace("Z", "+00:00"))

        # segments keys may be strings ("train"/"evaluate") from old files
        raw_segments = data.get("segments", {})
        segs: Dict[ResultType, List[Segment]] = {ResultType.TRAIN: [], ResultType.EVALUATE: []}
        for k, v in raw_segments.items():
            key = k
            if isinstance(k, str):
                # accept either enum value or name
                try:
                    key = ResultType(k)  # value
                except Exception:
                    key = ResultType[k.upper()]  # name
            items = [Segment(**s) if isinstance(s, dict) else s for s in v]
            segs[key] = items

        return cls(
            job_id=data["job_id"],
            path=Path(data["path"]),
            log_path=Path(data["log_path"]),
            configs_path=Path(data["configs_path"]),
            checkpoints_path=Path(data["checkpoints_path"]),
            created_at=data.get("created_at", datetime.now(timezone.utc)),
            schema_version=int(data.get("schema_version", 1)),
            segments=segs
        )

    def atomic_write(self) -> None:
        """Atomically write JSON to self.path."""
        dest = Path(self.path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        body = asdict(self)
        # convert Enum keys → string values for JSON
        body["segments"] = {
            k.value if hasattr(k, "value") else str(k): [asdict(s) for s in v]
            for k, v in self.segments.items()
        }
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(dest.parent)) as tmp:
            json.dump(body, tmp, ensure_ascii=False, indent=2, default=json_default)
            tmp_path = Path(tmp.name)
        tmp_path.replace(dest)

    # -------- segment helpers --------
    def last_segment(self, kind: ResultType) -> Optional[Segment]:
        segs = self.segments.get(kind, [])
        return segs[-1] if segs else None


    def ensure_segment_for_step(
            self,
            kind: ResultType,
            step: int,
            segment_steps: int,
            base_dir: Path,
            compressed: bool,
    ) -> Path:
        """Return absolute path to current segment file for `kind`/`step`,
        rotating when `step` crosses the boundary."""
        segs = self.segments[kind]
        ext = ".jsonl.zst" if compressed else ".jsonl"

        if not segs:
            start = step - (step % segment_steps)
            seg_index = 1
            rel = f"{kind.value}/part-{seg_index:05d}{ext}"
            segs.append(Segment(path=rel, start=start, end=step))
            self.atomic_write()
            return base_dir / rel

        last = segs[-1]
        last_boundary_end = last.start + segment_steps - 1
        if step > last_boundary_end:
            # rotate
            start = step - (step % segment_steps)
            seg_index = len(segs) + 1
            rel = f"{kind.value}/part-{seg_index:05d}{ext}"
            segs.append(Segment(path=rel, start=start, end=step))
            self.atomic_write()
            return base_dir / rel

        # continue current
        last.end = max(last.end, step)
        self.atomic_write()
        return base_dir / last.path
