from dataclasses import dataclass,field


@dataclass
class LogConfig:
    segment_steps: int = field(default=50000)
    buffer_rows: int = field(default=2)
    compress: bool = field(default=False)
    tb_writer: bool = field(default=True)