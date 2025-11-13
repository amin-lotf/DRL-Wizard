from pydantic import BaseModel, Field


class LogConfigSchema(BaseModel):
    segment_steps: int = Field(
        default=50_000, gt=0,
        title="Segment Steps",
        description=(
            "Number of training steps per log segment. "
            "Each segment produces one log file chunk for easier streaming and compression."
        )
    )
    buffer_rows: int = Field(
        default=2, gt=0,
        title="Buffered Log Rows",
        description=(
            "Number of metric rows buffered in memory before writing to disk. "
            "Higher values reduce I/O but increase latency."
        )
    )
    compress: bool = Field(
        default=False,
        title="Compress Logs",
        description=(
            "Enable compression (e.g., Zstandard) for each log segment file. "
            "Saves disk space at a small CPU cost."
        )
    )
    tb_writer: bool = Field(
        default=True,
        title="TensorBoard Writer",
        description=(
            "Enable TensorBoard logging for training metrics and evaluation results."
        )
    )