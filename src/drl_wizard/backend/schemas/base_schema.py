from datetime import datetime, timezone

from pydantic import BaseModel


class GlobalModel(BaseModel):
    model_config = {
        "from_attributes": True,
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.astimezone(timezone.utc).isoformat() if v else None
        }
    }