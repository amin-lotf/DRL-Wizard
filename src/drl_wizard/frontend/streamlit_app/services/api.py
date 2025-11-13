import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import requests

import httpx

from drl_wizard.common.types import ResultType
from drl_wizard.frontend.streamlit_app.extras import handle_httpx_errors


def flatten_record(obj: dict) -> dict:
    """
    Input record shape:
      {
        "general":{"step":12928},
        "value_loss":{"mean":116.78,...},
        "policy_loss":{"mean":-0.26,...},
        ...
      }
    Output one flat row keyed by metric 'mean' (and step):
      {"step":12928, "value_loss":116.78, "policy_loss":-0.26, ...}
    """
    out = {}
    step = obj.get("general", {}).get("step")
    if step is None:
        return {}
    out["step"] = step
    for k, v in obj.items():
        if k == "general":
            continue
        if isinstance(v, dict):
            # choose what you want to plot; commonly 'mean'
            if "mean" in v:
                out[k] = v["mean"]
    return out



def flatten_env_record(rec: dict) -> dict | None:
    """
    Input example:
      {
        "job_id": 3,
        "data": {
          "general": {"step": 4020},
          "eval_average_episode_rewards": {"mean": -35.3, "count": 1, ...}
        }
      }

    Output example (one row):
      { "job_id": 3, "step": 4020, "eval_average_episode_rewards": -35.3 }
    """
    try:
        job_id = rec["job_id"]
        step = rec["data"]["general"]["step"]
    except Exception:
        return None

    row = {"job_id": job_id, "step": int(step)}
    for k, v in rec.get("data", {}).items():
        if k == "general":
            continue
        # prefer the "mean" if present; skip if can't read
        if isinstance(v, dict) and "mean" in v:
            row[k] = float(v["mean"])
    # If there are no metric columns, skip
    metric_cols = [c for c in row.keys() if c not in ("job_id", "step")]
    return row if metric_cols else None



class Api:
    def __init__(self,base_url,timeout:float=30.0):
        self._client=httpx.Client(base_url=base_url.rstrip("/"),timeout=timeout)

    @handle_httpx_errors
    def get_env_list(self)->List[Dict[str,Any]]:
        r=self._client.get("/training_service/environments")
        r.raise_for_status()
        return r.json()

    @handle_httpx_errors
    def get_algo_list(self)->List[Dict[str,Any]]:
        r=self._client.get("/training_service/algorithms")
        r.raise_for_status()
        return r.json()

    @handle_httpx_errors
    def get_algo_config(self,algo_id:int)->Dict[str,Any]:
        r=self._client.get(f"/training_service/algorithms/{algo_id}/config")
        r.raise_for_status()
        return r.json()

    @handle_httpx_errors
    def get_general_config(self,env_id:str)->Dict[str,Any]:
        r=self._client.get(f"/training_service/environments/{env_id}/general_config")
        r.raise_for_status()
        return r.json()

    @handle_httpx_errors
    def get_log_config(self)->Dict[str,Any]:
        r=self._client.get("/training_service/logs/log_config")
        r.raise_for_status()
        return r.json()

    @handle_httpx_errors
    def get_job_list(self)->List[Dict[str,Any]]:
        r=self._client.get("/training_service/all")
        r.raise_for_status()
        return r.json()

    @handle_httpx_errors
    def start_training(self,payload:Dict[str,Any]) -> Dict[str,Any]:
        r=self._client.post("/training_service/train",json=payload)
        r.raise_for_status()
        return r.json()

    @handle_httpx_errors
    def get_job_status(self,job_id:str) -> Dict[str,Any]:
        r=self._client.get(f"/training_service/{job_id}")
        r.raise_for_status()
        return r.json()

    @handle_httpx_errors
    def stop_job(self,job_id:str):
        r=self._client.patch(f"/training_service/{job_id}/stop")
        r.raise_for_status()
        return True

    @handle_httpx_errors
    def get_job_results(self,job_id:str,result_type:ResultType)->pd.DataFrame:
        url=f"/training_service/{job_id}/results/{result_type.value}/stream"
        rows = []
        with self._client.stream("GET", url, timeout=60.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if isinstance(line, (bytes, bytearray)):
                    line = line.decode("utf-8", "ignore")
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    flat = flatten_record(rec)
                    if flat:
                        rows.append(flat)
                except json.JSONDecodeError:
                    # tolerate partial/keep-alive chunks
                    continue
        # Build dataframe
        df = pd.DataFrame(rows)  # drop empties
        if not df.empty:
            df = df.sort_values("step").drop_duplicates(subset=["step"], keep="last")
        return df

    @handle_httpx_errors
    def get_env_results(self, env_id: str, result_type: ResultType) -> pd.DataFrame:
        url = f"/training_service/environments/{env_id}/results/{result_type.value}/stream"
        rows = []
        with self._client.stream("GET", url, timeout=60.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if isinstance(line, (bytes, bytearray)):
                    line = line.decode("utf-8", "ignore")
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    flat = flatten_env_record(rec)
                    if flat:
                        rows.append(flat)
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # sort + dedup per (job_id, step)
        df = df.sort_values(["job_id", "step"]).drop_duplicates(subset=["job_id", "step"], keep="last")

        # make tidy long form: (job_id, step, metric, value)
        metric_cols = [c for c in df.columns if c not in ("job_id", "step")]
        if metric_cols:
            df = df.melt(
                id_vars=["job_id", "step"],
                value_vars=metric_cols,
                var_name="metric",
                value_name="value",
            ).dropna(subset=["value"])
        return df

    @handle_httpx_errors
    def delete_job(self, job_id: Union[int, str]) -> bool:
        """
        Delete a job by ID. Returns True on success.
        """
        r = self._client.delete(f"/training_service/{job_id}")
        # FastAPI returns 204 No Content on success, but allow 200 too.
        if r.status_code not in (200, 204):
            r.raise_for_status()
        return True

    @handle_httpx_errors
    def download_job_zip(
            self,
            job_id: Union[int, str],
            dest_dir: Optional[Union[str, Path]] = None,
            filename: Optional[str] = None,
            timeout: float = 300.0,
            chunk_size: int = 1024 * 64,
    ) -> Path:
        """
        Download the zipped data for a job (from /training_service/{job_id}/data/zip) and save it to disk.

        Args:
            job_id: The job id.
            dest_dir: Directory to save the file. If None, a secure temporary file is created.
            filename: Optional override for the output filename. If not provided, tries to
                      use 'Content-Disposition' header; falls back to '{job_id}-data.zip'.
            timeout: Request timeout in seconds.
            chunk_size: Streaming chunk size in bytes.

        Returns:
            Path to the saved .zip file.
        """
        url = f"/training_service/{job_id}/data/zip"
        with self._client.stream("GET", url, timeout=timeout) as r:
            r.raise_for_status()

            # Infer filename from Content-Disposition if not supplied
            if filename is None:
                cd = r.headers.get("content-disposition") or r.headers.get("Content-Disposition")
                inferred = None
                if cd:
                    # e.g., 'attachment; filename="123-data.zip"'
                    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
                    if m:
                        inferred = m.group(1)
                filename = inferred or f"{job_id}-data.zip"

            # Choose destination path
            if dest_dir is None:
                # Create a temp file with the requested filename suffix for clarity
                fd, tmp_path = tempfile.mkstemp(prefix=f"{job_id}-", suffix=".zip")
                os.close(fd)  # we'll reopen below
                out_path = Path(tmp_path)
            else:
                dest = Path(dest_dir)
                dest.mkdir(parents=True, exist_ok=True)
                out_path = dest / filename

            # Stream to disk
            with open(out_path, "wb") as f:
                for chunk in r.iter_bytes(chunk_size):
                    if chunk:
                        f.write(chunk)

        return out_path


    
