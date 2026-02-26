import os
import logging
import json
import io
import time
import zipfile
import uuid
from typing import Optional, Any, Dict, List, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self, use_miner_u: bool = True):
        self.use_miner_u = use_miner_u

    def parse(self, pdf_path: str) -> str:
        """
        Parse a PDF file and return its content as Markdown.
        """
        if not self._is_url(pdf_path) and not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if self.use_miner_u:
            try:
                return self._parse_with_mineru(pdf_path)
            except ImportError:
                logger.warning("MinerU (magic-pdf) not installed or failed to import. Falling back to PyMuPDF.")
            except Exception as e:
                logger.error(f"MinerU parsing failed: {e}. Falling back to PyMuPDF.")

        if self._is_url(pdf_path):
            raise ValueError("Cannot parse PDF URL without MinerU. Please set MINERU_API_TOKEN or install magic-pdf.")

        return self._parse_with_pymupdf(pdf_path)

    def _parse_with_mineru(self, pdf_path: str) -> str:
        if self._is_url(pdf_path):
            token = os.getenv("MINERU_API_TOKEN") or os.getenv("MINERU_TOKEN")
            if not token:
                raise ValueError("MINERU_API_TOKEN not set for MinerU API parsing.")
            return self._parse_with_mineru_api(pdf_url=pdf_path, token=token)

        token = os.getenv("MINERU_API_TOKEN") or os.getenv("MINERU_TOKEN")
        if token and self._env_truthy("MINERU_API_UPLOAD_LOCAL", default=False):
            return self._parse_with_mineru_upload_api(file_path=pdf_path, token=token)

        try:
            from magic_pdf.pipe.UNIPipe import UNIPipe
        except Exception as e:
            if token:
                return self._parse_with_mineru_upload_api(file_path=pdf_path, token=token)
            raise ImportError("magic-pdf not installed or failed to import") from e

        pipe = UNIPipe(pdf_path)
        pipe.pipe_classify()
        pipe.pipe_parse()
        return pipe.get_markdown()

    def _parse_with_mineru_api(self, pdf_url: str, token: str) -> str:
        base_url = os.getenv("MINERU_API_BASE_URL", "https://mineru.net/api/v4").rstrip("/")
        model_version = os.getenv("MINERU_MODEL_VERSION", "vlm")
        timeout_s = self._env_int("MINERU_API_TIMEOUT_S", default=600)
        poll_interval_s = self._env_int("MINERU_API_POLL_INTERVAL_S", default=2)

        payload = {"url": pdf_url, "model_version": model_version}
        task = self._mineru_post_json(path="/extract/task", payload=payload, token=token, base_url=base_url)
        task_id = self._get_str(task, ["data", "task_id"])
        if not task_id:
            raise RuntimeError(f"MinerU API create task response missing task_id: {task}")

        started = time.time()
        while True:
            if time.time() - started > timeout_s:
                raise TimeoutError(f"MinerU API task timed out: {task_id}")

            result = self._mineru_get_json(path=f"/extract/task/{task_id}", token=token, base_url=base_url)
            state = self._get_str(result, ["data", "state"]) or ""
            if state in ("pending", "running", "converting"):
                time.sleep(poll_interval_s)
                continue

            if state == "failed":
                err_msg = self._get_str(result, ["data", "err_msg"]) or "unknown"
                raise RuntimeError(f"MinerU API task failed: {err_msg}")

            if state == "done":
                zip_url = self._get_str(result, ["data", "full_zip_url"])
                if not zip_url:
                    extracted = self._extract_markdown_from_mineru_response(result)
                    if extracted:
                        return extracted
                    raise RuntimeError(f"MinerU API done but missing full_zip_url: {result}")
                return self._download_zip_and_extract_markdown(zip_url)

            extracted = self._extract_markdown_from_mineru_response(result)
            if extracted:
                return extracted
            raise RuntimeError(f"MinerU API returned unexpected task state: {state}, payload: {result}")

    def _parse_with_mineru_upload_api(self, file_path: str, token: str) -> str:
        base_url = os.getenv("MINERU_API_BASE_URL", "https://mineru.net/api/v4").rstrip("/")
        model_version = os.getenv("MINERU_MODEL_VERSION", "vlm")
        timeout_s = self._env_int("MINERU_API_TIMEOUT_S", default=600)
        poll_interval_s = self._env_int("MINERU_API_POLL_INTERVAL_S", default=2)

        file_name = os.path.basename(file_path)
        data_id = self._make_data_id(file_name)
        payload: Dict[str, Any] = {"files": [{"name": file_name, "data_id": data_id}], "model_version": model_version}
        apply_resp = self._mineru_post_json(path="/file-urls/batch", payload=payload, token=token, base_url=base_url)
        batch_id = self._get_str(apply_resp, ["data", "batch_id"])
        file_urls = self._get_list(apply_resp, ["data", "file_urls"])
        if not batch_id or not file_urls:
            raise RuntimeError(f"MinerU API upload url response missing batch_id/file_urls: {apply_resp}")
        upload_url = str(file_urls[0])

        self._upload_file_put(upload_url=upload_url, file_path=file_path)

        started = time.time()
        while True:
            if time.time() - started > timeout_s:
                raise TimeoutError(f"MinerU API batch timed out: {batch_id}")

            results = self._mineru_get_json(path=f"/extract-results/batch/{batch_id}", token=token, base_url=base_url)
            items = self._get_list(results, ["data", "extract_result"]) or []
            target = None
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("file_name") == file_name or item.get("data_id") == data_id:
                    target = item
                    break

            if not target:
                time.sleep(poll_interval_s)
                continue

            state = str(target.get("state") or "")
            if state in ("waiting-file", "pending", "running", "converting"):
                time.sleep(poll_interval_s)
                continue

            if state == "failed":
                err_msg = str(target.get("err_msg") or "unknown")
                raise RuntimeError(f"MinerU API batch task failed: {err_msg}")

            if state == "done":
                zip_url = target.get("full_zip_url")
                if isinstance(zip_url, str) and zip_url.startswith(("http://", "https://")):
                    return self._download_zip_and_extract_markdown(zip_url)
                raise RuntimeError(f"MinerU API batch done but missing full_zip_url: {target}")

            raise RuntimeError(f"MinerU API batch returned unexpected state: {state}, payload: {target}")

    def _download_zip_and_extract_markdown(self, zip_url: str) -> str:
        req = Request(zip_url, method="GET")
        with urlopen(req, timeout=120) as resp:
            zip_bytes = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            md_name = self._pick_markdown_file(zf.namelist())
            if not md_name:
                names = zf.namelist()[:50]
                raise RuntimeError(f"MinerU result zip did not contain markdown file. Entries: {names}")
            content_bytes = zf.read(md_name)
            return content_bytes.decode("utf-8", errors="replace")

    def _pick_markdown_file(self, names: List[str]) -> Optional[str]:
        candidates = [n for n in names if isinstance(n, str) and n.lower().endswith(".md")]
        if not candidates:
            return None

        def score(n: str) -> Tuple[int, int, int, str]:
            low = n.lower()
            prefer_full = 0 if (low.endswith("/full.md") or low.endswith("full.md") or "/full_" in low) else 1
            prefer_root = 0 if ("/" not in n.strip("/")) else 1
            length = len(n)
            return (prefer_full, prefer_root, length, n)

        return sorted(candidates, key=score)[0]

    def _upload_file_put(self, upload_url: str, file_path: str) -> None:
        try:
            import requests

            with open(file_path, "rb") as f:
                resp = requests.put(upload_url, data=f, timeout=120)
            if resp.status_code != 200:
                raise RuntimeError(f"Upload failed: HTTP {resp.status_code}, body: {resp.text[:500]}")
            return
        except Exception:
            pass

        with open(file_path, "rb") as f:
            data = f.read()
        req = Request(upload_url, method="PUT", data=data)
        with urlopen(req, timeout=120) as resp:
            if getattr(resp, "status", 200) != 200:
                raise RuntimeError(f"Upload failed: HTTP {getattr(resp, 'status', 'unknown')}")

    def _mineru_post_json(self, path: str, payload: Dict[str, Any], token: str, base_url: str) -> Dict[str, Any]:
        endpoint = f"{base_url}{path}"
        req = Request(
            endpoint,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            data=json.dumps(payload).encode("utf-8"),
        )
        raw = self._urlopen_read_text(req)
        data = self._parse_json(raw)
        if isinstance(data, dict) and data.get("code") not in (None, 0):
            raise RuntimeError(f"MinerU API error: {data}")
        if not isinstance(data, dict):
            raise RuntimeError(f"MinerU API returned unexpected payload: {data}")
        return data

    def _mineru_get_json(self, path: str, token: str, base_url: str) -> Dict[str, Any]:
        endpoint = f"{base_url}{path}"
        req = Request(endpoint, method="GET", headers={"Authorization": f"Bearer {token}"})
        raw = self._urlopen_read_text(req)
        data = self._parse_json(raw)
        if isinstance(data, dict) and data.get("code") not in (None, 0):
            raise RuntimeError(f"MinerU API error: {data}")
        if not isinstance(data, dict):
            raise RuntimeError(f"MinerU API returned unexpected payload: {data}")
        return data

    def _urlopen_read_text(self, req: Request) -> str:
        try:
            with urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(f"MinerU API HTTPError {e.code}: {detail or str(e)}") from e
        except URLError as e:
            raise RuntimeError(f"MinerU API request failed: {e}") from e

    def _parse_json(self, raw: str) -> Any:
        try:
            return json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"MinerU API returned non-JSON response: {raw[:500]}") from e

    def _make_data_id(self, file_name: str) -> str:
        base = os.path.splitext(os.path.basename(file_name))[0]
        base = base.replace(" ", "_")
        cleaned = []
        for ch in base:
            if ch.isalnum() or ch in ("_", "-", "."):
                cleaned.append(ch)
        prefix = "".join(cleaned) or "file"
        suffix = uuid.uuid4().hex[:8]
        data_id = f"{prefix}-{suffix}"
        return data_id[:128]

    def _env_truthy(self, key: str, default: bool = False) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

    def _env_int(self, key: str, default: int) -> int:
        val = os.getenv(key)
        if val is None:
            return default
        try:
            return int(str(val).strip())
        except Exception:
            return default

    def _get_str(self, obj: Any, path: List[str]) -> Optional[str]:
        cur = obj
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur if isinstance(cur, str) else None

    def _get_list(self, obj: Any, path: List[str]) -> Optional[List[Any]]:
        cur = obj
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur if isinstance(cur, list) else None

    def _extract_markdown_from_mineru_response(self, payload: Any) -> Optional[str]:
        if isinstance(payload, str):
            return payload

        if not isinstance(payload, dict):
            return None

        data = payload.get("data", payload)
        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            for key in ("markdown", "md", "content", "result", "text"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val

            for key in ("markdown_url", "md_url", "result_url", "content_url"):
                url_val = data.get(key)
                if isinstance(url_val, str) and url_val.startswith(("http://", "https://")):
                    return self._fetch_text_url(url_val)

        return None

    def _fetch_text_url(self, url: str) -> str:
        req = Request(url, method="GET")
        with urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def _is_url(self, value: str) -> bool:
        return value.startswith(("http://", "https://"))

    def _parse_with_pymupdf(self, pdf_path: str) -> str:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return "Error: PyMuPDF (fitz) not installed. Please install pymupdf."

        doc = fitz.open(pdf_path)
        markdown_content = ""
        
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            markdown_content += f"## Page {page_num + 1}\n\n{text}\n\n"
            
        return markdown_content

if __name__ == "__main__":
    # Test
    parser = PDFParser(use_miner_u=False)
    # print(parser.parse("path/to/test.pdf"))
