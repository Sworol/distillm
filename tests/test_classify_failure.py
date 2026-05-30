"""Unit tests for ``autopipe.io_utils.classify_failure``.

Covers all 15 classification patterns plus priority-ordering regression tests.
Each test writes a canned log snippet to a temporary file, then asserts the
correct failure type — matching the priority chain documented in the source.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from autopipe.io_utils import classify_failure


def _write_log(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Individual pattern tests (one per class)
# ---------------------------------------------------------------------------

def test_loss_scale():
    log = _write_log(
        "[rank0] overflow detected, loss scale minimum reached, cannot decrease further\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: loss scale cannot decrease\n"
    )
    assert classify_failure(log) == "loss_scale"


def test_cuda_oom():
    log = _write_log(
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
        "  (GPU 0; 23.65 GiB total capacity; 21.30 GiB already allocated)\n"
    )
    assert classify_failure(log) == "oom"


def test_system_oom_killer():
    # Kernel OOM message — "killed" and "out of memory" close together.
    log = _write_log(
        "Out of memory: Killed process 45123 (python) total-vm:67108864kB\n"
    )
    assert classify_failure(log) == "oom"


def test_system_oom_killer_proximity():
    # Verify _proximity_match works: keywords must be within 500 chars.
    # Put them 100 chars apart → should match.
    fill = "x" * 100
    log = _write_log(f"killed{fill}out of memory\n")
    assert classify_failure(log) == "oom"


def test_nan_loss():
    log = _write_log(
        "[rank0] loss=nan at step 1234, gradient contains NaN\n"
        "Traceback: ...   AssertionError: tensor is NaN\n"
    )
    assert classify_failure(log) == "nan"


def test_disk_full():
    log = _write_log(
        "OSError: [Errno 28] No space left on device: './checkpoints/step_1000/pytorch_model.bin'\n"
    )
    assert classify_failure(log) == "disk_full"


def test_huggingface_timeout():
    log = _write_log(
        "requests.exceptions.ConnectionError: HTTPSConnectionPool(host='huggingface.co', port=443): "
        "Read timed out. (read timeout=10)\n"
    )
    assert classify_failure(log) == "hf"


def test_network_dns():
    log = _write_log(
        "socket.gaierror: [Errno -3] Temporary failure in name resolution\n"
    )
    assert classify_failure(log) == "net"


def test_network_connection_reset():
    log = _write_log(
        "ConnectionResetError: [Errno 104] Connection reset by peer\n"
    )
    assert classify_failure(log) == "net"


def test_import_error():
    log = _write_log(
        "ModuleNotFoundError: No module named 'tensorboardX'\n"
    )
    assert classify_failure(log) == "import"


def test_port_in_use():
    log = _write_log(
        "RuntimeError: Address already in use. Cannot bind to 0.0.0.0:29500\n"
    )
    assert classify_failure(log) == "port"


def test_nccl_error():
    log = _write_log(
        "[rank1] NCCL error in: ../torch/lib/c10d/ProcessGroupNCCL.cpp:1284, "
        "unhandled system error, NCCL version 2.19.3\n"
    )
    assert classify_failure(log) == "nccl"


def test_nccl_abort():
    log = _write_log(
        "NCCL WARN NET/IB : Got async event, abort the GPU\n"
    )
    assert classify_failure(log) == "nccl"


def test_nccl_timeout():
    log = _write_log(
        "[rank3] NCCL timeout: watch dog triggered, rank 3 did not send in time.\n"
    )
    assert classify_failure(log) == "nccl"


def test_file_not_found():
    log = _write_log(
        "FileNotFoundError: [Errno 2] No such file or directory: 'processed_data/train.bin'\n"
    )
    assert classify_failure(log) == "path"


def test_json_decode_error():
    log = _write_log(
        "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)\n"
    )
    assert classify_failure(log) == "data"


def test_checkpoint_missing():
    log = _write_log(
        "RuntimeError: checkpoint missing: checkpoints/step_500/model.pt\n"
    )
    assert classify_failure(log) == "ckpt"


def test_shape_mismatch():
    log = _write_log(
        "RuntimeError: size mismatch for linear.weight: copying a param with shape "
        "torch.Size([768, 50257]) from model, the shape in current model is "
        "torch.Size([768, 50272]).\n"
    )
    assert classify_failure(log) == "shape"


def test_invalid_shape():
    log = _write_log(
        "RuntimeError: invalid shape for input tensor: [16, 512, 1024]\n"
    )
    assert classify_failure(log) == "shape"


def test_assertion_error():
    log = _write_log(
        "AssertionError: model output shape does not match expected shape\n"
        "  expected: (16, 128, 50257), got: (16, 128, 50272)\n"
    )
    assert classify_failure(log) == "assert"


def test_killed_sigterm():
    log = _write_log(
        "ERROR: torch.distributed.elastic.agent.server.api: Received signal SIGTERM, shutting down\n"
    )
    assert classify_failure(log) == "killed"


def test_killed_keyboard_interrupt():
    log = _write_log(
        "[rank0] KeyboardInterrupt caught, shutting down gracefully\n"
    )
    assert classify_failure(log) == "killed"


def test_unknown():
    log = _write_log(
        "Some random log message without any recognizable error pattern whatsoever.\n"
    )
    assert classify_failure(log) == "other"


# ---------------------------------------------------------------------------
# Priority ordering (the order in the classify_failure chain matters)
# ---------------------------------------------------------------------------

def test_priority_loss_scale_before_oom():
    """loss_scale check MUST fire before OOM — gradient overflow is root cause."""
    log = _write_log(
        "loss scale cannot decrease: overflow detected at iteration 500\n"
        "CUDA out of memory. Tried to allocate ...\n"
    )
    assert classify_failure(log) == "loss_scale"


def test_priority_nan_before_assert():
    """NaN check MUST fire before AssertionError — NaN is the root cause."""
    log = _write_log(
        "loss is nan at step 2000, gradient is nan\n"
        "AssertionError: tensor check failed\n"
    )
    assert classify_failure(log) == "nan"


def test_priority_oom_before_killed():
    """CUDA OOM is more actionable than a generic SIGTERM/killed."""
    log = _write_log(
        "torch.cuda.OutOfMemoryError: CUDA out of memory ...\n"
        "[rank0] SIGTERM signal received.\n"
    )
    assert classify_failure(log) == "oom"


def test_warning_lines_filtered():
    """WARNING/INFO lines should NOT cause false positives."""
    log = _write_log(
        "WARNING: No space left on temporary device (ignored)\n"
        "[rank0] INFO: disk full check passed, proceeding\n"
        "Training completed successfully.\n"
    )
    assert classify_failure(log) == "other"


def test_warning_filtered_cuda_oom_still_detected():
    """WARNING about disk should not mask real CUDA OOM elsewhere."""
    log = _write_log(
        "[rank1] WARNING: disk usage at 90%, some files may fail\n"
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...\n"
    )
    assert classify_failure(log) == "oom"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_file():
    log = _write_log("")
    assert classify_failure(log) == "other"


def test_large_log_error_in_middle(tmp_path: Path):
    """Error in the middle of a big file (not just the tail) should be found.

    Uses a file large enough to exercise mid+tail scanning (~600 KB) but places
    the error near the 65 % position so it lands inside the mid-scan window.
    """
    prefix = "Training log line\n" * 20000  # ~380 KB before error
    error = "torch.cuda.OutOfMemoryError: CUDA out of memory ...\n"
    suffix = "Cleanup log line\n" * 10000   # ~190 KB after error
    log_file = tmp_path / "big.log"
    log_file.write_text(prefix + error + suffix, encoding="utf-8")
    # file_size ≈ 570 KB, error at ~380 KB ≈ 66 % — within the 65 % mid window.
    assert classify_failure(log_file) == "oom"


def test_mid_low_coverage(tmp_path: Path):
    """Error at 50 % of a large log — caught by the 25 % low window.

    Previously this was a known coverage gap (mid at 65 % + tail missed it).
    The third window at 25 % closes it for files up to ~1.5 MB.
    """
    prefix = "Training log line\n" * 20000   # ~380 KB
    error = "torch.cuda.OutOfMemoryError: CUDA out of memory ...\n"
    suffix = "Cleanup log line\n" * 20000     # ~380 KB
    log_file = tmp_path / "big.log"
    log_file.write_text(prefix + error + suffix, encoding="utf-8")
    assert classify_failure(log_file) == "oom"
