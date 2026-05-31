from __future__ import annotations

from pathlib import Path

from autopipe.io_utils import classify_failure

from .conftest import make_log_with_tail, write_log


class TestClassifyFailure:
    """Unit tests for classify_failure covering all 15+ error patterns."""

    # ---- OOM patterns -------------------------------------------------------
    def test_oom_cuda(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
        assert classify_failure(p) == "oom"

    def test_oom_system_killer_proximity(self, tmp_path: Path) -> None:
        # "killed" and "out of memory" within 500 chars
        p = write_log(tmp_path, "run.log",
            "Out of memory: Killed process 12345 (python) total-vm:64GB")
        assert classify_failure(p) == "oom"

    def test_oom_system_killer_far_apart_no_match(self, tmp_path: Path) -> None:
        # "killed" and "out of memory" far apart (>500 chars) → should NOT match
        body = "killed\n" + ("padding\n" * 600) + "out of memory"
        p = write_log(tmp_path, "run.log", body)
        assert classify_failure(p) != "oom"

    # ---- loss_scale (checked before oom) ------------------------------------
    def test_loss_scale_minimum(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "loss scale cannot decrease below minimum 1.0")
        assert classify_failure(p) == "loss_scale"

    def test_loss_scale_with_cuda_oom_text_returns_loss_scale(self, tmp_path: Path) -> None:
        # loss_scale takes priority over oom
        p = write_log(tmp_path, "run.log",
            "loss scale minimum reached. Also CUDA out of memory occurred later.")
        assert classify_failure(p) == "loss_scale"

    # ---- NaN patterns -------------------------------------------------------
    def test_nan_loss(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "Training loss is NaN at step 100. Gradient norm: nan")
        assert classify_failure(p) == "nan"

    def test_nan_tensor(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "RuntimeError: tensor contains nan values")
        assert classify_failure(p) == "nan"

    def test_nan_before_assert(self, tmp_path: Path) -> None:
        # NaN should be detected before AssertionError
        p = write_log(tmp_path, "run.log",
            "loss is nan\nAssertionError: loss should be finite")
        assert classify_failure(p) == "nan"

    # ---- disk_full ----------------------------------------------------------
    def test_disk_full_os_message(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "OSError: [Errno 28] No space left on device")
        assert classify_failure(p) == "disk_full"

    def test_disk_full_pytorch_message(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "RuntimeError: file write failed: inline_container")
        assert classify_failure(p) == "disk_full"

    # ---- HuggingFace --------------------------------------------------------
    def test_hf_timeout(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "ConnectionError: huggingface.co timed out after 30s")
        assert classify_failure(p) == "hf"

    def test_hf_rate_limit(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "HTTPError: 429 rate limit exceeded on huggingface.co")
        assert classify_failure(p) == "hf"

    # ---- Network ------------------------------------------------------------
    def test_net_dns(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "socket.gaierror: temporary failure in name resolution")
        assert classify_failure(p) == "net"

    def test_net_connection_reset(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "ConnectionError: connection reset by peer")
        assert classify_failure(p) == "net"

    # ---- Import -------------------------------------------------------------
    def test_import_module_not_found(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "ModuleNotFoundError: No module named 'transformers'")
        assert classify_failure(p) == "import"

    def test_import_importerror(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "ImportError: cannot import name 'AutoModel'")
        assert classify_failure(p) == "import"

    # ---- Port ---------------------------------------------------------------
    def test_port_address_in_use(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "RuntimeError: address already in use")
        assert classify_failure(p) == "port"

    def test_port_eaddrinuse(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "OSError: EADDRINUSE: address already in use 0.0.0.0:29500")
        assert classify_failure(p) == "port"

    # ---- NCCL ---------------------------------------------------------------
    def test_nccl_error(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "nccl error: unhandled system error")
        assert classify_failure(p) == "nccl"

    def test_nccl_timeout(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "nccl timeout: abort signal received")
        assert classify_failure(p) == "nccl"

    # ---- Path ---------------------------------------------------------------
    def test_path_file_not_found(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "FileNotFoundError: No such file or directory: 'data.bin'")
        assert classify_failure(p) == "path"

    # ---- Data ---------------------------------------------------------------
    def test_data_json_decode(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "jsondecodeerror: Expecting value: line 1 column 1")
        assert classify_failure(p) == "data"

    def test_data_keyerror(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "KeyError: 'missing_field' in json data")
        assert classify_failure(p) == "data"

    # ---- Checkpoint ---------------------------------------------------------
    def test_ckpt_missing(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "RuntimeError: checkpoint missing: global_step100")
        assert classify_failure(p) == "ckpt"

    def test_ckpt_size_mismatch(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "checkpoint size mismatch: expected 1024 got 512")
        assert classify_failure(p) == "ckpt"

    # ---- Shape --------------------------------------------------------------
    def test_shape_size_mismatch(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "RuntimeError: size mismatch: mat1 and mat2 shapes cannot be multiplied")
        assert classify_failure(p) == "shape"

    def test_shape_invalid(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "RuntimeError: invalid shape for tensor")
        assert classify_failure(p) == "shape"

    # ---- Assertion ----------------------------------------------------------
    def test_assert_error(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "AssertionError: loss should be positive")
        assert classify_failure(p) == "assert"

    def test_assert_failed(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "assertion failed: x.size(0) == batch_size")
        assert classify_failure(p) == "assert"

    # ---- Killed -------------------------------------------------------------
    def test_killed_sigterm(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "Process terminated by SIGTERM")
        assert classify_failure(p) == "killed"

    def test_killed_keyboard_interrupt(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "Training interrupted: KeyboardInterrupt")
        assert classify_failure(p) == "killed"

    def test_killed_process_killed(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "process killed by signal 9")
        assert classify_failure(p) == "killed"

    # ---- Other / edge cases -------------------------------------------------
    def test_empty_log_returns_other(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log", "")
        assert classify_failure(p) == "other"

    def test_no_match_returns_other(self, tmp_path: Path) -> None:
        p = write_log(tmp_path, "run.log",
            "Some random message that matches nothing specific")
        assert classify_failure(p) == "other"

    def test_missing_log_file(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.log"
        assert classify_failure(p) == "other"

    # ---- Large log scanning (tail + mid) ------------------------------------
    def test_error_in_tail_of_large_log(self, tmp_path: Path) -> None:
        p = make_log_with_tail(tmp_path, "big.log",
            body="Step 100: loss=2.345 no errors here",
            tail="RuntimeError: CUDA out of memory. Tried to allocate 4.00 GiB")
        assert classify_failure(p) == "oom"

    def test_error_in_mid_of_large_log(self, tmp_path: Path) -> None:
        # Write an error in the middle portion of a large log
        p = tmp_path / "big_mid.log"
        header = "header\n" * 100
        error = "RuntimeError: FileNotFoundError: No such file or directory: 'checkpoint/step100'\n" * 50
        footer = "footer\n" * 20000  # make it large so mid region is hit
        p.write_text(header + error + footer, encoding="utf-8")
        assert classify_failure(p) == "path"

    # ---- Warning/INFO noise filtering ---------------------------------------
    def test_warning_line_with_error_keyword_not_filtered(self, tmp_path: Path) -> None:
        # WARNING line containing "error" keyword should NOT be filtered out
        p = write_log(tmp_path, "run.log",
            "[rank0]:WARNING: NCCL error in AllReduce")
        assert classify_failure(p) == "nccl"

    def test_clean_warning_filtered_out(self, tmp_path: Path) -> None:
        # Pure WARNING line without error keywords should be filtered
        p = write_log(tmp_path, "run.log",
            "WARNING: some benign message about checkpoint saving\n"
            "ModuleNotFoundError: No module named 'torch'")
        assert classify_failure(p) == "import"

    # ---- torchrun ChildFailedError noise filtering --------------------------
    def test_child_failed_error_not_misclassified(self, tmp_path: Path) -> None:
        # ChildFailedError alone should not produce false matches
        p = write_log(tmp_path, "run.log",
            "torchrun ChildFailedError: root cause in rank 0\n"
            "RuntimeError: CUDA out of memory")
        assert classify_failure(p) == "oom"
