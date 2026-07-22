import argparse
import contextlib
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import calibrate_train_update as pipeline


APP_DIR = Path(__file__).resolve().parent


class QueueWriter:
    def __init__(self, output_queue):
        self.output_queue = output_queue

    def write(self, text):
        if text:
            self.output_queue.put(("log", text))

    def flush(self):
        pass


class CalibrationGui:
    def __init__(self, root):
        self.root = root
        self.root.title("EMG Calibration")
        self.root.geometry("980x760")
        self.output_queue = queue.Queue()
        self.worker = None
        self.pending_collect_event = None

        defaults = pipeline.build_arg_parser().parse_args([])
        self.vars = {
            "port": tk.StringVar(value=defaults.port),
            "baud": tk.StringVar(value=str(defaults.baud)),
            "seconds": tk.StringVar(value=str(defaults.seconds)),
            "sets": tk.StringVar(value=str(defaults.sets)),
            "settle": tk.StringVar(value=str(defaults.settle)),
            "skip_collect": tk.BooleanVar(value=defaults.skip_collect),
            "max_depth": tk.StringVar(value=str(defaults.max_depth)),
            "split_mode": tk.StringVar(value=defaults.split_mode),
            "random_state": tk.StringVar(value=str(defaults.random_state)),
            "sample_rate": tk.StringVar(value=str(defaults.sample_rate)),
            "window_ms": tk.StringVar(value=str(defaults.window_ms)),
            "step_ms": tk.StringVar(value=str(defaults.step_ms)),
            "ino": tk.StringVar(value=defaults.ino),
            "header": tk.StringVar(value=defaults.header),
            "arduino_cli": tk.StringVar(value=defaults.arduino_cli),
            "fqbn": tk.StringVar(value=defaults.fqbn),
            "sensor_sketch": tk.StringVar(value=defaults.sensor_sketch),
            "classify_sketch": tk.StringVar(value=defaults.classify_sketch or ""),
            "no_upload": tk.BooleanVar(value=defaults.no_upload),
            "no_sensor_upload": tk.BooleanVar(value=defaults.no_sensor_upload),
            "no_classify_upload": tk.BooleanVar(value=defaults.no_classify_upload),
            "upload_wait": tk.StringVar(value=str(defaults.upload_wait)),
        }

        self.status_var = tk.StringVar(value="설정을 확인한 뒤 실행하세요.")
        self._build_ui()
        self.root.after(100, self._drain_queue)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        settings = ttk.Frame(outer)
        settings.pack(fill="x")
        settings.columnconfigure(0, weight=1)
        settings.columnconfigure(1, weight=1)

        self._build_connection_frame(settings).grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        self._build_calibration_frame(settings).grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8))
        self._build_training_frame(settings).grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        self._build_upload_frame(settings).grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8))
        self._build_file_frame(settings).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 8))

        self.start_button = ttk.Button(controls, text="실행", command=self.start)
        self.start_button.pack(side="left")

        self.collect_button = ttk.Button(controls, text="수집 시작", command=self.confirm_collection, state="disabled")
        self.collect_button.pack(side="left", padx=(8, 0))

        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=(12, 0))

        log_frame = ttk.LabelFrame(outer, text="로그")
        log_frame.pack(fill="both", expand=True)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=18, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _build_connection_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="연결")
        self._entry(frame, "포트", "port", 0)
        self._entry(frame, "Baud", "baud", 1)
        self._entry(frame, "Sample rate", "sample_rate", 2)
        return frame

    def _build_calibration_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="캘리브레이션")
        self._entry(frame, "상태별 수집 초", "seconds", 0)
        self._entry(frame, "세트 수", "sets", 1)
        self._entry(frame, "준비 대기 초", "settle", 2)
        ttk.Checkbutton(frame, text="기존 txt로 학습만 진행", variable=self.vars["skip_collect"]).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 8)
        )
        return frame

    def _build_training_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="학습")
        self._entry(frame, "Max depth", "max_depth", 0)
        self._entry(frame, "Window ms", "window_ms", 1)
        self._entry(frame, "Step ms", "step_ms", 2)
        ttk.Label(frame, text="Split").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        split = ttk.Combobox(
            frame,
            textvariable=self.vars["split_mode"],
            values=("sequential", "random"),
            state="readonly",
        )
        split.grid(row=3, column=1, sticky="ew", padx=8, pady=4)
        self._entry(frame, "Random state", "random_state", 4)
        frame.columnconfigure(1, weight=1)
        return frame

    def _build_upload_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="업로드")
        self._entry(frame, "arduino-cli", "arduino_cli", 0)
        self._entry(frame, "FQBN", "fqbn", 1)
        self._entry(frame, "Sensor sketch", "sensor_sketch", 2)
        self._entry(frame, "Classifier sketch", "classify_sketch", 3)
        self._entry(frame, "업로드 후 대기 초", "upload_wait", 4)
        ttk.Checkbutton(frame, text="전체 업로드 안 함", variable=self.vars["no_upload"]).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 0)
        )
        ttk.Checkbutton(frame, text="센서 스케치 업로드 안 함", variable=self.vars["no_sensor_upload"]).grid(
            row=6, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 0)
        )
        ttk.Checkbutton(frame, text="분류 스케치 업로드 안 함", variable=self.vars["no_classify_upload"]).grid(
            row=7, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 8)
        )
        return frame

    def _build_file_frame(self, parent):
        frame = ttk.LabelFrame(parent, text="파일")
        self._entry(frame, "INO", "ino", 0)
        self._entry(frame, "Header", "header", 1)
        return frame

    def _entry(self, parent, label, key, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
        entry = ttk.Entry(parent, textvariable=self.vars[key])
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        parent.columnconfigure(1, weight=1)
        return entry

    def start(self):
        if self.worker and self.worker.is_alive():
            return

        try:
            args = self._build_args()
        except ValueError as exc:
            messagebox.showerror("설정 오류", str(exc))
            return

        self.log_text.delete("1.0", "end")
        self.status_var.set("실행 중...")
        self.start_button.configure(state="disabled")
        self.collect_button.configure(state="disabled")

        self.worker = threading.Thread(target=self._run_pipeline, args=(args,), daemon=True)
        self.worker.start()

    def _build_args(self):
        def as_int(key, label):
            try:
                return int(self.vars[key].get())
            except ValueError as exc:
                raise ValueError(f"{label} 값은 정수여야 합니다.") from exc

        def as_float(key, label):
            try:
                return float(self.vars[key].get())
            except ValueError as exc:
                raise ValueError(f"{label} 값은 숫자여야 합니다.") from exc

        classify_sketch = self.vars["classify_sketch"].get().strip() or None
        return argparse.Namespace(
            port=self.vars["port"].get().strip(),
            baud=as_int("baud", "Baud"),
            seconds=as_float("seconds", "상태별 수집 초"),
            sets=as_int("sets", "세트 수"),
            settle=as_float("settle", "준비 대기 초"),
            skip_collect=self.vars["skip_collect"].get(),
            max_depth=as_int("max_depth", "Max depth"),
            split_mode=self.vars["split_mode"].get(),
            random_state=as_int("random_state", "Random state"),
            sample_rate=as_int("sample_rate", "Sample rate"),
            window_ms=as_int("window_ms", "Window ms"),
            step_ms=as_int("step_ms", "Step ms"),
            ino=self.vars["ino"].get().strip(),
            header=self.vars["header"].get().strip(),
            arduino_cli=self.vars["arduino_cli"].get().strip(),
            fqbn=self.vars["fqbn"].get().strip(),
            sensor_sketch=self.vars["sensor_sketch"].get().strip(),
            classify_sketch=classify_sketch,
            no_upload=self.vars["no_upload"].get(),
            no_sensor_upload=self.vars["no_sensor_upload"].get(),
            no_classify_upload=self.vars["no_classify_upload"].get(),
            upload_wait=as_float("upload_wait", "업로드 후 대기 초"),
        )

    def _run_pipeline(self, args):
        try:
            with contextlib.redirect_stdout(QueueWriter(self.output_queue)), contextlib.redirect_stderr(
                QueueWriter(self.output_queue)
            ):
                pipeline.run_pipeline(args, before_collect=self._wait_for_collection_start)
        except Exception as exc:
            self.output_queue.put(("error", str(exc)))
        else:
            self.output_queue.put(("done", None))

    def _wait_for_collection_start(self, name, set_index, set_count):
        event = threading.Event()
        self.output_queue.put(("collect_prompt", name, set_index, set_count, event))
        event.wait()

    def confirm_collection(self):
        if self.pending_collect_event is None:
            return

        self.pending_collect_event.set()
        self.pending_collect_event = None
        self.collect_button.configure(state="disabled")
        self.status_var.set("수집 중...")

    def _drain_queue(self):
        while True:
            try:
                item = self.output_queue.get_nowait()
            except queue.Empty:
                break

            kind = item[0]
            if kind == "log":
                self._append_log(item[1])
            elif kind == "collect_prompt":
                _kind, name, set_index, set_count, event = item
                self.pending_collect_event = event
                self.status_var.set(f"{set_index}/{set_count}세트: {name} 자세 준비 후 수집 시작")
                self.collect_button.configure(state="normal")
                self._append_log(f"\nPrepare '{name}' ({set_index}/{set_count}), then click 수집 시작.\n")
            elif kind == "error":
                self.status_var.set("오류 발생")
                self.start_button.configure(state="normal")
                self.collect_button.configure(state="disabled")
                self._append_log(f"\nERROR: {item[1]}\n")
                messagebox.showerror("실행 오류", item[1])
            elif kind == "done":
                self.status_var.set("완료")
                self.start_button.configure(state="normal")
                self.collect_button.configure(state="disabled")

        self.root.after(100, self._drain_queue)

    def _append_log(self, text):
        self.log_text.insert("end", text)
        self.log_text.see("end")


def main():
    os.chdir(APP_DIR)
    root = tk.Tk()
    CalibrationGui(root)
    root.mainloop()


if __name__ == "__main__":
    sys.path.insert(0, str(APP_DIR))
    main()
