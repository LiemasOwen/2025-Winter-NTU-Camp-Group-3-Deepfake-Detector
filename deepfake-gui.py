import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

try:
    import cv2
except Exception:
    cv2 = None

import AI

SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif")
COMMON_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg")


def _format_exts(exts: Tuple[str, ...]) -> str:
    return ", ".join(e.upper().lstrip(".") for e in exts)


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _find_model_file() -> Optional[str]:
    env_path = os.environ.get("DEEPFAKE_MODEL_PATH")
    if env_path and (os.path.isfile(env_path) or os.path.isdir(env_path)):
        return env_path

    candidates = [
        "model.keras",
        "model.h5",
        "deepfake_detector.keras",
        "deepfake_detector.h5",
        "deepfake-detector.keras",
        "deepfake-detector.h5",
        "weights.h5",
        "model-weights.h5",
        "deepfake-weights.h5",
        "deepfake_detector_model.keras",
    ]

    base = _script_dir()
    for filename in candidates:
        path = os.path.join(base, filename)
        if os.path.isfile(path) or os.path.isdir(path):
            return path

    for filename in os.listdir(base):
        lower = filename.lower()
        if lower.endswith((".keras", ".h5")):
            path = os.path.join(base, filename)
            if os.path.isfile(path) or os.path.isdir(path):
                return path

    return None


def load_detector_model() -> Tuple[object, str]:
    model_path = _find_model_file()

    if model_path is None:
        return AI.createDeepfakeDetector(), "No model file found. Using untrained model."

    try:
        import tensorflow as tf

        try:
            model = tf.keras.models.load_model(model_path)
            return model, f"Loaded model: {os.path.basename(model_path)}"
        except Exception as e:
            if os.path.isdir(model_path):
                weights_path = os.path.join(model_path, "model.weights.h5")
                if os.path.exists(weights_path):
                    try:
                        model = AI.createDeepfakeDetector()
                        model.load_weights(weights_path)
                        return model, f"Loaded weights from {os.path.basename(model_path)}/model.weights.h5"
                    except Exception:
                        pass
            
            try:
                model = AI.createDeepfakeDetector()
                model.load_weights(model_path)
                return model, f"Loaded weights: {os.path.basename(model_path)}"
            except Exception:
                pass
            
            return AI.createDeepfakeDetector(), f"Model found at {os.path.basename(model_path)} but failed to load. Using untrained model."
    except Exception:
        return AI.createDeepfakeDetector(), "TensorFlow not available. Using untrained model."


def _thumbnail_from_image(path: str, max_size: Tuple[int, int]) -> Optional["ImageTk.PhotoImage"]:
    if Image is None or ImageTk is None:
        return None

    img = Image.open(path)
    img.thumbnail(max_size)
    return ImageTk.PhotoImage(img)


def _thumbnail_from_video(path: str, max_size: Tuple[int, int]) -> Optional["ImageTk.PhotoImage"]:
    if cv2 is None or Image is None or ImageTk is None:
        return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img.thumbnail(max_size)
    return ImageTk.PhotoImage(img)


class DeepfakeApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Deepfake Detector")
        self.minsize(700, 520)

        self.model, self.model_status = load_detector_model()

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.frames: Dict[str, ttk.Frame] = {}
        for FrameCls in (HomePage, UploadPage, ResultPage):
            frame = FrameCls(container, self)
            self.frames[FrameCls.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.current_mode: Optional[str] = None
        self.current_path: Optional[str] = None
        self.current_result: Optional[Dict[str, Any]] = None

        self.show_home()

    def show_home(self) -> None:
        self.current_mode = None
        self.current_path = None
        self.current_result = None
        self._show_frame("HomePage")

    def show_upload(self, mode: str) -> None:
        self.current_mode = mode
        self.current_path = None
        self.current_result = None
        upload: UploadPage = self.frames["UploadPage"]  # type: ignore[assignment]
        upload.configure_for_mode(mode)
        self._show_frame("UploadPage")

    def show_result(self, result: dict) -> None:
        self.current_result = result
        result_page: ResultPage = self.frames["ResultPage"]  # type: ignore[assignment]
        result_page.set_result(result)
        self._show_frame("ResultPage")

    def _show_frame(self, name: str) -> None:
        frame = self.frames[name]
        frame.tkraise()


class HomePage(ttk.Frame):
    def __init__(self, parent: ttk.Frame, controller: DeepfakeApp) -> None:
        super().__init__(parent)
        self.controller = controller

        self.columnconfigure(0, weight=1)

        title = ttk.Label(self, text="Deepfake Detection using CNN", font=("TkDefaultFont", 22, "bold"))
        title.grid(row=0, column=0, pady=(40, 8), padx=20)

        subtitle = ttk.Label(
            self,
            text="By Group 3 of the 2025 Winter NTU Immersion Competition Camp",
            font=("TkDefaultFont", 11),
        )
        subtitle.grid(row=1, column=0, pady=(0, 24), padx=20)

        button_frame = ttk.Frame(self)
        button_frame.grid(row=2, column=0, pady=10)

        img_btn = ttk.Button(button_frame, text="Input Image", command=lambda: controller.show_upload("image"))
        img_btn.grid(row=0, column=0, padx=10, ipadx=10, ipady=6)

        vid_btn = ttk.Button(button_frame, text="Input Video", command=lambda: controller.show_upload("video"))
        vid_btn.grid(row=0, column=1, padx=10, ipadx=10, ipady=6)

        status = ttk.Label(self, text=controller.model_status, font=("TkDefaultFont", 9))
        status.grid(row=3, column=0, pady=(30, 0), padx=20)


class UploadPage(ttk.Frame):
    def __init__(self, parent: ttk.Frame, controller: DeepfakeApp) -> None:
        super().__init__(parent)
        self.controller = controller

        self.mode: str = "image"
        self.supported_exts: Tuple[str, ...] = SUPPORTED_IMAGE_EXTS
        self.filetypes: List[Tuple[str, str]] = []

        self.columnconfigure(0, weight=1)

        banner = ttk.Frame(self)
        banner.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 10))
        banner.columnconfigure(0, weight=1)

        self.banner_title = ttk.Label(banner, text="Upload", font=("TkDefaultFont", 14, "bold"))
        self.banner_title.grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))

        self.banner_types = ttk.Label(banner, text="", font=("TkDefaultFont", 9))
        self.banner_types.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

        self.go_btn = ttk.Button(banner, text="Go", command=self._on_go, state="disabled")
        self.go_btn.grid(row=0, column=1, rowspan=2, sticky="e", padx=8)

        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew", padx=12)
        body.columnconfigure(1, weight=1)

        ttk.Label(body, text="Selected file:").grid(row=0, column=0, sticky="w", pady=(20, 6))

        self.path_var = tk.StringVar(value="")
        self.path_entry = ttk.Entry(body, textvariable=self.path_var, state="readonly")
        self.path_entry.grid(row=0, column=1, sticky="ew", pady=(20, 6), padx=(8, 8))

        browse_btn = ttk.Button(body, text="Browse...", command=self._on_browse)
        browse_btn.grid(row=0, column=2, sticky="e", pady=(20, 6))

        self.status_var = tk.StringVar(value="")
        status = ttk.Label(body, textvariable=self.status_var)
        status.grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))

        home_btn = ttk.Button(self, text="Return to Home", command=controller.show_home)
        home_btn.grid(row=2, column=0, pady=(18, 12))

    def configure_for_mode(self, mode: str) -> None:
        self.mode = mode
        if mode == "video":
            self.supported_exts = tuple()
            self.filetypes = [
                ("Video files", " ".join(f"*{e}" for e in COMMON_VIDEO_EXTS)),
                ("All files", "*.*"),
            ]
            self.banner_title.configure(text="Upload Video")
            self.banner_types.configure(text="Supported video types: Any format supported by OpenCV (commonly: MP4, AVI, MOV, MKV)")
        else:
            self.supported_exts = SUPPORTED_IMAGE_EXTS
            self.filetypes = [
                ("Image files", " ".join(f"*{e}" for e in SUPPORTED_IMAGE_EXTS)),
                ("All files", "*.*"),
            ]
            self.banner_title.configure(text="Upload Image")
            self.banner_types.configure(text=f"Supported image types: {_format_exts(SUPPORTED_IMAGE_EXTS)}")

        self.path_var.set("")
        self.status_var.set("")
        self.go_btn.configure(state="disabled")

    def _on_browse(self) -> None:
        path = filedialog.askopenfilename(title="Select file", filetypes=self.filetypes)
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if self.mode == "image" and ext and ext not in self.supported_exts:
            messagebox.showerror(
                "Unsupported file",
                f"Selected file type ({ext}) is not supported.\n\nSupported: {_format_exts(self.supported_exts)}",
            )
            return

        self.controller.current_path = path
        self.path_var.set(path)
        self.go_btn.configure(state="normal")
        self.status_var.set("")

    def _on_go(self) -> None:
        if not self.controller.current_path:
            return

        self.go_btn.configure(state="disabled")
        self.status_var.set("Running detection... please wait")

        threading.Thread(target=self._run_prediction_thread, daemon=True).start()

    def _run_prediction_thread(self) -> None:
        path = self.controller.current_path
        mode = self.mode

        try:
            if mode == "video":
                prob_fake, label, _frame_probs = AI.predictVideo(self.controller.model, path)
            else:
                prob_fake, label = AI.predictImage(self.controller.model, path)

            prob_fake = float(prob_fake)
            prob_real = float(1.0 - prob_fake)
            verdict = "FAKE" if int(label) == 1 else "REAL"

            result = {
                "mode": mode,
                "path": path,
                "prob_fake": prob_fake,
                "prob_real": prob_real,
                "verdict": verdict,
            }

            self.after(0, lambda: self.controller.show_result(result))
        except Exception as e:
            self.after(0, lambda: self._show_error(e))

    def _show_error(self, err: Exception) -> None:
        self.status_var.set("")
        self.go_btn.configure(state="normal")
        messagebox.showerror("Detection failed", str(err))


class ResultPage(ttk.Frame):
    def __init__(self, parent: ttk.Frame, controller: DeepfakeApp) -> None:
        super().__init__(parent)
        self.controller = controller

        self.columnconfigure(0, weight=1)

        self.title_label = ttk.Label(self, text="Result", font=("TkDefaultFont", 16, "bold"))
        self.title_label.grid(row=0, column=0, pady=(18, 10), padx=20)

        self.thumbnail_label = ttk.Label(self)
        self.thumbnail_label.grid(row=1, column=0, pady=(0, 12), padx=20)

        self.verdict_label = ttk.Label(self, text="", font=("TkDefaultFont", 18, "bold"))
        self.verdict_label.grid(row=2, column=0, pady=(0, 8), padx=20)

        self.prob_label = ttk.Label(self, text="", font=("TkDefaultFont", 11))
        self.prob_label.grid(row=3, column=0, pady=(0, 14), padx=20)

        disclaimer = ttk.Label(
            self,
            text="Disclaimer: This detector is not 100% accurate",
            font=("TkDefaultFont", 9, "italic"),
        )
        disclaimer.grid(row=4, column=0, pady=(14, 8), padx=20)

        home_btn = ttk.Button(self, text="Return to Home", command=controller.show_home)
        home_btn.grid(row=5, column=0, pady=(8, 18))

        self._thumb_ref: Optional["ImageTk.PhotoImage"] = None

    def set_result(self, result: Dict[str, Any]) -> None:
        verdict = result.get("verdict", "")
        prob_fake = float(result.get("prob_fake", 0.0))
        prob_real = float(result.get("prob_real", 0.0))
        mode = result.get("mode")
        path = result.get("path")

        self.verdict_label.configure(text=f"Prediction: {verdict}")
        self.prob_label.configure(
            text=(
                f"Real probability: {prob_real * 100:.2f}%\n"
                f"Fake probability: {prob_fake * 100:.2f}%"
            )
        )

        thumb = None
        if isinstance(path, str) and os.path.isfile(path):
            if mode == "video":
                thumb = _thumbnail_from_video(path, (420, 280))
            else:
                thumb = _thumbnail_from_image(path, (420, 280))

        self._thumb_ref = thumb
        if thumb is None:
            self.thumbnail_label.configure(text="(Thumbnail unavailable)", image="")
        else:
            self.thumbnail_label.configure(text="", image=thumb)


if __name__ == "__main__":
    app = DeepfakeApp()
    app.mainloop()
