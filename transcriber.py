import sys, re, threading, json, datetime, pathlib, numpy as np, sounddevice as sd
from faster_whisper import WhisperModel

APP_VERSION = "1.0.0"
MODEL_NAME = "large-v3-turbo"
TELEGRAM_URL = "https://t.me/+CB7yA0PY32U5Y2M6"

HISTORY_PATH = pathlib.Path.home() / "transcriber-history.jsonl"
ARCHIVE_PATH = pathlib.Path.home() / "transcriber-history-archive.jsonl"

# Известные галлюцинации Whisper на русском (выучены из YouTube-субтитров).
# Сегмент выкидывается только при ПОЛНОМ совпадении — легитимная речь,
# где эти слова встречаются внутри фразы, не страдает.
HALLUCINATION_SEGMENTS = re.compile(
    r"^\s*("
    r"субтитры\s+(с|по)?дел[ао]л[аи]?\s+\S+"
    r"|субтитры\s+создавал\s+\S+"
    r"|dimatorzok|дима\s+торжок"
    r"|продолжение\s+следует\W*"
    r"|спасибо\s+за\s+просмотр\W*"
    r"|редактор\s+субтитров\s+.*"
    r"|корректор\s+а\.\s*\S+"
    r")\s*$",
    re.IGNORECASE,
)
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                               QPushButton, QTextEdit, QComboBox, QLabel, QHBoxLayout,
                               QTabWidget, QListWidget, QMenu, QMessageBox)
from PySide6.QtCore import Signal, QObject, Qt

class Signals(QObject):
    text_ready = Signal(str)
    status_update = Signal(str)
    history_added = Signal(dict)

class Transcriber(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Whisper Transcriber")
        self.setMinimumSize(600, 400)
        self.recording = False
        self.audio_data = []
        self.signals = Signals()
        self.signals.text_ready.connect(self.append_text)
        self.signals.status_update.connect(self.update_status)
        self.signals.history_added.connect(self.add_history_item)
        self.model = None
        self.init_ui()
        threading.Thread(target=self.load_model, daemon=True).start()

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        central = QWidget()
        self.tabs.addTab(central, "Диктовка")
        layout = QVBoxLayout(central)
        top = QHBoxLayout()
        self.device_combo = QComboBox()
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0 and d['hostapi'] == 0:
                self.device_combo.addItem(d['name'], i)
        top.addWidget(QLabel("Microphone:"))
        top.addWidget(self.device_combo, 1)
        layout.addLayout(top)
        self.status_label = QLabel("Loading model... Please wait")
        layout.addWidget(self.status_label)
        self.record_btn = QPushButton("RECORD")
        self.record_btn.setEnabled(False)
        self.record_btn.setStyleSheet("font-size:18px;padding:10px;background:#cc3333;color:white;")
        self.record_btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_btn)
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Transcription will appear here...")
        layout.addWidget(self.text_edit)
        btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy to clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.text_edit.toPlainText()))
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.text_edit.clear)
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)
        hist = QWidget()
        self.tabs.addTab(hist, "История")
        hist_layout = QVBoxLayout(hist)
        self.history_list = QListWidget()
        self.history_list.currentRowChanged.connect(self.show_history_entry)
        hist_layout.addWidget(self.history_list, 1)
        self.history_view = QTextEdit()
        self.history_view.setReadOnly(True)
        self.history_view.setPlaceholderText("Click an entry above to see the full text...")
        hist_layout.addWidget(self.history_view, 1)
        hist_btn_row = QHBoxLayout()
        hist_copy_btn = QPushButton("Copy to clipboard")
        hist_copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.history_view.toPlainText()))
        hist_btn_row.addWidget(hist_copy_btn)
        clear_hist_btn = QPushButton("Очистить историю…")
        clear_menu = QMenu(clear_hist_btn)
        clear_menu.addAction("Оставить последние 7 дней", lambda: self.clear_history(keep_days=7))
        clear_menu.addAction("Оставить последние 30 дней", lambda: self.clear_history(keep_days=30))
        clear_menu.addSeparator()
        clear_menu.addAction("Удалить всё", lambda: self.clear_history(keep_days=None))
        clear_hist_btn.setMenu(clear_menu)
        hist_btn_row.addWidget(clear_hist_btn)
        hist_layout.addLayout(hist_btn_row)
        self.history_entries = []  # новые первыми, индексы совпадают со строками списка
        self.load_history()
        about = QWidget()
        self.tabs.addTab(about, "О программе")
        about_layout = QVBoxLayout(about)
        about_label = QLabel(
            f"<h2>Local Whisper Transcriber</h2>"
            f"<p>Версия {APP_VERSION} · модель {MODEL_NAME} (faster-whisper)</p>"
            f"<p>Голос распознаётся полностью на вашем компьютере.<br>"
            f"Ничего не отправляется в облако, работает без интернета.</p>"
            f"<p>Автор: Илья Гужов — про AI и автоматизацию:<br>"
            f"<a href='{TELEGRAM_URL}'>Telegram-канал</a></p>"
        )
        about_label.setOpenExternalLinks(True)
        about_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        about_label.setWordWrap(True)
        about_layout.addWidget(about_label)

    def load_model(self):
        try:
            self.signals.status_update.emit(f"Loading {MODEL_NAME} on GPU...")
            self.model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
            self.signals.status_update.emit(f"Model ready (GPU, {MODEL_NAME})! Press RECORD.")
        except Exception as e:
            self.signals.status_update.emit(f"GPU failed ({e}), loading on CPU...")
            self.model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
            self.signals.status_update.emit(f"Model ready (CPU, {MODEL_NAME}). Press RECORD.")
        self.record_btn.setEnabled(True)

    def update_status(self, text):
        self.status_label.setText(text)

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.record_btn.setText("STOP")
            self.record_btn.setStyleSheet("font-size:18px;padding:10px;background:#33aa33;color:white;")
            self.audio_data = []
            dev_idx = self.device_combo.currentData()
            self.stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32',
                                          device=dev_idx, callback=self.audio_callback)
            self.stream.start()
            self.signals.status_update.emit("Recording...")
        else:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            self.record_btn.setText("RECORD")
            self.record_btn.setStyleSheet("font-size:18px;padding:10px;background:#cc3333;color:white;")
            self.signals.status_update.emit("Transcribing...")
            threading.Thread(target=self.transcribe, daemon=True).start()

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    def transcribe(self):
        audio = np.concatenate(self.audio_data, axis=0).flatten()
        segments, _ = self.model.transcribe(
            audio, language="ru", beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            temperature=0,
            hotwords="Claude, Claude Code, API, Anthropic, skill, workspace",
        )
        text = " ".join(
            t for s in segments
            if not HALLUCINATION_SEGMENTS.match(t := s.text.strip())
        )
        self.save_history(text)
        self.signals.text_ready.emit(text + "\n")
        self.signals.status_update.emit("Done! Press RECORD for next.")

    def save_history(self, text):
        if not text.strip():
            return
        try:
            entry = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "text": text}
            with open(HISTORY_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.signals.history_added.emit(entry)
        except Exception:
            pass  # история не должна ронять транскрипцию

    def load_history(self):
        try:
            with open(HISTORY_PATH, encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.history_entries.insert(0, entry)
            self.history_list.insertItem(0, self.history_label(entry))

    def clear_history(self, keep_days=None):
        if not self.history_entries:
            return
        cutoff = None
        if keep_days is not None:
            cutoff = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        keep, drop = [], []
        for e in self.history_entries:  # порядок: новые первыми
            try:
                ts = datetime.datetime.fromisoformat(e.get("ts", ""))
            except ValueError:
                ts = None
            if cutoff is not None and ts is not None and ts >= cutoff:
                keep.append(e)
            else:
                drop.append(e)
        if not drop:
            QMessageBox.information(self, "История", "Нечего удалять: все записи свежее выбранного срока.")
            return
        answer = QMessageBox.question(
            self, "Очистить историю",
            f"Убрать записей: {len(drop)} (останется {len(keep)}).\n"
            f"Они будут перенесены в архивный файл:\n{ARCHIVE_PATH}",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            with open(ARCHIVE_PATH, "a", encoding="utf-8") as f:
                for e in reversed(drop):  # в архив хронологически, старые первыми
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                for e in reversed(keep):
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        except OSError as err:
            QMessageBox.warning(self, "История", f"Не удалось очистить: {err}")
            return
        self.history_entries = keep
        self.history_list.clear()
        for e in keep:  # keep уже новые-первыми, addItem сохраняет порядок
            self.history_list.addItem(self.history_label(e))
        self.history_view.clear()

    def add_history_item(self, entry):
        self.history_entries.insert(0, entry)
        self.history_list.insertItem(0, self.history_label(entry))

    def show_history_entry(self, row):
        if 0 <= row < len(self.history_entries):
            self.history_view.setPlainText(self.history_entries[row]["text"])

    @staticmethod
    def history_label(entry):
        ts = entry.get("ts", "")[:16].replace("T", " ")
        preview = entry.get("text", "").strip().replace("\n", " ")
        if len(preview) > 70:
            preview = preview[:70] + "…"
        return f"{ts} · {preview}"

    def append_text(self, text):
        self.text_edit.append(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Transcriber()
    w.show()
    sys.exit(app.exec())
