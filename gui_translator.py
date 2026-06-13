import sys
import os
import threading
import json
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate

# ---------- Логирование в файл ----------
LOG_FILE = "app.log"

def log_to_file(message, level="INFO"):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{level}] {message}\n")

def log_exception(exc_type, exc_value, exc_traceback):
    log_to_file("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)), "CRITICAL")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = log_exception

# ---------- Кэширование модели Whisper ----------
_whisper_models = {}

def get_whisper_model(model_size):
    if model_size not in _whisper_models:
        log_to_file(f"Загрузка модели Whisper: {model_size} на CPU")
        _whisper_models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _whisper_models[model_size]

# ---------- Пакет перевода ----------
def ensure_translation_package():
    try:
        argostranslate.package.update_package_index()
        installed = argostranslate.package.get_installed_packages()
        if any(p.from_code == "en" and p.to_code == "ru" for p in installed):
            return
        available = argostranslate.package.get_available_packages()
        pkg = next(p for p in available if p.from_code == "en" and p.to_code == "ru")
        pkg.download()
        argostranslate.package.install_from_path(pkg.download())
        log_to_file("Пакет перевода en->ru установлен")
    except Exception as e:
        log_to_file(f"Ошибка установки пакета перевода: {e}", "ERROR")
        raise

def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def transcribe_and_translate(input_path, model_size, progress_callback, done_callback, cancel_event, beam_size=1):
    try:
        progress_callback("Загружаем Whisper...", 0)
        model = get_whisper_model(model_size)
        if cancel_event.is_set():
            done_callback(False, "Отменено.", None)
            return

        progress_callback("Распознаём и переводим...", 5)
        segments, _ = model.transcribe(
            input_path,
            language="en",
            beam_size=beam_size,
            best_of=1,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        seg_list = []
        for seg in segments:
            if cancel_event.is_set():
                done_callback(False, "Отменено.", None)
                return
            seg_list.append(seg)
        total = len(seg_list)
        if total == 0:
            done_callback(False, "Нет распознанных фраз.", None)
            return

        out_srt = os.path.splitext(input_path)[0] + "_ru.srt"
        with open(out_srt, "w", encoding="utf-8") as f:
            idx = 1
            for i, segment in enumerate(seg_list):
                if cancel_event.is_set():
                    f.close()
                    os.remove(out_srt)
                    done_callback(False, "Отменено.", None)
                    return
                text_en = segment.text.strip()
                if not text_en:
                    continue
                text_ru = argostranslate.translate.translate(text_en, "en", "ru")
                f.write(f"{idx}\n{fmt_time(segment.start)} --> {fmt_time(segment.end)}\n{text_ru}\n\n")
                idx += 1
                percent = 5 + int(85 * (i+1) / total)
                progress_callback(f"Переведено {i+1}/{total}", percent)
        progress_callback("Перевод готов", 100)
        done_callback(True, f"Субтитры: {out_srt}", out_srt)
    except Exception as e:
        log_to_file(traceback.format_exc(), "ERROR")
        done_callback(False, f"Ошибка: {e}", None)

# ---------- GUI ----------
CONFIG_FILE = "config.json"

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        root.title("🎬 Переводчик видео в субтитры (en->ru)")
        root.geometry("700x600")
        root.resizable(True, True)

        self.config = self.load_config()
        try:
            ensure_translation_package()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось установить пакет перевода:\n{e}")
            root.destroy()
            return

        self.files_list = []
        self.current_thread = None
        self.cancel_event = threading.Event()
        self.progress_var = tk.IntVar(value=0)

        self.setup_ui()
        self.apply_config()

    def load_config(self):
        default = {"model_size": "medium"}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    default.update(saved)
            except Exception as e:
                log_to_file(f"Ошибка загрузки config: {e}", "ERROR")
        return default

    def save_config(self):
        config = {"model_size": self.model_var.get()}
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            log_to_file("Настройки сохранены")
        except Exception as e:
            log_to_file(f"Ошибка сохранения config: {e}", "ERROR")

    def setup_ui(self):
        # Пробуем красивую тему (если установлена sv-ttk)
        try:
            import sv_ttk
            sv_ttk.set_theme("dark")
        except ImportError:
            pass

        # Фрейм выбора файлов
        file_frame = ttk.LabelFrame(self.root, text="Файлы для обработки")
        file_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(file_frame, text="Добавьте видео или аудио файлы (английский язык):").pack(anchor="w", padx=5, pady=5)

        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.listbox = tk.Listbox(list_frame, height=8)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(btn_frame, text="📂 Добавить файлы", command=self.add_files).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="❌ Удалить выбранный", command=self.remove_selected).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="🗑 Очистить всё", command=self.clear_all).pack(side="left", padx=2)

        # Настройки
        settings_frame = ttk.LabelFrame(self.root, text="Настройки")
        settings_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(settings_frame, text="Модель Whisper:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_var = tk.StringVar(value=self.config.get("model_size", "medium"))
        models = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
        ttk.OptionMenu(settings_frame, self.model_var, self.model_var.get(), *models).grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(settings_frame, text="💾 Сохранить настройки", command=self.save_config).grid(row=1, column=0, columnspan=2, pady=5)

        # Управление
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=5)
        self.start_btn = ttk.Button(control_frame, text="▶ Начать перевод", command=self.start_batch)
        self.start_btn.pack(side="left", padx=2)
        self.cancel_btn = ttk.Button(control_frame, text="⏹ Отмена", command=self.cancel_batch, state="disabled")
        self.cancel_btn.pack(side="left", padx=2)

        # Прогресс
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100, mode='determinate')
        self.progress_bar.pack(fill="x", padx=10, pady=5)

        # Статус и лог
        self.status_label = ttk.Label(self.root, text="Готов", foreground="gray")
        self.status_label.pack(pady=(5,0))
        self.log_text = tk.Text(self.root, height=10, state="disabled", bg="#f0f0f0")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

        self.log("Переводчик готов. Добавьте файлы и нажмите 'Начать перевод'.")
        log_to_file("Приложение запущено")

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Выберите видео или аудио",
            filetypes=[("Медиафайлы", "*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.m4a *.flac"), ("Все файлы", "*.*")]
        )
        for p in paths:
            if p not in self.files_list:
                self.files_list.append(p)
                self.listbox.insert("end", os.path.basename(p))
        self.log(f"Добавлено файлов: {len(paths)}")

    def remove_selected(self):
        selected = self.listbox.curselection()
        if selected:
            idx = selected[0]
            self.listbox.delete(idx)
            del self.files_list[idx]
            self.log(f"Удалён файл {idx+1}")

    def clear_all(self):
        self.listbox.delete(0, "end")
        self.files_list.clear()
        self.log("Список очищен")

    def log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        log_to_file(message, "GUI")

    def set_status(self, text, color="gray"):
        self.status_label.config(text=text, foreground=color)

    def update_progress(self, msg, percent=None):
        self.log(msg)
        self.set_status(msg, "blue")
        if percent is not None:
            self.progress_var.set(percent)

    def cancel_batch(self):
        if self.current_thread and self.current_thread.is_alive():
            self.cancel_event.set()
            self.set_status("Отмена...", "orange")
            self.log("Пользователь запросил отмену.")
            self.cancel_btn.config(state="disabled")

    def start_batch(self):
        if not self.files_list:
            messagebox.showwarning("Предупреждение", "Добавьте хотя бы один файл.")
            return

        self.save_config()
        self.start_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress_var.set(0)
        self.cancel_event.clear()
        self.current_thread = threading.Thread(target=self.process_batch, daemon=True)
        self.current_thread.start()

    def process_batch(self):
        total_files = len(self.files_list)
        for idx, file_path in enumerate(self.files_list):
            if self.cancel_event.is_set():
                self.root.after(0, self.batch_done, False, "Обработка прервана пользователем.")
                return
            self.root.after(0, self.log, f"\n--- Обработка файла {idx+1}/{total_files}: {os.path.basename(file_path)} ---")
            try:
                srt_result = [None]
                def trans_progress(msg, percent):
                    self.root.after(0, self.update_progress, msg, percent)
                def trans_done(success, msg, srt):
                    if success:
                        srt_result[0] = srt
                        self.root.after(0, self.log, f"[OK] {msg}")
                    else:
                        raise Exception(msg)
                trans_event = threading.Event()
                def trans_wrapper():
                    transcribe_and_translate(
                        file_path, self.model_var.get(),
                        trans_progress, trans_done, self.cancel_event
                    )
                    trans_event.set()
                t = threading.Thread(target=trans_wrapper)
                t.start()
                while not trans_event.is_set() and not self.cancel_event.is_set():
                    t.join(0.1)
                if self.cancel_event.is_set():
                    return
                if not srt_result[0]:
                    raise Exception("Перевод не удался")
                file_percent = int((idx+1) / total_files * 100)
                self.root.after(0, self.update_progress, f"Файл {idx+1}/{total_files} обработан", file_percent)
            except Exception as e:
                self.root.after(0, self.log, f"[ОШИБКА] {e}")
                log_to_file(traceback.format_exc(), "ERROR")
                if self.cancel_event.is_set():
                    break
        self.root.after(0, self.batch_done, True, "Пакетная обработка завершена.")

    def batch_done(self, success, msg):
        self.start_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.progress_var.set(100 if success else 0)
        self.set_status(msg, "green" if success else "red")
        self.log(msg)
        self.current_thread = None

    def apply_config(self):
        self.model_var.set(self.config.get("model_size", "medium"))

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()