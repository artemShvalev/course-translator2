import sys
import os
import threading
import json
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pysrt
import torch
from pydub import AudioSegment
from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate

# ---------- Логирование в файл ----------
LOG_FILE = "app.log"

def log_to_file(message, level="INFO"):
    """Записывает сообщение в лог-файл с временной меткой."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{level}] {message}\n")

def log_exception(exc_type, exc_value, exc_traceback):
    """Перехватывает непойманные исключения и пишет в лог."""
    log_to_file("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)), "CRITICAL")
    # Затем вызовем стандартный обработчик (чтобы программа не молча умирала)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = log_exception

# ---------- Кэширование моделей ----------
_whisper_models = {}
_silero_model = None
_silero_device = None

def get_whisper_model(model_size, device="cpu"):
    key = (model_size, device)
    if key not in _whisper_models:
        log_to_file(f"Загрузка модели Whisper: {model_size} на {device}")
        _whisper_models[key] = WhisperModel(model_size, device=device, compute_type="int8")
    return _whisper_models[key]

def get_silero_tts(device="cpu"):
    global _silero_model, _silero_device
    if _silero_model is None or _silero_device != device:
        log_to_file(f"Загрузка Silero TTS на {device}")
        _silero_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v3_1_ru'
        )
        _silero_model.to(device)
        _silero_device = device
    return _silero_model

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

def transcribe_and_translate(input_path, model_size, device, progress_callback, done_callback, cancel_event, beam_size=1):
    try:
        progress_callback("Загружаем Whisper...", 0)
        model = get_whisper_model(model_size, device)
        if cancel_event.is_set():
            done_callback(False, "Отменено.", None)
            return

        progress_callback("Распознаём и переводим...", 5)
        segments, _ = model.transcribe(
            input_path,
            language="en",
            beam_size=beam_size,          # Ускорение (1 = максимальная скорость)
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

def generate_audio_from_srt(srt_path, progress_callback, done_callback, cancel_event, speaker='xenia', speed=1.0):
    try:
        progress_callback("Загружаем Silero...", 0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_silero_tts(device)
        sample_rate = 48000

        subs = pysrt.open(srt_path, encoding='utf-8')
        total = len(subs)
        if total == 0:
            done_callback(False, "SRT пуст.", None)
            return

        final_audio = AudioSegment.silent(duration=500)
        for idx, sub in enumerate(subs):
            if cancel_event.is_set():
                done_callback(False, "Отменено.", None)
                return
            text = sub.text.replace('\n', ' ').strip()
            if not text:
                continue
            start_ms = sub.start.ordinal
            end_ms = sub.end.ordinal
            orig_duration = end_ms - start_ms

            # Синтез с выбранным голосом
            audio_np = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
            audio_seg = AudioSegment(
                (audio_np * 32767).astype('int16').tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )

            # Применяем изменение скорости (если нужно)
            if speed != 1.0:
                audio_seg = audio_seg.speedup(playback_speed=1.0/speed)  # speedup ожидает множитель ускорения (>1 ускоряет)

            # Выравнивание по времени
            current_len = len(final_audio)
            if current_len < start_ms:
                final_audio += AudioSegment.silent(duration=start_ms - current_len)

            phrase_duration = len(audio_seg)
            if phrase_duration > orig_duration:
                ratio = orig_duration / phrase_duration
                if ratio >= 0.7:
                    audio_seg = audio_seg.speedup(playback_speed=1.0/ratio)
                else:
                    audio_seg = audio_seg[:orig_duration].fade_out(100)
            final_audio += audio_seg
            percent = int(90 * (idx+1) / total)
            progress_callback(f"Синтез {idx+1}/{total}", percent)

        wav_path = os.path.splitext(srt_path)[0] + "_dub.wav"
        final_audio.export(wav_path, format="wav")
        progress_callback("Озвучка готова", 100)
        done_callback(True, f"Аудио: {wav_path}", wav_path)
    except Exception as e:
        log_to_file(traceback.format_exc(), "ERROR")
        done_callback(False, f"Ошибка: {e}", None)

def mix_audio_with_video(video_path, audio_path, progress_callback, done_callback, cancel_event):
    try:
        progress_callback("Склейка...", 50)
        if cancel_event.is_set():
            done_callback(False, "Отменено.", None)
            return
        output_video = os.path.splitext(video_path)[0] + "_dubbed.mp4"
        import subprocess
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
               "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_video]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        progress_callback("Склейка завершена", 100)
        done_callback(True, f"Видео: {output_video}", output_video)
    except Exception as e:
        log_to_file(traceback.format_exc(), "ERROR")
        done_callback(False, f"Ошибка ffmpeg: {e}", None)

# ---------- GUI ----------
CONFIG_FILE = "config.json"

class TranslatorApp:
    def __init__(self, root):
        self.root = root
        root.title("🎬 Переводчик курсов с озвучкой")
        root.geometry("800x750")
        root.resizable(True, True)

        # Загрузка сохранённых настроек
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

        # Создаём виджеты
        self.setup_ui()

        # Применяем сохранённые настройки
        self.apply_config()

    def load_config(self):
        """Загружает настройки из config.json"""
        default_config = {
            "model_size": "medium",
            "device": "cpu",
            "do_translate": True,
            "do_dub": True,
            "do_mix": True,
            "tts_speaker": "xenia",
            "tts_speed": 1.0
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    default_config.update(saved)
                    log_to_file("Настройки загружены")
            except Exception as e:
                log_to_file(f"Ошибка загрузки config: {e}", "ERROR")
        return default_config

    def save_config(self):
        """Сохраняет текущие настройки в config.json"""
        config = {
            "model_size": self.model_var.get(),
            "device": self.device_var.get(),
            "do_translate": self.do_translate.get(),
            "do_dub": self.do_dub.get(),
            "do_mix": self.do_mix.get(),
            "tts_speaker": self.tts_speaker_var.get(),
            "tts_speed": self.tts_speed_var.get()
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            log_to_file("Настройки сохранены")
        except Exception as e:
            log_to_file(f"Ошибка сохранения config: {e}", "ERROR")

    def setup_ui(self):
        # Стилизация (попробуем импортировать sv-ttk, если установлен)
        try:
            import sv_ttk
            sv_ttk.set_theme("dark")
        except ImportError:
            pass

        # Вкладки
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Вкладка "Файлы"
        tab_files = ttk.Frame(notebook)
        notebook.add(tab_files, text="📁 Файлы")

        # Вкладка "Настройки"
        tab_settings = ttk.Frame(notebook)
        notebook.add(tab_settings, text="⚙ Настройки")

        # ---------- Вкладка файлов ----------
        ttk.Label(tab_files, text="Файлы для обработки:").pack(anchor="w", pady=5, padx=10)
        list_frame = ttk.Frame(tab_files)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.listbox = tk.Listbox(list_frame, height=8)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(tab_files)
        btn_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(btn_frame, text="📂 Добавить файлы", command=self.add_files).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="❌ Удалить выбранный", command=self.remove_selected).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="🗑 Очистить всё", command=self.clear_all).pack(side="left", padx=2)

        # ---------- Вкладка настроек ----------
        settings_frame = ttk.Frame(tab_settings)
        settings_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Модель Whisper
        ttk.Label(settings_frame, text="Модель Whisper:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_var = tk.StringVar(value=self.config.get("model_size", "medium"))
        models = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
        ttk.OptionMenu(settings_frame, self.model_var, self.model_var.get(), *models).grid(row=0, column=1, padx=5, pady=5)

        # Устройство
        ttk.Label(settings_frame, text="Устройство:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        device_frame = ttk.Frame(settings_frame)
        device_frame.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.device_var = tk.StringVar(value=self.config.get("device", "cpu"))
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_var, value="cpu").pack(side="left")
        ttk.Radiobutton(device_frame, text="GPU (CUDA)", variable=self.device_var, value="cuda").pack(side="left", padx=10)

        # Действия
        self.do_translate = tk.BooleanVar(value=self.config.get("do_translate", True))
        self.do_dub = tk.BooleanVar(value=self.config.get("do_dub", True))
        self.do_mix = tk.BooleanVar(value=self.config.get("do_mix", True))
        ttk.Checkbutton(settings_frame, text="Перевести (создать SRT)", variable=self.do_translate).grid(row=2, column=0, columnspan=2, sticky="w", padx=5)
        ttk.Checkbutton(settings_frame, text="Озвучить (синтез речи)", variable=self.do_dub).grid(row=3, column=0, columnspan=2, sticky="w", padx=5)
        ttk.Checkbutton(settings_frame, text="Склеить с видео", variable=self.do_mix).grid(row=4, column=0, columnspan=2, sticky="w", padx=5)

        # Настройки TTS
        ttk.Label(settings_frame, text="Голос озвучки:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.tts_speaker_var = tk.StringVar(value=self.config.get("tts_speaker", "xenia"))
        speakers = ["xenia", "aidar", "baya"]
        ttk.OptionMenu(settings_frame, self.tts_speaker_var, self.tts_speaker_var.get(), *speakers).grid(row=5, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(settings_frame, text="Скорость речи:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        speed_frame = ttk.Frame(settings_frame)
        speed_frame.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        self.tts_speed_var = tk.DoubleVar(value=self.config.get("tts_speed", 1.0))
        speed_scale = ttk.Scale(speed_frame, from_=0.8, to=1.5, variable=self.tts_speed_var, orient="horizontal", length=150)
        speed_scale.pack(side="left")
        self.speed_label = ttk.Label(speed_frame, text=f"{self.tts_speed_var.get():.1f}x")
        self.speed_label.pack(side="left", padx=5)
        speed_scale.configure(command=lambda x: self.speed_label.config(text=f"{float(x):.1f}x"))

        # Кнопка сохранения настроек
        ttk.Button(settings_frame, text="💾 Сохранить настройки", command=self.save_config).grid(row=7, column=0, columnspan=2, pady=10)

        # ---------- Общие элементы управления ----------
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=20, pady=5)
        self.start_btn = ttk.Button(control_frame, text="▶ Начать обработку", command=self.start_batch)
        self.start_btn.pack(side="left", padx=2)
        self.cancel_btn = ttk.Button(control_frame, text="⏹ Отмена", command=self.cancel_batch, state="disabled")
        self.cancel_btn.pack(side="left", padx=2)

        # Прогресс-бар
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100, mode='determinate')
        self.progress_bar.pack(fill="x", padx=20, pady=5)

        # Статус и лог
        self.status_label = ttk.Label(self.root, text="Готов", foreground="gray")
        self.status_label.pack(pady=(5,0))
        self.log_text = tk.Text(self.root, height=12, state="disabled", bg="#f0f0f0")
        self.log_text.pack(fill="both", expand=True, padx=20, pady=10)

        self.log("Пакет перевода en->ru готов.\nДобавьте файлы и нажмите 'Начать обработку'.")
        log_to_file("Приложение запущено")

    # ---------- Методы работы с файлами ----------
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
        if not self.do_translate.get() and not self.do_dub.get() and not self.do_mix.get():
            messagebox.showwarning("Предупреждение", "Выберите хотя бы одно действие.")
            return

        # Сохраняем настройки перед запуском
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
            current_srt = None
            current_audio = None
            try:
                # Перевод
                if self.do_translate.get():
                    self.root.after(0, self.set_status, f"Перевод: {os.path.basename(file_path)}", "blue")
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
                            file_path, self.model_var.get(), self.device_var.get(),
                            trans_progress, trans_done, self.cancel_event
                        )
                        trans_event.set()
                    t = threading.Thread(target=trans_wrapper)
                    t.start()
                    while not trans_event.is_set() and not self.cancel_event.is_set():
                        t.join(0.1)
                    if self.cancel_event.is_set():
                        return
                    current_srt = srt_result[0]
                    if not current_srt:
                        raise Exception("Перевод не удался")
                # Озвучка
                if self.do_dub.get():
                    if not current_srt:
                        candidate = os.path.splitext(file_path)[0] + "_ru.srt"
                        if os.path.exists(candidate):
                            current_srt = candidate
                        else:
                            raise Exception("Для озвучки нужен SRT файл.")
                    self.root.after(0, self.set_status, f"Озвучка: {os.path.basename(file_path)}", "blue")
                    audio_result = [None]
                    def dub_progress(msg, percent):
                        self.root.after(0, self.update_progress, msg, percent)
                    def dub_done(success, msg, audio):
                        if success:
                            audio_result[0] = audio
                            self.root.after(0, self.log, f"[OK] {msg}")
                        else:
                            raise Exception(msg)
                    dub_event = threading.Event()
                    def dub_wrapper():
                        generate_audio_from_srt(
                            current_srt, dub_progress, dub_done, self.cancel_event,
                            speaker=self.tts_speaker_var.get(),
                            speed=self.tts_speed_var.get()
                        )
                        dub_event.set()
                    t2 = threading.Thread(target=dub_wrapper)
                    t2.start()
                    while not dub_event.is_set() and not self.cancel_event.is_set():
                        t2.join(0.1)
                    if self.cancel_event.is_set():
                        return
                    current_audio = audio_result[0]
                    if not current_audio:
                        raise Exception("Озвучка не удалась")
                # Склейка
                if self.do_mix.get():
                    if not current_audio:
                        candidate_audio = os.path.splitext(file_path)[0] + "_dub.wav"
                        if os.path.exists(candidate_audio):
                            current_audio = candidate_audio
                        else:
                            raise Exception("Для склейки нужна аудиодорожка.")
                    self.root.after(0, self.set_status, f"Склейка: {os.path.basename(file_path)}", "blue")
                    mix_result = [None]
                    def mix_progress(msg, percent):
                        self.root.after(0, self.update_progress, msg, percent)
                    def mix_done(success, msg, video):
                        if success:
                            mix_result[0] = video
                            self.root.after(0, self.log, f"[OK] {msg}")
                        else:
                            raise Exception(msg)
                    mix_event = threading.Event()
                    def mix_wrapper():
                        mix_audio_with_video(file_path, current_audio, mix_progress, mix_done, self.cancel_event)
                        mix_event.set()
                    t3 = threading.Thread(target=mix_wrapper)
                    t3.start()
                    while not mix_event.is_set() and not self.cancel_event.is_set():
                        t3.join(0.1)
                    if self.cancel_event.is_set():
                        return
                    if not mix_result[0]:
                        raise Exception("Склейка не удалась")
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
        """Применяет загруженные настройки к виджетам."""
        self.model_var.set(self.config.get("model_size", "medium"))
        self.device_var.set(self.config.get("device", "cpu"))
        self.do_translate.set(self.config.get("do_translate", True))
        self.do_dub.set(self.config.get("do_dub", True))
        self.do_mix.set(self.config.get("do_mix", True))
        self.tts_speaker_var.set(self.config.get("tts_speaker", "xenia"))
        self.tts_speed_var.set(self.config.get("tts_speed", 1.0))
        if hasattr(self, 'speed_label'):
            self.speed_label.config(text=f"{self.tts_speed_var.get():.1f}x")

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()