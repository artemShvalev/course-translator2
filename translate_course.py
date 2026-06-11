import sys
import os
from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate

def ensure_translation_package():
    """Убедимся, что пакет en->ru установлен."""
    argostranslate.package.update_package_index()
    installed = argostranslate.package.get_installed_packages()
    if any(p.from_code == "en" and p.to_code == "ru" for p in installed):
        return  # уже есть
    available = argostranslate.package.get_available_packages()
    try:
        pkg = next(p for p in available if p.from_code == "en" and p.to_code == "ru")
    except StopIteration:
        print("Пакет en->ru не найден в репозитории, проверьте интернет или обновите argostranslate")
        sys.exit(1)
    print("Устанавливаем языковой пакет English -> Russian...")
    download_path = pkg.download()
    argostranslate.package.install_from_path(download_path)
    print("Готово.")

def transcribe_and_translate(audio_path, model_size="medium", device="cpu"):
    # Можно заменить "medium" на "tiny", "base", "small", "large-v3"
    print(f"Загружаем модель Whisper ({model_size})...")
    model = WhisperModel(model_size, device=device, compute_type="int8")
    
    print("Распознаём речь и переводим...")
    segments, _ = model.transcribe(audio_path, language="en")
    
    out_srt = os.path.splitext(audio_path)[0] + "_ru.srt"
    with open(out_srt, "w", encoding="utf-8") as f:
        idx = 1
        for segment in segments:
            start = segment.start
            end = segment.end
            text_en = segment.text.strip()
            if not text_en:
                continue
            text_ru = argostranslate.translate.translate(text_en, "en", "ru")
            f.write(f"{idx}\n")
            f.write(f"{fmt_time(start)} --> {fmt_time(end)}\n")
            f.write(f"{text_ru}\n\n")
            idx += 1
    print(f"Русские субтитры сохранены в {out_srt}")

def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python translate_course.py видео.mp4 [medium] [cpu]")
        sys.exit(1)
    
    ensure_translation_package()
    
    input_file = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    
    transcribe_and_translate(input_file, model_size, device)