#!/usr/bin/env python3

import argparse
import queue
import sys
import sounddevice as sd
import wave
import time
from vosk import Model, KaldiRecognizer

q = queue.Queue()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-l", "--list-devices", action="store_true",
    help="show list of audio devices and exit")
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    "-f", "--filename", type=str, metavar="FILENAME",
    help="audio file to store recording to", default="output.wav")
parser.add_argument(
    "-d", "--device", type=int_or_str,
    help="input device (numeric ID or substring)")
parser.add_argument(
    "-r", "--samplerate", type=int, help="sampling rate")
parser.add_argument(
    "-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is tr")
parser.add_argument(
    "-s", "--silence", type=int, help="silence duration in seconds to stop recording", default=2)
args = parser.parse_args(remaining)

def write_wave(filename, data, samplerate):
    """Helper function to write data to a .wav file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(samplerate)
        wf.writeframes(b''.join(data))

def append_wave(temp_filename, final_filename):
    """Append temporary .wav file to final .wav file"""
    with wave.open(temp_filename, 'rb') as temp_wf:
        with wave.open(final_filename, 'wb') as final_wf:
            final_wf.setnchannels(temp_wf.getnchannels())
            final_wf.setsampwidth(temp_wf.getsampwidth())
            final_wf.setframerate(temp_wf.getframerate())
            while True:
                frames = temp_wf.readframes(1024)
                if not frames:
                    break
                final_wf.writeframes(frames)

def create_wave_header(filename, samplerate):
    """Helper function to create the .wav file header."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(samplerate)

try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, "input")
        args.samplerate = int(device_info["default_samplerate"])

    if args.model is None:
        model = Model(lang="tr")
    else:
        model = Model(lang=args.model)

    temp_filename = "temp_output.wav"
    create_wave_header(temp_filename, args.samplerate)

    with sd.RawInputStream(samplerate=args.samplerate, blocksize=8000, device=args.device,
                           dtype="int16", channels=1, callback=callback):
        print("#" * 80)
        print("Konuşmaya başladığınızda kayıt otomatik olarak başlayacak.")
        print("Kaydı sonlandırmak için sessizlik süresi: {} saniye veya Ctrl+C'ye basın.".format(args.silence))
        print("#" * 80)

        rec = KaldiRecognizer(model, args.samplerate)
        recording = False
        audio_data = []
        silence_timer = None

        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = rec.Result()
                if '"text" : ""' not in result:
                    if not recording:
                        print("Konuşma algılandı, kayıt başlatılıyor...")
                        recording = True
                    silence_timer = None
                    audio_data.append(data)
                else:
                    if recording and silence_timer is None:
                        silence_timer = time.time() + args.silence  # specified seconds of silence before stopping
                    elif recording and silence_timer and time.time() > silence_timer:
                        print("Konuşma sona erdi, kayıt tamamlanıyor...")
                        recording = False
                        write_wave(temp_filename, audio_data, args.samplerate)
                        append_wave(temp_filename, args.filename)
                        print(f"Kayıt {args.filename} dosyasına kaydedildi.")
                        print("Program sonlandırılıyor...")
                        break
            else:
                if recording:
                    audio_data.append(data)
            
            if recording and silence_timer:
                print("Sessizlik algılandı, kayıt sonlandırılacak...")
            elif not recording and not silence_timer:
                print("Konuşmayı başlatın...")

except KeyboardInterrupt:
    print("\nKullanıcı tarafından sonlandırıldı.")
    if recording:
        print("Kayıt sonlandırılıyor...")
        write_wave(temp_filename, audio_data, args.samplerate)
        append_wave(temp_filename, args.filename)
        print(f"Kayıt {args.filename} dosyasına kaydedildi.")
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ": " + str(e))

