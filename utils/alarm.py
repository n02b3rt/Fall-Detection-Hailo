"""
Funkcje alarmowe dla wykrywania upadków
"""
import os


def trigger_alarm():
    """
    Wyzwala alarm gdy wykryty zostanie upadek.
    
    Możesz tutaj dodać:
    - Wysyłanie SMS
    - Powiadomienia push
    - Zapis do bazy danych
    - Dźwięk alarmu
    - Mail notifications
    """
    print("\n" + "=" * 50)
    print("!!!  ALARM - WYKRYTO UPADEK  !!!")
    print("=" * 50 + "\n")
    
    # Przykład: odtwórz dźwięk alarmu (jeśli masz speaker)
    try:
        os.system("aplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null &")
    except Exception as e:
        print(f"[WARNING] Nie udało się odtworzyć dźwięku: {e}")


def clear_alarm():
    """
    Kasuje alarm gdy osoba wstała.
    """
    print("[INFO] Osoba wstała - alarm skasowany")
