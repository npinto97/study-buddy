from study_buddy.utils.tools import OpenAISpeechToText

print("Test manuale della trascrizione...")
test_file_path = "c:/Users/Ningo/AppData/Local/Temp/tmpnc3vzbe_.wav"
transcriber = OpenAISpeechToText()
try:
    result = transcriber.transcribe_audio(test_file_path)
    print(f"Test OK! Testo trascritto: {result}")
except Exception as e:
    print(f"Errore nella trascrizione manuale: {e}")
