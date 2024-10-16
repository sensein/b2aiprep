import os
import shutil

def copy_wav_file(src_file_path, dest_directory, file_names):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    
    for file_name in file_names:
        dest_file_path = os.path.join(dest_directory, file_name)
        shutil.copy(src_file_path, dest_file_path)

# Example usage
src_file_path = '/path/to/your/source/file.wav'  # Update this with the path to your source file
dest_directory = '/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/b2ai-data-bids-like-curated/sub-0aeebf70-44a5-4537-af1b-1c24840f104d/ses-C63E6402-5ECC-45B5-8A57-6FE638A766A5/audio'
file_names = [
    "Animal-fluency_rec-Animal-fluency.wav",
    "Audio-Check_rec-Audio-Check-1.wav",
    "Audio-Check_rec-Audio-Check-2.wav",
    "Audio-Check_rec-Audio-Check-3.wav",
    "Audio-Check_rec-Audio-Check-4.wav",
    "Diadochokinesis_rec-Diadochokinesis-buttercup.wav",
    "Diadochokinesis_rec-Diadochokinesis-KA.wav",
    "Diadochokinesis_rec-Diadochokinesis-PA.wav",
    "Diadochokinesis_rec-Diadochokinesis-Pataka.wav",
    "Diadochokinesis_rec-Diadochokinesis-TA.wav",
    "Free-speech_rec-Free-speech-1.wav",
    "Free-speech_rec-Free-speech-2.wav",
    "Free-speech_rec-Free-speech-3.wav",
    "Glides_rec-Glides-High-to-Low.wav",
    "Glides_rec-Glides-Low-to-High.wav",
    "Loudness_rec-Loudness.wav",
    "Maximum-phonation-time_rec-Maximum-phonation-time-1.wav",
    "Maximum-phonation-time_rec-Maximum-phonation-time-2.wav",
    "Maximum-phonation-time_rec-Maximum-phonation-time-3.wav",
    "Open-response-questions_rec-Open-response-questions.wav",
    "Picture-description_rec-Picture-description.wav",
    "Prolonged-vowel_rec-Prolonged-vowel.wav",
    "Rainbow-Passage_rec-Rainbow-Passage.wav",
    "Respiration-and-cough_rec-Respiration-and-cough-Breath-1.wav",
    "Respiration-and-cough_rec-Respiration-and-cough-Breath-2.wav",
    "Respiration-and-cough_rec-Respiration-and-cough-Cough-1.wav",
    "Respiration-and-cough_rec-Respiration-and-cough-Cough-2.wav",
    "Respiration-and-cough_rec-Respiration-and-cough-FiveBreaths-1.wav",
    "Respiration-and-cough_rec-Respiration-and-cough-FiveBreaths-2.wav",
    "Respiration-and-cough_rec-Respiration-and-cough-FiveBreaths-3.wav",
    "Respiration-and-cough_rec-Respiration-and-cough-FiveBreaths-4.wav",
    "Story-recall_rec-Story-recall.wav"
]

copy_wav_file(src_file_path, dest_directory, file_names)
