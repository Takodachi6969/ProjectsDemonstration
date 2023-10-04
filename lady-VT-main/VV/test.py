from twitchio.ext import commands
from chat import *
from google.cloud import texttospeech_v1beta1 as texttospeech
import vlc
import os
import time
import nltk
from voicevox import Client

def play_voicevox_message(message: str):
    with Client() as client:
        audio_query = client.create_audio_query(
            message, speaker=0
        )
        with open("output.wav", "wb") as f:
            f.write(audio_query.synthesis())
        audio_file = os.path.dirname(__file__) + '\output.wav'
        media = vlc.MediaPlayer(audio_file)
        media.play()

response = "こんにちは！私の名前はブリテンです。今日からきっと楽しい話を交わしましょう？"
input_text = response
sound = play_voicevox_message(input_text)