from bilibili_api import live, sync
from twitchio.ext import commands
from chat import *
from google.cloud import texttospeech_v1beta1 as texttospeech
import vlc
import os
import time
import nltk
from voicevox import Client
import re
from googletrans import Translator
import wave
import threading
from itertools import chain
import tempfile
import shutil
import json
import pytchat
import openai
from pytchat import LiveChat, SpeedCalculator
import time
import requests
from pydub import AudioSegment
from pydub.playback import play
import io
import pyttsx3
import sys
import argparse
import pygame
room_id = int(input('Stream code'))
room = live.LiveDanmaku(room_id)

last_run_time = time.time()
Count = 0
language = -1
conversation = list()

@room.on('DANMU_MSG')
async def on_danmaku(event):
    global last_run_time
    current_time = time.time()+120
    # if current_time - last_run_time < 10000 and messageauthorname != 'ikaros_senpai':  # check if the time since last run is less than 10 seconds
    #     return
    if current_time - last_run_time > 5:

        print(current_time)
        print(last_run_time)
        # check if the time since last run is less than 10 seconds
        last_run_time = current_time
        message_content = event["data"]["info"][1]
        messageauthorname = event["data"]["info"][2][1]

        global Count
        global language
        if messageauthorname == 'Yakiniku烤肉君' and message_content == 'English please':
            language = 1
            return
        if messageauthorname == 'Yakiniku烤肉君' and message_content == 'Japanese please':
            language = -1
            return

        # Using voiceVox engine to generate japanese text
        async def Sound(text, k):
            async with Client() as client:
                audio_query = await client.create_audio_query(
                    text, speaker=k
                )
                with open("output.wav", "wb") as f:
                    f.write(await audio_query.synthesis())

        # Messages with echo set to True are messages sent by the bot...
        # For now we just want to ignore them...

        # if message.echo:
        #     return

        # download the words corpus
        # nltk.download('words')

        # Check if the message contains english words
        # if not any(word in message_content for word in nltk.corpus.words.words()):
        #     return
        # Check if the message is too long
        if len(message_content) > 1000:
            return


        print('------------------------------------------------------')
        print(message_content)
        print(conversation)
        print(conversation)
        print(Count)
        translator = Translator()

        if Count >= 5:
            conversation.clear()
            Count = 0

        def is_english_or_chinese(string):
            english_range = chain(range(0x0041, 0x005a), range(0x0061, 0x007a))  # ASCII A-Z, a-z
            chinese_range = range(0x4e00, 0x9fff)  # Basic Chinese character set

            translator = Translator()
            contains_english = False
            contains_chinese = False

            for char in string:
                if ord(char) in english_range:
                    contains_english = True
                elif ord(char) in chinese_range:
                    contains_chinese = True

                if contains_english and contains_chinese:
                    break

            if contains_chinese and not contains_english:
                string = translator.translate(string, dest='en').text

            return string if contains_english or contains_chinese else "unknown"

        if is_english_or_chinese(message_content) == "unknown":
            return

        # Replace 'message_content' with the actual string you want to test
        result = is_english_or_chinese(message_content)

        conversation.append(f'CHAT: {result}')
        text_block = '\n'.join(conversation)

        prompt = open_file('prompt_chat.txt').replace('<<BLOCK>>', text_block)
        if messageauthorname == "Yakiniku烤肉君":
            print(prompt + 'You are now speaking to Peter' + '\nBRITAIN:')
            prompt = prompt + 'You are now speaking to Peter' + '\nBRITAIN:'

        else:
            print(prompt + 'You are now speaking to your viewers' + '\nBRITAIN:')
            prompt = prompt + 'You are now speaking to your viewers' + messageauthorname + '\nBRITAIN:'

        if messageauthorname == 'Yakiniku烤肉君' and message_content == 'sudo STFU':
            response = 'filtered'
        else:

            response = gpt3_completion(prompt)
        # response = "Hello, how are you?"
        # print(prompt)
        Count = Count + 1

        with open("EnglishOut.txt", "w", encoding="utf-8") as file:
            # Write the string to the file
            file.write(messageauthorname + ': ' + message_content + '. ' + response)
        with open("ChineseOut.txt", "w", encoding="utf-8") as file:
            p = messageauthorname + ': ' + message_content + '. ' + translator.translate(response, dest='zh-cn').text
            file.write(p)

        japanese_matches = translator.translate(response, dest='ja')

        def replace_english_with_hello(input_string):
            # Find all English words in the input string
            english_words = re.findall(r'[a-zA-Z]+', input_string)

            # Replace each English word with "Hello"
            for word in english_words:
                translated = translator.translate(word, dest='ja')
                input_string = input_string.replace(word, translated.text)

            return input_string

        input_string = japanese_matches.text
        nametranslated = translator.translate(messageauthorname, dest='ja')
        contenttranslated = translator.translate(message_content, dest='ja')
        namevoice = replace_english_with_hello(nametranslated.text)
        messagevoice = replace_english_with_hello(contenttranslated.text)
        output_string = replace_english_with_hello(input_string)
        print(namevoice + ' ' + messagevoice + ' ' + output_string)

        print('BRITAIN:', response)
        with open("record.txt", "a", encoding="utf-8") as out:
            out.write(message_content + response)

        if (conversation.count('BRITAIN: ' + response) == 0):
            conversation.append(f'BRITAIN: {response}')

        with open("config.json", "r") as json_file:
            data = json.load(json_file)

        class EL:
            key = data["keys"][0]["EL_key"]
            voice = data["EL_data"][0]["voice"]

        async def EL_TTS(message):

            url = f'https://api.elevenlabs.io/v1/text-to-speech/{EL.voice}'
            headers = {
                'accept': 'audio/mpeg',
                'xi-api-key': '73fbf517c938e65b7ceacca8402341e2',
                'Content-Type': 'application/json'
            }
            data = {
                'text': message,
                'voice_settings': {
                    'stability': 0,
                    'similarity_boost': 0
                }
            }

            responseMusic = requests.post(url, headers=headers, json=data, stream=True)
            with open("output.mp3", "wb") as f:
                f.write(responseMusic.content)
            # audio_content = AudioSegment.from_file(io.BytesIO(response.content), format="wav")
            # play(audio_content)
        k = 0
        if language == 1:
            print('EnglishTTS')
            await EL_TTS(messageauthorname + ' ' + message_content + ' ' + response)
            audio = AudioSegment.from_file('output.mp3', format='mp3')

        if language == -1:
            print('JPTTS')
            await Sound(namevoice + ' ' + messagevoice + ' ' + output_string, k)
            audio = AudioSegment.from_file('output.wav', format='wav')

        # Load MP3 file
        # audio = AudioSegment.from_file('output.mp3', format='mp3')

        # Play MP3 file
        # play(audio)

        # pygame.mixer.init()
        #
        # # Load MP3 file
        # pygame.mixer.music.load('output.mp3')
        #
        # # Play MP3 file
        # pygame.mixer.music.play()
        #
        # # Wait for MP3 file to finish playing
        # while pygame.mixer.music.get_busy():
        #     pygame.time.Clock().tick(10)
        # os.remove("output.mp3")

        # audio_file = os.path.dirname(__file__) + '\output.mp3'
        # media = vlc.MediaPlayer(audio_file)
        # media.play()
        # Play the sound
        # def get_audio_duration(wav_file):
        #     with wave.open(wav_file, 'r') as audio_file:
        #         frames = audio_file.getnframes()
        #         rate = audio_file.getframerate()
        #         duration = frames / float(rate)
        #         return duration
        #
        # def get_char_duration(transcript_file):
        #     with open(transcript_file, 'r', encoding="utf-8") as file:
        #         text = file.read()
        #         num_chars = len(text)
        #         return num_chars
        #
        # def get_words_and_durations(transcript_file, audio_duration):
        #     with open(transcript_file, 'r', encoding="utf-8") as file:
        #         text = file.read()
        #         words = re.findall(r'\w+|\W+', text)
        #     durations = [len(word) * audio_duration / get_char_duration(transcript_file) for word in words]
        #     return words, durations
        #
        # def play_audio(wav_file):
        #     player = vlc.MediaPlayer(wav_file)
        #     player.play()
        #     while player.get_state() != vlc.State.Ended:
        #         time.sleep(0.1)
        #
        # def update_subtitle_file(output_file, current_words):
        #     with tempfile.NamedTemporaryFile('w', encoding="utf-8", delete=False) as tmp_file:
        #         tmp_file.write("".join(current_words))
        #         temp_path = tmp_file.name
        #
        #     shutil.move(temp_path, output_file)
        #
        # def play_audio_and_update_subtitles(wav_file, transcript_file, output_file):
        #     audio_duration = get_audio_duration(wav_file)
        #     words, durations = get_words_and_durations(transcript_file, audio_duration)
        #
        #     audio_thread = threading.Thread(target=play_audio, args=(wav_file,))
        #     audio_thread.start()
        #
        #     start_time = time.time()
        #
        #     start_index = 0
        #     end_index = 0
        #     while end_index < len(words):
        #         current_time = time.time() - start_time
        #         chunk_start_time = sum(durations[:end_index])
        #
        #         if current_time >= chunk_start_time:
        #             if end_index - start_index < 12:
        #                 end_index += 1
        #             else:
        #                 start_index += 1
        #                 end_index += 1
        #
        #         current_words = words[start_index:end_index]
        #         update_subtitle_file(output_file, current_words)
        #         time.sleep(0.01)
        #
        #     audio_thread.join()
        #
        # # Usage
        # wav_file = 'output.wav'
        # transcript_file = 'EnglishOut.txt'
        # output_file = 'output.txt'
        # play_audio_and_update_subtitles(wav_file, transcript_file, output_file)
        with open('ChineseOut.txt', 'r', encoding = "utf-8") as file:
            string = file.read()
        substrings = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]

        # Save the substrings to a text file
        with open('output.txt', 'w', encoding='utf-8') as file:
            file.write('\n'.join(substrings))

        with open('EnglishOut.txt', 'r', encoding = "utf-8") as file:
            string = file.read()
        substrings = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]

        # Save the substrings to a text file
        with open('output2.txt', 'w', encoding='utf-8') as file:
            file.write('\n'.join(substrings))
        with open('ChineseOut.txt', 'r', encoding = "utf-8") as file:
            string = file.read()
        substrings = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]
        play(audio)

        # substrings = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]
        # audiolength = 22.4
        # timediff = audiolength / len(substrings)
        # starttime = time.time()


        # with open('ChineseOut.txt', 'r', encoding = "utf-8") as file:
        #     string = file.read()
        # substrings = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]
        #
        # for i in range(len(substrings)):
        #     with open('output.txt', 'w', encoding='utf-8') as file:
        #         file.write('\n' + substrings[i])
        #     # time.sleep(timediff)
        #
        # with open('EnglishOut.txt', 'r', encoding = "utf-8") as file:
        #     string = file.read()
        # substrings = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]
        #
        # # Save the substrings to a text file
        # for i in range(len(substrings)):
        #     with open('output2.txt', 'w', encoding='utf-8') as file:
        #         file.write('\n' + substrings[i])
        #     # time.sleep(timediff)


        # client = texttospeech.TextToSpeechClient()
        #
        # response = message_content + "? " + response
        # ssml_text = '<speak>'
        # response_counter = 0
        # mark_array = []
        # for s in response.split(' '):
        #     ssml_text += f'<mark name="{response_counter}"/>{s}'
        #     mark_array.append(s)
        #     response_counter += 1
        # ssml_text += '</speak>'
        #
        # input_text = texttospeech.SynthesisInput(ssml = ssml_text)
        #
        # # Note: the voice can also be specified by name.
        # # Names of voices can be retrieved with client.list_voices().
        # voice = texttospeech.VoiceSelectionParams(
        #     language_code="en-GB",
        #     name= "en-GB-News-G",
        #     ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        # )
        #
        # audio_config = texttospeech.AudioConfig(
        #     audio_encoding=texttospeech.AudioEncoding.MP3,
        # )
        #
        #
        # response = client.synthesize_speech(
        #     request={"input": input_text, "voice": voice, "audio_config": audio_config, "enable_time_pointing": ["SSML_MARK"]}
        # )
        #
        #
        # # The response's audio_content is binary.
        # with open("output.mp3", "wb") as out:
        #     out.write(response.audio_content)
        #
        # audio_file = os.path.dirname(__file__) + '\output.mp3'
        # media = vlc.MediaPlayer(audio_file)
        # media.play()
        # playsound(audio_file, winsound.SND_ASYNC)

        # count = 0
        # current = 0
        # for i in range(len(response.timepoints)):
        #     count += 1
        #     current += 1
        #     with open("output.txt", "a", encoding="utf-8") as out:
        #         out.write(mark_array[int(response.timepoints[i].mark_name)] + " ")
        #     if i != len(response.timepoints) - 1:
        #         total_time = response.timepoints[i + 1].time_seconds
        #         time.sleep(total_time - response.timepoints[i].time_seconds)
        #     if current == 25:
        #             open('output.txt', 'w', encoding="utf-8").close()
        #             current = 0
        #             count = 0
        #     elif count % 7 == 0:
        #         with open("output.txt", "a", encoding="utf-8") as out:
        #             out.write("\n")
        # time.sleep(2)
        # open('output.txt', 'w').close()

        # Print the contents of our message to console...

        print('------------------------------------------------------')
        # os.remove(audio_file)

        # Since we have commands and are overriding the default `event_message`
        # We must let the bot know we want to handle and invoke our commands...
        last_run_time = current_time
    else:
        return
sync(room.connect())