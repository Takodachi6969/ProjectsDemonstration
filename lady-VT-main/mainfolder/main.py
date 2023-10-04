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


last_run_time = 0
Count = 0
language = -1

class Bot(commands.Bot):
    conversation = list()


    def __init__(self):
        # Initialise our Bot with our access token, prefix and a list of channels to join on boot...
        # prefix can be a callable, which returns a list of strings or a string...
        # initial_channels can also be a callable which returns a list of strings...

        super().__init__(token='gnbo4vak7yvm82u3l589rnd10q3wnw', prefix='!', initial_channels=['ikaros_senpai'])

    async def event_ready(self):
        # Notify us when everything is ready!
        # We are logged in and ready to chat and use commands...
        print(f'Logged in as | {self.nick}')

    async def event_message(self, message):
        global Count
        global language
        if message.author.name == 'ikaros_senpai' and message.content == 'English please':
            language = 1
            return
        if message.author.name == 'ikaros_senpai' and message.content == 'Japanese please':
            language = -1
            return
        #Using voiceVox engine to generate japanese text
        async def Sound(text,k):
            async with Client() as client:
                audio_query = await client.create_audio_query(
                    text, speaker= k
                )
                with open("output.wav", "wb") as f:
                    f.write(await audio_query.synthesis())

        # Messages with echo set to True are messages sent by the bot...
        # For now we just want to ignore them...

        if message.echo:
            return

        # download the words corpus
        # nltk.download('words')

        # Check if the message contains english words
        # if not any(word in message.content for word in nltk.corpus.words.words()):
        #     return
        # Check if the message is too long
        if len(message.content) > 1000:
            return
        global last_run_time
        current_time = time.time()
        if current_time - last_run_time < 5 and message.author.name != 'ikaros_senpai':  # check if the time since last run is less than 10 seconds
            return
        print('------------------------------------------------------')
        print(message.content)
        print(message.author.name)
        print(Bot.conversation)
        print(Count)
        translator = Translator()

        if Count >=5:
            Bot.conversation.clear()
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
        if is_english_or_chinese(message.content) == "unknown":
            return

        # Replace 'message.content' with the actual string you want to test
        result = is_english_or_chinese(message.content)


        Bot.conversation.append(f'CHAT: {result}')
        text_block = '\n'.join(Bot.conversation)

        prompt = open_file('prompt_chat.txt').replace('<<BLOCK>>', text_block)
        if message.author.name == "ikaros_senpai":
            print(prompt + 'You are now speaking to Peter'+ '\nBRITAIN:')
            prompt = prompt + 'You are now speaking to Peter'+ '\nBRITAIN:'

        else:
            print(prompt + 'You are now speaking to your viewers'+ '\nBRITAIN:')
            prompt = prompt + 'You are now speaking to your viewers' + message.author.name + '\nBRITAIN:'

        if message.author.name == 'ikaros_senpai' and message.content == 'sudo STFU':
            response = 'filtered'
        else:

            response = gpt3_completion(prompt)
        # response = "Hello, how are you?"
        # print(prompt)
        Count = Count + 1

        with open("EnglishOut.txt", "w", encoding = "utf-8") as file:
            # Write the string to the file
            file.write(message.author.name + ': ' + message.content + '. '+response)
        with open("ChineseOut.txt", "w", encoding = "utf-8") as file:
            p = message.author.name + ': ' + message.content + '. '+ translator.translate(response, dest='zh-cn').text
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
        nametranslated = translator.translate(message.author.name, dest='ja')
        contenttranslated = translator.translate(message.content, dest='ja')
        namevoice = replace_english_with_hello(nametranslated.text)
        messagevoice = replace_english_with_hello(contenttranslated.text)
        output_string = replace_english_with_hello(input_string)
        print(namevoice + ' ' + messagevoice + ' ' + output_string)

        print('BRITAIN:' , response)
        with open("record.txt", "a", encoding="utf-8") as out:
            out.write(message.content + response)

        if(Bot.conversation.count('BRITAIN: ' + response) == 0):
            Bot.conversation.append(f'BRITAIN: {response}')
        k = 0

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

            responseV = requests.post(url, headers=headers, json=data, stream=True)
            with open("output.mp3", "wb") as f:
                f.write(responseV.content)
            # audio_content = AudioSegment.from_file(io.BytesIO(response.content), format="wav")
            # play(audio_content)



        if language == 1:
            print('ENTTS')
            await EL_TTS(message.author.name + ' ' + message.content + ' ' + response)
            audio = AudioSegment.from_file('output.mp3', format='mp3')
        if language == -1:
            print('JPtTS')
            await Sound(namevoice + ' ' + messagevoice + ' ' + output_string, k)
            audio = AudioSegment.from_file('output.wav', format='wav')

        # Load MP3 file


        # Play MP3 file
        play(audio)

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
        #Play the sound
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
        # with open('ChineseOut.txt', 'r', encoding = "utf-8") as file:
        #     string = file.read()
        # substrings = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]

        # Save the substrings to a text file
        # with open('output.txt', 'w', encoding='utf-8') as file:
        #     file.write('\n'.join(substrings))
        #
        # with open('EnglishOut.txt', 'r', encoding = "utf-8") as file:
        #     string = file.read()
        # substrings = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]
        #
        # # Save the substrings to a text file
        # with open('output2.txt', 'w', encoding='utf-8') as file:
        #     file.write('\n'.join(substrings))
        # with open('ChineseOut.txt', 'r', encoding = "utf-8") as file:
        #     string = file.read()
        # substrings = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]
        #
        # Save the substrings to a text file
        with open('ChineseOut.txt', 'r', encoding = "utf-8") as file:
            string = file.read()
        substringss = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]

        for i in range(len(substringss)):
            with open('output.txt', 'w', encoding='utf-8') as file:
                file.write('\n' + substringss[i])
            time.sleep(timediff)

        with open('EnglishOut.txt', 'r', encoding = "utf-8") as file:
            string = file.read()
        substrings = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]

        # Save the substrings to a text file
        for i in range(len(substrings)):
            with open('output2.txt', 'w', encoding='utf-8') as file:
                file.write('\n' + substrings[i])
            time.sleep(timediff)


        
        # client = texttospeech.TextToSpeechClient()
        #
        # response = message.content + "? " + response
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
        #playsound(audio_file, winsound.SND_ASYNC)


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
        await self.handle_commands(message)
        last_run_time = current_time ##################################################

    @commands.command()
    async def hello(self, ctx: commands.Context):
        # Here we have a command hello, we can invoke our command with our prefix and command name
        # e.g ?hello
        # We can also give our commands aliases (different names) to invoke with.

        # Send a hello back!
        # Sending a reply back to the channel is easy... Below is an example.
        await ctx.send(f'Hello {ctx.author.name}!')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'neuro-380516-7894fbc5bba3.json'
bot = Bot()
bot.run()
# bot.run() is blocking and will stop execution of any below code here until stopped or closed.



