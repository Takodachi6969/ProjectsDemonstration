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

string = 'The following text contains quiet abit of information. The length of the sentence depends on its content, for example, a text with more content has more words, and vice versa.'

substrings = [s.strip() for s in re.split('[.,，。;；!！？?]', string)]
audiolength = 22.4
timediff = audiolength/len(substrings)
starttime = time.time()
for i in range(len(substrings)):
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write('\n'+substrings[i])
    time.sleep(timediff)

