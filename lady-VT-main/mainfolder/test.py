from googletrans import Translator

# Create a translator object
translator = Translator()

# Define the English string to be translated
english_text = "Hello, how are you?"

# Use the translate() method to translate the English string to Japanese
japanese_translation = translator.translate(english_text, dest='ja')

# Print the translated text
print(japanese_translation.text)