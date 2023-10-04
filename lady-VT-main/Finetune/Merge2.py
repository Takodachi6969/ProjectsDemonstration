import json

# Sample JSON data
json_data = 'compilation'

# Parse the JSON data into a Python dictionary
dict_data = json.loads(json_data)

# Convert the dictionary to a string
text_data = json.dumps(dict_data)

# Write the text data to a file
with open('output.txt', 'w') as f:
    f.write(text_data)