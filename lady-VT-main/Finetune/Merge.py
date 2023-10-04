import json
import re

# read in the JSON file
with open('compilation.json', 'r') as f:
    json_data = json.load(f)

# remove any markers from the JSON data
json_str = json.dumps(json_data, separators=(',', ':'))

# merge all content together into a single string
merged_str = ' '.join(re.findall(r'\S+|\s+', json_str))

# write the merged string to a file
with open('merged.json', 'w') as f:
    f.write(merged_str)