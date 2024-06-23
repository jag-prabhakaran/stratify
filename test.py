import json

score = [0, 1, 2]

with open("file.json", 'w') as f:
    # indent=2 is not needed but makes the file human-readable 
    # if the data is nested
    json.dump(score, f, indent=2) 
