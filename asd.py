import json
with open('metadata.json') as fp:
    meta = json.load(fp)
    sorted_meta = dict(sorted(meta.items(), key=lambda item: item[1]))