import json

# docs = ["test.json", "ar-train.json", "en-train.json", "dev.json"]
docs = ["dev.json", "en1k-ar-train-sf.json", "en200-ar-train-sf.json", "en500-ar-train-sf.json" , "test.json", "ar-train.json"]
for doc in docs:
    fr = open("../large/" + doc, "r")
    fw = open("../large-filter/" + doc , "w")
    line = fr.readline()
    key = 0
    while(line):
        data = json.loads(line)
        sent_lengths = [len(x) for x in data['sentences']]
        if min(sent_lengths) >= 2:
            data["doc_key"] = key
            fw.write(json.dumps(data, default=int) + "\n")
            key += 1 
        line = fr.readline()
