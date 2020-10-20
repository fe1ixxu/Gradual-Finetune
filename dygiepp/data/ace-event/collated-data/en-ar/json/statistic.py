import json

# docs = ["test.json", "ar-train.json", "en-train.json", "dev.json"]
docs = ["dev.json", "test.json", "ar-train.json", "en-train.json"]
for doc in docs:
    fr = open(doc, "r")
    line = fr.readline()
    sent_lengths = 0
    line_num = 0
    while(line):
        data = json.loads(line)
        sent_lengths +=  len(data['sentences'])
        line_num += 1
        line = fr.readline()
    print("doc {} has {} documents, {} sentences".format(doc, line_num, sent_lengths))