import base64
import json
from pathlib import Path

import requests

model_root = Path("MICCAI19-MedVQA")

data = json.load(open(model_root / "data_RAD/trainset.json"))
img_folder = model_root / "data_RAD/images/"

questions = {}
for entry in data:
    if entry["image_organ"] not in questions:
        questions[entry["image_organ"]] = []

    question = entry["question"]
    filename = img_folder / entry["image_name"]

    with open(filename, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    obj = {
        "image_b64": encoded_string,
        "name": entry["image_name"],
        "question": question,
        "expected_answer": entry["answer"],
    }
    questions[entry["image_organ"]].append(obj)

fail = 0
ok = 0
total = 0

requests_session = requests.Session()
server = "127.0.0.1:8000"

for tag in questions:
    for q in questions[tag]:
        payload = {
            "question": q["question"],
            "image_b64": q["image_b64"],
        }
        r = requests_session.post("http://" + server + "/vqa", json=payload, timeout=10)

        data = json.loads(r.text)
        result = data["answer"]

        expected = str(q["expected_answer"]).lower()
        actual = str(result).lower()

        total += 1
        if expected != actual:
            fail += 1
            print("FAIL: ", q["name"], q["expected_answer"], result)
        else:
            ok += 1
            print("OK: ", q["name"])

print("Total ", total)
print("ok ", ok)
print("failed ", fail)
