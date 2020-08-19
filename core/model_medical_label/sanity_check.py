import requests
from io import BytesIO
import base64
import json

medical_imgs = [
    "https://media.sciencephoto.com/image/c0371577/800wm/C0371577-Stroke,_MRI_brain_scan.jpg",
    "https://prod-images-static.radiopaedia.org/images/34839897/e0bfac31c00d077d18aca7ab33b495_gallery.jpeg",
    "https://prod-images-static.radiopaedia.org/images/157210/332aa0c67cb2e035e372c7cb3ceca2_jumbo.jpg",
    "https://image.freepik.com/photos-gratuite/technologie-rayon-chirurgie-x-ray-xray_1172-444.jpg",
    "https://media.wired.com/photos/5ba015a0ab6e142d95f93dac/125:94/w_1196,h_900,c_limit/R.Kim-eyescan-w.jpg",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ4IevMy0w_3XO3Wc-PNRB5lBgwqvoSSttiAw&usqp=CAU",
    "https://www.startradiology.com/uploads/images/english-class-x-elbow-fig-5-normal-anatomy-elbow-lateral-blanco.jpg",
]


requests_session = requests.Session()
server = "http://127.0.0.1:8000/prefilter"


for medical_img in medical_imgs:
    response = requests.get(medical_img)
    img = BytesIO(response.content).getvalue()
    encoded_string = base64.b64encode(img).decode()

    payload = {
        "image_b64": encoded_string,
    }

    r = requests_session.post(server, json=payload, timeout=10)

    data = json.loads(r.text)
    output = data["answer"]
    print(output)
