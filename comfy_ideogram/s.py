import requests

url = "https://api-ideogram-proxy.gempoll.com/remix"

payload = {'image_request': '''{
  "prompt": "a car on the land",
  "aspect_ratio": "ASPECT_16_9",
  "model": "V_2",
  "magic_prompt_option": "AUTO",
  "image_weight": 100,
  "color_palette": {
    "members": [
      {
        "color_hex": "#121212",
        "color_weight": 0.8
      },
      {
        "color_hex": "#443434",
        "color_weight": 0.2
      }
    ]
  }
}'''}
files=[
  ('image_file',('1ef7cbc8-63cc-4480-b18c-03c8d1efc968',open('1.png','rb'),'application/octet-stream'))
]
headers = {
  'Api-Key': 'SxxB2UzNnp4UGAmL9Fl3uSE37vT7JEOLuqHAxQ-LVuD8l_12PQTPIIkcXSOlGP5KR2R0tpBb5qLtKoRIYTs-2A'
}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)
