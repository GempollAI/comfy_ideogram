import requests

url = "https://api.ideogram.ai/describe"

payload = {}
files=[
  ('image_file',('1ef7cd45-c964-4e40-89f5-d746af26b41d',open('postman-cloud:///1ef7cd45-c964-4e40-89f5-d746af26b41d','rb'),'application/octet-stream'))
]
headers = {
  'Api-Key': 'SxxB2UzNnp4UGAmL9Fl3uSE37vT7JEOLuqHAxQ-LVuD8l_12PQTPIIkcXSOlGP5KR2R0tpBb5qLtKoRIYTs-2A',
  'Cookie': '__cf_bm=Q.tIVcWUvQF6yMgbmNPDgEqQUETqT6SmpH3.07BQQIM-1727505631-1.0.1.1-YMYR.5MKXg6hSGrAoiBxZ5KQREvf48nICr7TKXu5OskSVXZK3C23orBwLhEQ._CI6J8hscXxjhugVq4h1zeXxw'
}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)
