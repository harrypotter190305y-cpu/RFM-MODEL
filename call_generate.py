import urllib.request
import json

url='http://127.0.0.1:5000/generate_segments'
req=urllib.request.Request(url, data=json.dumps({}).encode('utf-8'), headers={'Content-Type':'application/json'})
resp=urllib.request.urlopen(req, timeout=60)
print(resp.status)
print(resp.read()[:200])
