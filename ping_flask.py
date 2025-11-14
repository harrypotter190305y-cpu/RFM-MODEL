import urllib.request

paths = ['/','/dashboard','/dataset','/generate_segments']
for path in paths:
    try:
        url = 'http://127.0.0.1:5000' + path
        if path == '/generate_segments':
            req = urllib.request.Request(url, data=b'{}', headers={'Content-Type':'application/json'})
            resp = urllib.request.urlopen(req, timeout=20)
        else:
            resp = urllib.request.urlopen(url, timeout=20)
        print(path, '->', resp.getcode())
    except Exception as e:
        print(path, 'error', e)
