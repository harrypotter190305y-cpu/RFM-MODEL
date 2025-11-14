import urllib.request, json
req = urllib.request.Request('http://127.0.0.1:5000/generate_segments', data=json.dumps({}).encode('utf-8'), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.loads(resp.read().decode())
    print('status', data.get('success'))
    pie = data.get('pie_chart')
    bar = data.get('bar_chart')
    print('pie present:', bool(pie), 'len:', len(pie) if pie else 0)
    print('bar present:', bool(bar), 'len:', len(bar) if bar else 0)
    summary = data.get('summary')
    print('summary count:', len(summary) if summary else 0)
