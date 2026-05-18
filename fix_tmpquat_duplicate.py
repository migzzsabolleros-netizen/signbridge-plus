from pathlib import Path
path = Path('vroid_app/templates/index.html')
data = path.read_bytes()
old = b'const tmpQuat = new THREE.Quaternion();\r\n\r\nfunction resolveBoneNode'
new = b'function resolveBoneNode'
if old not in data:
    raise SystemExit('duplicate tmpQuat not found')
path.write_bytes(data.replace(old, new, 1))
print('duplicate tmpQuat removed')
