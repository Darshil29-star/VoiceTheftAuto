import io
import json
import ssl
import urllib.request
from PIL import Image

ctx = ssl._create_unverified_context()


def test_task(name, task, query, roles, num_images=1):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()

    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="query"\r\n\r\n')
    body.write(f"{query}\r\n".encode())

    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="task"\r\n\r\n')
    body.write(f"{task}\r\n".encode())

    for r in roles:
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="roles"\r\n\r\n')
        body.write(f"{r}\r\n".encode())

    for i in range(num_images):
        img = Image.new(
            "RGB",
            (120, 120),
            color=(40 + i * 30, 100 + i * 20, 60 + i * 40),
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="files"; filename="img_{i}.jpg"\r\n'.encode()
        )
        body.write(b"Content-Type: image/jpeg\r\n\r\n")
        body.write(img_bytes)
        body.write(b"\r\n")

    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        "https://geosense-ai-phi.vercel.app/analyze",
        data=body.getvalue(),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Mozilla/5.0",
        },
    )

    with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
        print(f"[{name}] HTTP {r.status} | Task: {data.get('task')}")
        print(f"  Answer: {data.get('answer')[:110]}...")
        print(
            f"  Confidence: {data.get('confidence', {}).get('value')} ({data.get('confidence', {}).get('level')})"
        )
        print(f"  Visualizations: {len(data.get('visualizations', []))}")


test_task(
    "VQA Counting",
    "vqa",
    "How many fuel storage tanks and ships are visible?",
    ["optical"],
    1,
)
test_task(
    "Bi-Temporal Change",
    "change_analysis",
    "What changed between the before and after acquisitions?",
    ["before", "after"],
    2,
)
test_task(
    "Optical-SAR Fusion",
    "optical_sar_fusion",
    "Fuse optical and SAR imagery to identify water and urban structures",
    ["optical", "sar"],
    2,
)
