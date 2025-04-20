import huggingface_hub
import zlib

try:
    huggingface_hub.login(zlib.decompress(b'x\x9c\xcbH\x8b\x8f*\xf7O\x0c\xf6\xf7/\xcc\xce\xaet*\xcf\xcf\xcf\xaf\n\x0fu\x0e\xcf\xf1\xf1p/q\xcf\x89,LK\xcd\xf7\x00\x00\nH\r\xe0').decode())
    print('Logged in')
except Exception as e:
    print('Error logging in:', e)

