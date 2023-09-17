# coding=utf-8
import os
from pathlib import Path

import requests
import base64
import logging
import json
import hashlib

try:
    from settings import webhook_url
except:
    from .settings import webhook_url
Private = 'Private' # 个人
Group = 'Group'     # 群
Discuss = 'Discuss' # 组
URL = 'http://192.168.20.203:8900/api/cqp/send'

def sendqq(type, qq, msg, pics=None):
    def getByte(path):
        with open(path, 'rb') as f:
            img_byte = base64.b64encode(f.read())
        img_str = img_byte.decode('ascii')
        return img_str

    picsbuffer = []
    picsname = []

    if pics is None:
        pics = []
    if not (isinstance(pics, list) or isinstance(pics, tuple)):
        pics = [pics,]

    for path in pics:
        picsbuffer.append(getByte(path))
        picsname.append(os.path.basename(path))

    resp = requests.post(
        url=URL,
        data={
            'msg': msg,
            'qq': qq,
            'type': type,
            'pics': picsbuffer,
            'picsname': picsname
        }
    )
    if resp.status_code > 400:
        raise Exception(resp.reason if resp.content == '' else resp.content)


class QQSenderHandler(logging.Handler):
    def __init__(self, type, qq):
        logging.Handler.__init__(self)
        self._type = type
        self._qq = qq

    def emit(self, record):
        msg = self.format(record)
        # sendqq(self._type, self._qq, msg)


def send_wecom(text, bot=None):
    if bot is None:
        url = webhook_url
    else:
        url = bot
    headers = {"Content-Type": "application/json"}
    data = {
        "msgtype": "text",
        "text": {
            "content": text,
        },
        "safe": "0"}
    send_data = json.dumps(data, ensure_ascii=False).encode("utf-8")
    r = requests.post(url, data=send_data, headers=headers)


def send_wecom_markdown(text):
    url = webhook_url
    headers = {"Content-Type": "application/json"}
    data = {
        "msgtype": "markdown",
        "markdown": {
            "content": text,
        },
        "safe": "0"}
    send_data = json.dumps(data, ensure_ascii=False).encode("utf-8")
    r = requests.post(url, data=send_data, headers=headers)


def send_wecom_image(img_src, bot=None):
    img_src = str(img_src)
    if bot is None:
        url = webhook_url
    else:
        url = bot
    headers = {"Content-Type": "application/json"}
    f = open(img_src, 'rb')
    img_byte = base64.b64encode(f.read()).decode('utf-8')
    f.close()
    data = {
        "msgtype": "image",
        "image": {
            "base64": img_byte,
            "md5": hashlib.md5(open(img_src, 'rb').read()).hexdigest()
        },
        "safe": "0"}
    send_data = json.dumps(data, ensure_ascii=False).encode("utf-8")
    r = requests.post(url, data=send_data, headers=headers)


class WeComSenderHandler(logging.Handler):
    def __init__(self, ):
        logging.Handler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        send_wecom(msg)


if __name__ == '__main__':
    # pass
    # sendqq(Private, 304047349, u'test\r\n[CQ:image,file=图片1.jpg]', [u'D:\图片1.jpg'])
    # sendqq(Private, 304047349, 'test\r\n[CQ:image,file=图片1.jpg]', ['D:\图片1.jpg'])   # python3
    img = Path(r'C:\Users\ligy\Downloads\www.jpg')
    send_wecom_image(img)
