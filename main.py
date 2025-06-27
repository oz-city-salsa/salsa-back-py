from fastapi import FastAPI, WebSocket
import base64
import cv2
import numpy as np
import json
import logging
import os

# FastAPI app
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Cliente conectado")
    while True:
        try:
            data = await websocket.receive_text()
            # Directly send back the received data
            await websocket.send_text(data)
        except Exception as e:
            print("Erro:", e)
            break
