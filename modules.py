import esprima
import requests


class HuggingFaceClient():
    def __init__(self, token) -> None:
        self.TOKEN = token
        self.BASE_URL = "https://datasets-server.huggingface.co"
        self.AUTH = {"Authorization":f"Bearer {self.TOKEN}"}
        
    def get(self, method, params=None):
        if params != None:
            return requests.get(
                url=f"{self.BASE_URL}/{method}",
                headers=self.AUTH,
                params=params
            )
        else:
            return requests.get(
                url=f"{self.BASE_URL}/{method}",
                headers=self.AUTH,
            )
            

