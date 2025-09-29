import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/voice2text")
async def voice2text(
    request_id: str,
    audio_file1: str,
    audio_file2: str,
):
    # TODO 转录实现
    return "test"   # 返回结果

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9995, log_level='info')