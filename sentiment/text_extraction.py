from pypdf import PdfReader
from fastapi import UploadFile
import tempfile
from model_prediction import predict_sentiment

async def get_sentiment(file_object: UploadFile, sentiment_model, tokenizer):

    contents = await file_object.read()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(contents)
        tmp_file_path = tmp_file.name

    reader = PdfReader(tmp_file_path)

    content = ''
    for page in reader.pages:
        content += page.extract_text()

    import os
    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
        
    output = predict_sentiment(content, sentiment_model, tokenizer)

    return output