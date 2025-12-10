import io
import uuid
from fastapi import FastAPI, HTTPException, File, UploadFile, Body
from app.config import imagekit
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv


load_dotenv()


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/personalize")
async def create_persolized_img(
    face_image: UploadFile,
    prompt: str = Body(...)
):
    if not face_image   or not prompt:
        raise HTTPException(status_code=400, detail="Missing required parameters")
   
    client = genai.Client()
    image_bytes = await face_image.read()

    
    try:
        pil_input_image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")
    
    instruction = f"""you are an expert image generate please generate according to the given image
    
    the face in the provided image should be in the generated image with  given instructions: {prompt} """
    response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[instruction, pil_input_image],
    )

    generated_pil_image = None
    for part in response.parts:
        if part.inline_data is not None:
            generated_pil_image = part.as_image()
            break

    if generated_pil_image is None:
        print(f"Gemini response didn't contain an image. Full response: {response}")
        raise HTTPException(status_code=500, detail="AI model failed to generate an image.")
    
    img_byte_arr = io.BytesIO()
    generated_pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    unique_filename = f"gen_{uuid.uuid4().hex[:10]}.png"

    image_uploaded = imagekit.upload_file(
        file=img_byte_arr,
        file_name=unique_filename,
        options={
            "folder":"gemini_generated/",
        }
    )

    if not image_uploaded:
        raise HTTPException(status_code=500, detail="Failed to save image into imagekitio.")
    
    image_uploaded_url = image_uploaded.response_metadata.raw['url']

    return {"Message":"Successfully uploaded", "url":image_uploaded_url}





