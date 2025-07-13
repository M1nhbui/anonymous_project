from fastapi import FastAPI
from pydantic import BaseModel
import base64
import pickle
from infer import test_one_case

app = FastAPI()

class InferRequest(BaseModel):
    pet_array: str
    ct_array: str
    question: str

@app.post("/infer")
def infer(request: InferRequest):
    # Decode base64 -> numpy array
    pet_np = pickle.loads(base64.b64decode(request.pet_array))
    ct_np = pickle.loads(base64.b64decode(request.ct_array))
    print(pet_np.shape)
    result = test_one_case(
        pet_image=pet_np,
        ct_image=ct_np,
        question_text=request.question,
        model=app.model,
        tokenizer=app.tokenizer,
        model_args=app.model_args,
        training_args=app.training_args,
        data_args=app.data_args,
        device=app.device
    )

    return {
        "answer": result["answer"],
        "prompt": result["prompt"]
    }
