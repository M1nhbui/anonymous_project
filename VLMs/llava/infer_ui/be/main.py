import uvicorn
import argparse
from transformers import HfArgumentParser
from infer import ModelArguments, DataArguments, TrainingArguments
from app import app
from infer import build_model_and_tokenizer

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load model từ args
    model, tokenizer = build_model_and_tokenizer(model_args, data_args, training_args)
    model.eval()
    device = training_args.device

    # Gán vào app để sử dụng trong endpoint
    app.model = model
    app.tokenizer = tokenizer
    app.model_args = model_args
    app.training_args = training_args
    app.data_args = data_args
    app.device = device

    # Run server với tùy chọn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
