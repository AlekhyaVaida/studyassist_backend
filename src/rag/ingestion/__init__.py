from .pdf import pdf_to_json
from .image import image_to_json
from .url import webpage_to_json

def ingest(args: dict):
    if args["type"] == "pdf":
        return pdf_to_json(args["path"])
    elif args["type"] == "image":
        return image_to_json(args["path"])
    elif args["type"] == "url":
        return webpage_to_json(args["url"])
    else:
        raise ValueError(f"Unsupported file type: {args['type']}")