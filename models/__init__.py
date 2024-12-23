from . import mfq
from . import IL
MFQ = mfq.MFModel
IL = IL.ILModel
def choose_model(model_name):
    if model_name == "MFQ":
        model = MFQ()
    elif model_name == "IL":
        model = IL()
    return model

