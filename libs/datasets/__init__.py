from .voc import VOC, VOCAug, Custom

def get_dataset(name):
    return {
        "voc": VOC,
        "vocaug": VOCAug,
        "custom": Custom,
        "voccrf": VOCCRF,
    }[name]
