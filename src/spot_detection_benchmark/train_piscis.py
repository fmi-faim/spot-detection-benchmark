from piscis.training import train_model

train_model(
    model_name='piscis_customed',
    dataset_path=r"C:\Data\soucemma\spot-detection-benchmark\datasets\training_test\piscis_dataset",
    input_size=(256, 256), # maybe change to (512, 512) ?
    device='cpu', # or 'cuda' if you have a GPU
)





