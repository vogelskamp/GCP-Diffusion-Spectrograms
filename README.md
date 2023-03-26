# Denoising Diffusion Probabilistic Model to generate spectrograms of human voice
This project was developed for the Audio Data Science module at the University of Applied Sciences in Düsseldorf. Based on [dome272's implementation](https://github.com/dome272/Diffusion-Models-pytorch) of the [DDPM paper](https://arxiv.org/pdf/2006.11239.pdf), the project aims to train a diffusion model on spectrograms of human sentences in effort to produce [gibberish](https://en.wikipedia.org/wiki/Gibberish), aka nonsense speech. The project is meant to be baseline research for whether or not diffusion models can be used to generate spectrograms of human speech, so later approaches can use it for speech-to-text approaches including text vectors of the spoken sentences as input for the training.

# How to use
## Preparing the data
The `GCPSpectrogramSet` included in [`data_set.py`](./src/data_set.py) expects a numpy array saved as a `.npy` file stored in a GCP bucket. To generate the numpy array, open the [`batch_gen_data.py`](./data%20preparation/batch_gen_data.py) and adjust the `ORIGIN_FOLDER_PATH` and `DESTINATION_FOLDER_PATH` variables as required. Running the script will generate a numpy array for each processed audio file individually, so in a following step you need to run the [combine_data.py](./data%20preparation/combine_data.py) script to combine them into one file. Once the file is created, it can be uploaded to a GCP bucket of choice.

## Modifying the model
The underlying U-Net model is defined in [modules.py](./src/modules.py), where parameters for each layer can be modified as needed. Note that the layer sizes are dependent on each previous one, so local testing upon modifying the values is advised. This can be done by running the [modules.py](./src/modules.py) directly, which will do a test run with randomly generated data to check the validity of the model. To adjust hyperparameters, [argparse](https://docs.python.org/3/library/argparse.html) was used to parse command parameters. The following hyperparameters were set up to customize the training:
| name | short form | description | default |
| -- | -- | -- | -- |
| --epochs | -e | The amount of epochs the model should train for. | 100 |
| --batch-size| -bs | The batch size to use for training. | 5 |
| --image_size | -is | The image size in height, width format as a tuple. | (64, 256) |
| --learning_rate | -lr | The desired learning rate. | 3e-4 |
| --device | -d | The device to use for training. | cuda |
| --dataset | -d | The name of the dataset uploaded to the GCP bucket. | data256_test.py |
| --bucket_name | -bn | The name of the GCP bucket. | diffusion-project-data |
| --result_bucket | -rb | The name of the GCP bucket where the samples from each epoch, as well as the resulting model will be saved. | diffusion-project-results-na |
| --name | -n | The name of the diffusion model used when running the project. | DDPM_Unconditional256x |

## Training the model
The project is set up to use Google Cloud Platform for its training, so the only local dependency required to run the project is the [Google Cloud Cli](https://cloud.google.com/sdk/gcloud). After setup, the `publish.sh` script can be modified and run to create a custom docker container for the training and generate a custom job on the GCP Vertex AI platform used to train AI models.

## Testing results
The project includes a `image_to_audio` function in [audio_utils.py](./data%20exploration/audio_utils.py) to convert generated images back to audio.