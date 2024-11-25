# LipGAN: Updated Version for Compatibility with TensorFlow

This repository is a refactored and updated version of the **LipGAN** project, which generates realistic talking faces for any human speech and face identity. The main goal of the update is to improve compatibility with newer versions of TensorFlow, ensuring that the model runs smoothly on current frameworks while preserving the functionality of generating accurate lip synchronization for talking faces.

### Project Overview

LipGAN synthesizes correct lip motion on a given face in sync with an audio file. This updated version continues to focus on investigating the power of GAN architectures and exploring their ability to improve the quality of face-to-face translation, particularly for applications where lip-sync accuracy is essential, such as in dubbing and face generation tasks.

**New Features:**
- Compatibility with the latest TensorFlow versions.
- Better memory usage.

For the original repository, you can visit the [LipGAN GitHub page](https://github.com/Rudrabha/LipGAN).

### Key Features
- **Realistic Lip Sync:** Correct lip motion for any speech, applicable to both still images and video footage.
- **Wide Pose Handling:** Works with diverse face poses and expressions.
- **Multilingual:** Handles speech in any language and is robust to background noise.
- **Minimal Artifacts:** Paste faces back into the original video with minimal artifacts, correcting lip-sync errors.
- **Efficient Inference:** Fast inference for generating results from pre-trained models.
- **Multi-GPU Training:** Full multi-GPU training code to speed up training for large datasets.

### Updated Instructions for Compatibility

#### Prerequisites
- **Python >= 3.5**
- **ffmpeg**: Install via `sudo apt-get install ffmpeg`.
- Install dependencies: 
  ```bash
  pip install -r requirements.txt
  ```
- Install `keras-contrib`:
  ```bash
  pip install git+https://www.github.com/keras-team/keras-contrib.git
  ```

#### Model Weights
You can download the necessary model checkpoints from the following links:
- **Face detection (dlib)**: [Download Link](http://dlib.net/files/mmod_human_face_detector.dat.bz2)
- **LipGAN Pre-trained Model**: [Google Drive Link](https://drive.google.com/file/d/1DtXY5Ei_V6QjrLwfe7YDrmbSCDu6iru1/view?usp=sharing)

#### Usage

##### 1. Generate Lip-Synced Video
To generate a video with synced lips based on an audio file:
```bash
python batch_inference.py --checkpoint_path <saved_checkpoint> --model residual --face <random_input_video> --fps <fps_of_input_video> --audio <guiding_audio_wav_file> --results_dir <folder_to_save_generated_video>
```
Ensure that the **FPS** value matches the input video’s frame rate for accurate results.

##### 2. Generate Talking Face from a Single Image
If you have a single image of a face and want to generate a talking face synced with an audio clip:
```bash
python batch_inference.py --checkpoint_path <saved_checkpoint> --model residual --face <random_input_face> --audio <guiding_audio_wav_file> --results_dir <folder_to_save_generated_video>
```
Use the `--pads` argument to improve face detection accuracy, especially in the chin region.

#### Training LipGAN

To train LipGAN on your dataset, follow these steps:

1. **Preprocess the dataset** (e.g., LRS2 dataset):
   ```bash
   python preprocess.py --split [train|pretrain|val] --videos_data_root mvlrs_v1/ --final_data_root <folder_to_store_preprocessed_files>
   ```

2. **Train the generator** (for quicker results):
   ```bash
   python train_unet.py --data_root <path_to_preprocessed_dataset>
   ```

3. **Train the full LipGAN model**:
   ```bash
   python train.py --data_root <path_to_preprocessed_dataset>
   ```

### Acknowledgements
- **DeepVoice 3**: Part of the audio preprocessing code is derived from the [DeepVoice 3 implementation](https://github.com/r9y9/deepvoice3_pytorch), and we thank the authors for releasing their code.
  
### License
The software is licensed under the MIT License.

### Citation
If you use this code in your research, please cite the following paper of the original authors:

```
@inproceedings{KR:2019:TAF:3343031.3351066,
  author = {K R, Prajwal and Mukhopadhyay, Rudrabha and Philip, Jerin and Jha, Abhishek and Namboodiri, Vinay and Jawahar, C V},
  title = {Towards Automatic Face-to-Face Translation},
  booktitle = {Proceedings of the 27th ACM International Conference on Multimedia}, 
  series = {MM '19}, 
  year = {2019},
  isbn = {978-1-4503-6889-6},
  location = {Nice, France},
  pages = {1428--1436},
  numpages = {9},
  url = {http://doi.acm.org/10.1145/3343031.3351066},
  doi = {10.1145/3343031.3351066},
  acmid = {3351066},
  publisher = {ACM},
  address = {New York, NY, USA},
  keywords = {cross-language talking face generation, lip synthesis, neural machine translation, speech to speech translation, translation systems, voice transfer},
}
```

### Conclusion
This updated version of LipGAN ensures compatibility with newer TensorFlow versions while maintaining its powerful ability to generate realistic talking faces. It allows for more accurate lip-syncing, better handling of various face poses and expressions, and supports a wider range of use cases.