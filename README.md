This repository is cited in my graduation thesis. Thanks to chldkato for his implementation, it was really useful. Check out his implementation.

My MelGan is implemented in tensorflow. My contribution consist of:
* MelGAN class as a subclass of tf.keras.models.Model
* train_step as a method of MelGAN
* chldkato didn't used metric to evaluate the performance. I used Short Time Objective Intelligibility (STOI) and Mel Cepstral Distortion (MCD)

Original MelGAN paper:
@article{kumar2019melgan,
  title={Melgan: Generative adversarial networks for conditional waveform synthesis},
  author={Kumar, Kundan and Kumar, Rithesh and De Boissiere, Thibault and Gestin, Lucas and Teoh, Wei Zhen and Sotelo, Jose and De Brebisson, Alexandre and Bengio, Yoshua and Courville, Aaron C},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
