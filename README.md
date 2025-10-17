This repository is cited in my graduation thesis. Thanks to chldkato for his implementation, it was really useful. Check out his implementation.

My MelGan is implemented in tensorflow. My contribution consist of:
* MelGAN class as a subclass of tf.keras.models.Model
* train_step as a method of MelGAN
* chldkato didn't used metric to evaluate the performance. I used Short Time Objective Intelligibility (STOI) and Mel Cepstral Distortion (MCD)
