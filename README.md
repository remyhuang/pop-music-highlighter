# Pop Music Highlighter
TensorFlow implementation of [Pop Music Highlighter: Marking the Emotion Keypoints](https://arxiv.org/abs/1802.10495)
* An attention-based music highlight extraction model to capture the emotion attention score
* Model: Non-recurrent Neural Attention Modeling by Late Fusion with positional embeddings (NAM-LF (pos))

**Please cite this paper if this code/work is helpful:**

    @article{huang2018highlighter,
      title={Pop music highlighter: Marking the emotion keypoints},
      author={Huang, Yu-Siang and Chou, Szu-Yu and Yang, Yi-Hsuan},
      journal={arXiv preprint arXiv:1802.10495},
      year={2018}
    }

## Environment
* Python 2.7
* TensorFlow 1.2.0
* NumPy 1.11.0
* LibROSA 0.5.1

Note: you need to put your own audio(`.mp3 format`) in the `input` folder before you run the code.

	$ git clone https://github.com/remyhuang/pop-music-highlighter.git 	
    $ python main.py

## Outputs
In the `main.py`, you can set the __extracted highlight length__ and whether save the outputs of audio, score and highlight.
* __audio__: short clip of highlight from the original song (.wav format)
* __score__: emotion attention score of every second (.npy format)
* __highlight__: time interval of highlight (.npy format)

You can change the I/O settings by editing the `model.py` for your own purpose.

## Possible Errors
* No audio input in the `input` folder.
* The highlight length you set is shorter than the original length of audio.

## Contact
Please feel free to contact [Yu-Siang Huang](https://remyhuang.github.io/) if you have any questions.
