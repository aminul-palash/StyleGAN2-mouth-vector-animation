### Generated animated faces singing music.

For genererating all 5 stems from an audio I have used spleeter.
**Spleeter** is [Deezer](https://www.deezer.com/) source separation library with pretrained models
written in [Python](https://www.python.org/) and uses [Tensorflow](https://tensorflow.org/). It makes it easy
to train source separation model (assuming you have a dataset of isolated sources), and provides
already trained state of the art model for performing various flavour of separation :

* Vocals (singing voice) / accompaniment separation ([2 stems](https://github.com/deezer/spleeter/wiki/2.-Getting-started#using-2stems-model))
* Vocals / drums / bass / other separation ([4 stems](https://github.com/deezer/spleeter/wiki/2.-Getting-started#using-4stems-model))
* Vocals / drums / bass / piano / other separation ([5 stems](https://github.com/deezer/spleeter/wiki/2.-Getting-started#using-5stems-model))

## Quick start

You can easily generate all 5 stems without installing anything ? In spleeter official doccumentation they have set up a [Google Colab](https://colab.research.google.com/github/deezer/spleeter/blob/master/spleeter.ipynb). Remember to add **-p spleeter:5stems**

```bash
spleeter separate -i audio_example.mp3 -p spleeter:5stems -o output
```

you can play with the parameter of psi values, and the mouth_open multiplier. Also, there are more [vectors](https://rolux.org/media/stylegan2/vectors/) to try out.
**The "vocals" stem is tied to the mouth vectors, with bass and other parts of the music tied to other aspects of the image.
## Demo Example

Here is the [youtube link](https://youtu.be/fUNhu3mx5is) , that I have generated this audio reactive faces.
 
