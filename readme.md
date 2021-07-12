This is the repository for code and demo of the ISMIR 2021 paper CollageNet. The code is incomplete, and will be completed in the future.

# code

### Training
- CollageNet: sh scripts/train_adversarial.sh
- EC2-VAE: sh scripts/train_EC2.sh
- poly: sh scripts/train_poly.sh

### Dependencies
1) python 3.8.8

2) pytorch 1.7.1

3) cudatoolkit 11.0.221

4) pretty-midi 0.2.9

### Reference
<https://github.com/buggyyang/Deep-Music-Analogy-Demos>

<https://github.com/ZZWaang/polyphonic-chord-texture-disentanglement>

<https://github.com/music-x-lab/POP909-Dataset>

# demos

### fusion demo
Some fusion demos along with the one in Figure 1 of the paper.

The user control inputs are recorded in the file names.

### user control demo
Several outputs of the model with different user control inputs.

The user control input **cmel**=c_mp=c_mr=1-c_ac=1-c_at
