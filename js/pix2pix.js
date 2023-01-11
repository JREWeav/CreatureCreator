// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* eslint max-len: "off" */
/*
Pix2pix
The original pix2pix TensorFlow implementation was made by affinelayer: github.com/affinelayer/pix2pix-tensorflow
This version is heavily based on Christopher Hesse TensorFlow.js implementation: https://github.com/affinelayer/pix2pix-tensorflow/tree/master/server
*/

/*
Modified to add one layer to the encoder and decoder each
Also removing dependecy on other utils by adding array3DToImage into this file.
Justin Weaver 2023
*/

class Pix2pix {
  /**
   * Create a Pix2pix model.
   * @param {model} model - The path for a valid model.
   * @param {function} callback  - Optional. A function to run once the model has been loaded. If no callback is provided, it will return a promise that will be resolved once the model has loaded.
   */
  constructor(model, callback) {
    /**
     * Boolean to check if the model has loaded
     * @type {boolean}
     * @public
     */
    this.ready = false;
    this.loadCheckpoints(model).then(() => {
      this.ready = true;
      if (callback) {
        callback();
      }
    });
  }

  async loadCheckpoints(path) {
    const checkpointLoader = new CheckpointLoaderPix2pix(path);
    this.variables = await checkpointLoader.getAllVariables();
    return this;
  }

  /**
   * Given an canvas element, applies image-to-image translation using the provided model. Returns an image.
   * @param {HTMLCanvasElement} inputElement
   */
  async transfer(inputElement, callback = () => {}) {
    const input = tf.browser.fromPixels(inputElement);
    const inputData = input.dataSync();
    const floatInput = tf.tensor3d(inputData, input.shape);
    const normalizedInput = tf.div(floatInput, tf.scalar(255));

    const result = tf.tidy(() => {
      const preprocessedInput = Pix2pix.preprocess(normalizedInput);
      const layers = [];
      let filter = this.variables["generator/encoder_1/conv2d/kernel"];
      let bias = this.variables["generator/encoder_1/conv2d/bias"];
      let convolved = Pix2pix.conv2d(preprocessedInput, filter, bias);
      layers.push(convolved);

      for (let i = 2; i <= 9; i += 1) {
        const scope = `generator/encoder_${i.toString()}`;
        filter = this.variables[`${scope}/conv2d/kernel`];
        const bias2 = this.variables[`${scope}/conv2d/bias`];
        const layerInput = layers[layers.length - 1];
        const rectified = tf.leakyRelu(layerInput, 0.2);
        convolved = Pix2pix.conv2d(rectified, filter, bias2);
        const scale = this.variables[`${scope}/batch_normalization/gamma`];
        const offset = this.variables[`${scope}/batch_normalization/beta`];
        const normalized = Pix2pix.batchnorm(convolved, scale, offset);
        layers.push(normalized);
      }

      for (let i = 9; i >= 2; i -= 1) {
        let layerInput;
        if (i === 9) {
          layerInput = layers[layers.length - 1];
        } else {
          const skipLayer = i - 1;
          layerInput = tf.concat(
            [layers[layers.length - 1], layers[skipLayer]],
            2
          );
        }
        const rectified = tf.relu(layerInput);
        const scope = `generator/decoder_${i.toString()}`;
        filter = this.variables[`${scope}/conv2d_transpose/kernel`];
        bias = this.variables[`${scope}/conv2d_transpose/bias`];
        convolved = Pix2pix.deconv2d(rectified, filter, bias);
        const scale = this.variables[`${scope}/batch_normalization/gamma`];
        const offset = this.variables[`${scope}/batch_normalization/beta`];
        const normalized = Pix2pix.batchnorm(convolved, scale, offset);
        layers.push(normalized);
      }

      const layerInput = tf.concat([layers[layers.length - 1], layers[0]], 2);
      let rectified2 = tf.relu(layerInput);
      filter = this.variables["generator/decoder_1/conv2d_transpose/kernel"];
      const bias3 = this.variables["generator/decoder_1/conv2d_transpose/bias"];
      convolved = Pix2pix.deconv2d(rectified2, filter, bias3);
      rectified2 = tf.tanh(convolved);
      layers.push(rectified2);

      const output = layers[layers.length - 1];
      const deprocessedOutput = Pix2pix.deprocess(output);
      return deprocessedOutput;
    });

    await tf.nextFrame();
    callback(array3DToImage(result));
  }

  static preprocess(inputPreproc) {
    const result = tf.tidy(() => {
      return tf.sub(tf.mul(inputPreproc, tf.scalar(2)), tf.scalar(1));
    });
    return result;
  }

  static deprocess(inputDeproc) {
    const result = tf.tidy(() => {
      return tf.div(tf.add(inputDeproc, tf.scalar(1)), tf.scalar(2));
    });
    return result;
  }

  static batchnorm(inputBat, scale, offset) {
    const result = tf.tidy(() => {
      const moments = tf.moments(inputBat, [0, 1]);
      const varianceEpsilon = 1e-5;
      return tf.batchNorm(
        inputBat,
        moments.mean,
        moments.variance,
        offset,
        scale,
        varianceEpsilon
      );
    });
    return result;
  }

  static conv2d(inputCon, filterCon) {
    const tempFilter = filterCon.clone();
    const result = tf.tidy(() => {
      return tf.conv2d(inputCon, tempFilter, [2, 2], "same");
    });
    tempFilter.dispose();
    return result;
  }

  static deconv2d(inputDeconv, filterDeconv, biasDecon) {
    const result = tf.tidy(() => {
      const convolved = tf.conv2dTranspose(
        inputDeconv,
        filterDeconv,
        [
          inputDeconv.shape[0] * 2,
          inputDeconv.shape[1] * 2,
          filterDeconv.shape[2],
        ],
        [2, 2],
        "same"
      );
      const biased = tf.add(convolved, biasDecon);
      return biased;
    });
    return result;
  }
}

//Added the following function from utils.js from within the ML5.js distribution
const array3DToImage = (tensor) => {
  const [imgWidth, imgHeight] = tensor.shape;
  const data = tensor.dataSync();
  const canvas = document.createElement("canvas");
  canvas.width = imgWidth;
  canvas.height = imgHeight;
  const ctx = canvas.getContext("2d");
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < imgWidth * imgHeight; i += 1) {
    const j = i * 4;
    const k = i * 3;
    imageData.data[j + 0] = Math.floor(256 * data[k + 0]);
    imageData.data[j + 1] = Math.floor(256 * data[k + 1]);
    imageData.data[j + 2] = Math.floor(256 * data[k + 2]);
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);

  // Create img HTML element from canvas
  const dataUrl = canvas.toDataURL();
  const outputImg = document.createElement("img");
  outputImg.src = dataUrl;
  outputImg.style.width = imgWidth;
  outputImg.style.height = imgHeight;
  return outputImg;
};

const pix2pix = (model, callback = () => {}) => new Pix2pix(model, callback);
