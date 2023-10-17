const fs = require("fs")
const { NeuralNetworkDataPair } = require("./NeuralNetwork.js")

const load = (imagesPath, lablesPath, n_max) => {
  const imageBuffer = fs.readFileSync(imagesPath)
  const lableBuffer = fs.readFileSync(lablesPath)

  let n_images = imageBuffer.readInt32BE(4)
  if(n_max != undefined && n_images > n_max) n_images = n_max
  
  let rows = imageBuffer.readInt32BE(8)
  let columns = imageBuffer.readInt32BE(12)

  let images = []
  for(let i = 0; i < n_images; i++) {
    let pixels = []

    for(let y = 0; y < rows; y++) {
      for(let x = 0; x < columns; x++) {
        pixels.push(imageBuffer[i * rows * columns + (x + (y * rows)) + 16])
      }
    }

    images.push(new MNISTImageData(pixels, lableBuffer[i + 8], rows, columns))
  }

  return images;
}

const printImage = (imageData) => {
  console.log("=====", imageData.lable, "=====")

  let s = "";
  for(let y = 0; y < imageData.rows; y++) {
    for(let x = 0; x < imageData.columns; x++) {
      s += imageData.at(x, y) + ","
      for(let j = 0; j < 3 - imageData.at(x, y).toString().length; j++) {
        s += " ";
      }
    }
    s += "\n"
  }
      
  // for(let i = 0; i < imageData.pixels.length; i++) {
  //   if(i % 28 == 0) s += "\n"
  //   s += imageData.pixels[i] + ","
  //   for(let j = 0; j < 3 - imageData.pixels[i].toString().length; j++) {
  //     s += " ";
  //   }
  // }
  console.log(s, "\n")
}

class MNISTImageData extends NeuralNetworkDataPair {
  constructor(pixels = [], lable = -1, rows = 28, columns = 28) {
    let input = pixels.map(p => p / 255)
    let output = new Array(10).fill(0)
    output[lable] = 1

    
    super(input, output)
    
    this.pixels = pixels
    this.lable = lable

    this.rows = rows
    this.columns = columns
  }

  at(x, y) {
    return this.pixels[x + (y * this.rows)]
  }
}


module.exports = { load, printImage, MNISTImageData }