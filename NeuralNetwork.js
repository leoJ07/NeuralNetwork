const math = require("mathjs")
const bm = require("jsboxmuller")
const fs = require("fs")

class NeuralNetwork {

  constructor(structure = [], debug = false) {
    if(structure.length < 2) throw new Error("The network structure has to have at least one input and one output layer")
    
    this.structure = [...structure]
    this.layers = structure.length
    this.inputs = structure[-1] = structure.shift()
    this.debuging = debug

    this.buffer = { batches: [] }
    
    this.weights = structure.map((x, l) => this.createSNDRandomMatrix([structure[l], structure[l-1]]))
    this.biases = structure.map((x, l) => this.createSNDRandomMatrix([structure[l], 1]))
  }

  forward(a) {
    // if(math.size(a).length != 1) throw new Error("The input must be a vector")
    // if(math.size(a)[0] != this.inputs) throw new Error("Input data isn't the same size as the networks input layer. " + math.size(x) + " != " + this.inputs)
    // this.debug("___Running forward(a)___")
    // this.debug("a", math.size(a), ":", a)
    
    if(!math.equal(math.size(a), [this.inputs, 1]) && !math.equal(math.size(a), [this.inputs])) throw new Error("Input data isn't the same size as the networks input layer. " + math.size(a) + " != " + this.inputs)
    a = math.resize(a, [math.size(a)[0], 1])
    
    for(let l = 0; l < this.weights.length; l++) {
      let dot = math.multiply(this.weights[l], a)
      let z = math.add(dot, this.biases[l])
      a = math.map(z, this.sigmoid)

      // this.debug("weights", math.size(this.weights[l]), ":", this.weights[l])
      // this.debug("biases", math.size(this.biases[l]), ":", this.biases[l])
      // this.debug("dot", math.size(dot), ":", dot)
      // this.debug("z", math.size(z), ":", z)
      // this.debug("a", math.size(a), ":", a)
    }
    return a
  }

  SGD(trainingData, epochs, batchSize, eta, testData) {
    let n_test, n = trainingData.length
    if(testData) n_test = testData.length

    for(let j = 1; j <= epochs; j++) {      
      trainingData.sort((a, b) => 0.5 - Math.random())
      let miniBatches = []
      for(let i = 0; i < n; i += batchSize) {
        miniBatches.push(trainingData.slice(i, i + batchSize))
      }

      for(let i = 0; i < miniBatches.length; i++) {
        this.updateMiniBatch(miniBatches[i], eta)
      }

      if(testData)
        console.log("Epoche", j + ":", this.evaluate(testData) + "/" + n_test)
      else
        console.log("Epoche", j, "compleate")
    }
  }

  updateMiniBatch(minibatch, eta) {
    let nabla_w = []
    this.weights.forEach(w => nabla_w.push(math.zeros(math.size(w))))
    
    let nabla_b = []
    this.biases.forEach(b => nabla_b.push(math.zeros(math.size(b))))

    for(let i = 0; i < minibatch.length; i++) {
      let [ delta_nabla_w, delta_nabla_b ] = this.backprop(minibatch[i].input, minibatch[i].output)
      
      nabla_w = nabla_w.map((nw, i) => math.add(nw, delta_nabla_w[i]))
      nabla_b = nabla_b.map((nb, i) => math.add(nb, delta_nabla_b[i]))
    }

    this.debug("nabla_w", nabla_w)
    this.debug("nabla_b", nabla_b)

    this.weights = this.weights.map((w, i) => math.subtract(w, math.multiply(nabla_w[i], eta/minibatch.length)))
    this.biases = this.biases.map((b, i) => math.subtract(b, math.multiply(nabla_b[i], eta/minibatch.length)))
  }

  backprop(x, y) { // d is for gathering debugging data
    // if(math.size(x).length != 1 || math.size(y).length != 1) throw new Error("The input and output must be a vector")
    
    // if(math.size(x)[0] != this.inputs) throw new Error("Input data isn't the same size as the networks input layer. " + math.size(x) + " != " + this.inputs)
    // if(math.size(y)[0] != this.biases[this.biases.length - 1].length) throw new Error("Output data isn't the same size as the networks input output. " + math.size(y) + " != " + this.biases[this.biases.length - 1].length)
    if(!math.equal(math.size(x), [this.inputs, 1]) && !math.equal(math.size(x), [this.inputs])) throw new Error("Input data isn't the same size as the networks input layer. " + math.size(x) + " != " + this.inputs)
    if(!math.equal(math.size(y), [this.biases[this.biases.length - 1].length, 1]) && !math.equal(math.size(y), [this.biases[this.biases.length - 1].length])) throw new Error("Output data isn't the same size as the networks output layer. " + math.size(y) + " != " + this.biases[this.biases.length - 1].length);
    
    x = math.resize(x, [math.size(x)[0], 1])
    y = math.resize(y, [math.size(y)[0], 1])
    
    let nabla_w = []
    this.weights.forEach(w => nabla_w.push(math.zeros(math.size(w))))
    
    let nabla_b = []
    this.biases.forEach(b => nabla_b.push(math.zeros(math.size(b))))

    let activation = x
    let activations = [x]
    let zs = []

    this.weights.forEach((weights, l) => {
      // console.log("____Layer", l + "____")
      // console.log("weights", weights)
      // console.log("activation", activation)
      let dot = math.multiply(weights, activation)
      // console.log("dot", dot)
      // console.log("b", this.biases[l])
      let z = math.add(dot, this.biases[l])
      zs.push(z)

      // console.log("z", z)
    
      activation = math.map(z, e => this.sigmoid(e))
      // console.log("new activation", activation)
      activations.push(activation)
    })

    // console.log("y", y)
    let error = math.dotMultiply(this.cost_derivative(activations[activations.length - 1], y), math.map(zs[zs.length - 1], z => this.sigmoid_prime(z)))
    nabla_b[nabla_b.length - 1] = error
    // console.log(math.size(activations[activations.length - 2]), activations[activations.length - 2])
    // console.log(math.transpose(activations[activations.length - 2]))
    // console.log(math.size(error), error)
    nabla_w[nabla_w.length - 1] = math.multiply(error, math.transpose(activations[activations.length - 2]))

    for(let l = 2; l < this.layers; l++) {
      let z = zs[zs.length - l]
      let sp = math.map(z, e => this.sigmoid_prime(e))
      
      error = math.dotMultiply(math.multiply(math.transpose(this.weights[this.weights.length - l + 1]), error), sp)

      nabla_b[nabla_b.length - l] = error
      nabla_w[nabla_w.length - l] = math.multiply(error, math.transpose(activations[activations.length - l - 1]))
    }
    
    return [ nabla_w, nabla_b ]
  }

  evaluate(testData) {
    let testResults = testData.map(e => [this.argMax(this.forward(e.input)), this.argMax(e.output)])
    return testResults.reduce((p, c) => p + (c[0] === c[1]), 0)
  }

  cost_derivative(output_activations, y) {
    return math.subtract(output_activations, y)
  }

  sigmoid(z) {
    return 1 / (1 + math.exp(-z))
  }

  sigmoid_prime(z) {
    return this.sigmoid(z) * (1 - this.sigmoid(z))
  }

  // utility functions
  argMax(array) {
    return array.reduce((p, c, i) => p[0] >= c? p : [c, i], [array[0], 0])[1]
  }

  createSNDRandomMatrix(size) {
    let m = math.zeros(size)
    return math.map(m, () => bm())
    // let m = math.random(size)
    // return math.map(m, ran => Math.sqrt(-2.0 * Math.log(ran)) * Math.cos(2.0 * Math.PI * Math.random()))
  }

  // debuging
  debug(...strings) {
    if(!this.debuging) return
    console.log(...strings)
  }

  writeBufferToFile(file) {
    fs.writeFileSync(file, JSON.stringify(this.buffer, undefined, "\n"))
  }
}

class NeuralNetworkDataPair {
  constructor(input, output) {
    this.input = input
    this.output = output
  }
}

module.exports = { NeuralNetwork, NeuralNetworkDataPair }