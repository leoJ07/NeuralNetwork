const math = require("mathjs")
const MNIST = require("./MNIST-Loader.js")
const { NeuralNetwork } = require("./NeuralNetwork.js")




console.log("==== loading training and test data ====")
const trainingData = MNIST.load("./data/training-images", "./data/training-lables")
const testData = MNIST.load("./data/test-images", "./data/test-lables")

console.log("==== seting up the neural network ====")
let n = new NeuralNetwork([784, 30, 10], false)
// let res = n.forward(math.random([784, 1]))
// let e = trainingData[0]
// let b = n.forward(e.input)

console.log("==== training the network ====")
n.SGD(trainingData, 30, 10, 3.0, testData) // training, epocs, batchsize, eta, test
// n.updateMiniBatch(testData, 10)
// console.log(n.evaluate(testData))
// console.log("before:", b, "=", e.output)
// console.log("after:", n.forward(e.input), "=", e.output)
// console.log(n.backprop())

n.writeBufferToFile("./data.json")




// TODO: Change every vector to a 2d matrix of size [n, 1] and fix all the sigmoid map functions. Use math.resize to cnvert a vector to a matrix
// math.map loops through all the elements in the matrix even when multi dimentional. 