package versions

import java.io.File
import java.lang.StringBuilder

class Version3 {
    //Absolute Values
    val I_neuronsCount = 2
    val O_neuronsCount = 1
    val layersCount = 2
    val neuronsPerLayer = 3

    var epochsToTeach = 900

    var printInfo = false
    var loadWeightsFromFile = true
    val savedNeuronValuesFilePath = "./src/resources/myNeuralNetwork"

    val K = 0.7

    var allLayers: Array<Array<Neuron>> = emptyArray()

    var outPutNeurons: Array<Neuron> = emptyArray()

    val inputTable = listOf(listOf(0, 0), listOf(0, 1), listOf(1, 0), listOf(1, 1))
    val outputTable = listOf(0, 1, 1, 0)

    class Neuron(wsCount: Int) {
        var value: Double = 0.0

        var weights = Array(wsCount, {//init neuron's weights
            Math.random()
        }).plus(0.5)

        var error: Double = 0.0
    }

    fun f(x: Double): Double {
        return 1 / (1 + Math.exp(-x))
    }

    fun init() {
        for (i in 0 until layersCount) {//Init all layers
            allLayers += Array(neuronsPerLayer, {//add neuron array
                    it ->
                Neuron(neuronsPerLayer)
            })
        }

        for (i in 0 until O_neuronsCount) {
            outPutNeurons += (Neuron(neuronsPerLayer))
        }
    }

    fun main() {
        println("Creating neurons...")
        init()

        if (loadWeightsFromFile) {
            loadNeuronWeights(savedNeuronValuesFilePath)
            //saveNeuronWeights(savedNeuronValuesFilePath)
        } else {
            println("Teaching...")
            for (i in 0..epochsToTeach) {
                teach()
            }
            saveNeuronWeights(savedNeuronValuesFilePath)
        }

        println("/////////////////////////////////////////")
        for (i in 0 until 4) {
            println("input: ${inputTable[i]}")
            direct(i)
            println("Right answer: ${outputTable[i]}")
            println("Neuron network answer: ${outPutNeurons[0].value}")
            println("Analyzed answer: ${analyze()}")
            println("////////////////////")
        }

    }

    fun teach() {
        for (tableIndex in 0 until outputTable.size) {
            if (printInfo) println("Direct move:")
            direct(tableIndex)

            if (printInfo) println("Backpropagation:")
            backpropagation(tableIndex)
        }
    }

    fun backpropagation(tableIndex: Int) {
        if (printInfo) println("Outputs error:")
        for (neuron in 0 until O_neuronsCount) {
            outPutNeurons[neuron].error = outputTable[tableIndex] - outPutNeurons[neuron].value
            if (printInfo) println("#!ERROR!# ${outPutNeurons[neuron].error}")
        }
        if (printInfo) println("Rest layers' error:")
        for (layer in allLayers.size - 1 downTo 0) {
            if (layer == allLayers.size - 1) {
                for (neuron in 0 until allLayers[layer].size) {
                    allLayers[layer][neuron].error = 0.0
                    for (o_neuron in 0 until O_neuronsCount) {
                        allLayers[layer][neuron].error += outPutNeurons[o_neuron].error * outPutNeurons[o_neuron].weights[neuron]
                    }
                    if (printInfo) println("Layer $layer, $neuron neuron: ${allLayers[layer][neuron].error}")
                }
            } else {
                for (neuron in 0 until allLayers[layer].size) {
                    allLayers[layer][neuron].error = 0.0
                    for (prev_neuron in 0 until allLayers[layer + 1].size) {
                        allLayers[layer][neuron].error += allLayers[layer + 1][prev_neuron].error * allLayers[layer + 1][prev_neuron].weights[neuron]
                    }
                    if (printInfo) println("Layer $layer, $neuron neuron: ${allLayers[layer][neuron].error}")
                }
            }
        }
        if (printInfo) println("Rewriting output ws")
        for (neuron in outPutNeurons) {
            for (w in 0 until neuronsPerLayer) {
                neuron.weights[w] =
                    neuron.weights[w] + K * neuron.error * neuron.value * (1 - neuron.value) * allLayers[allLayers.size - 1][w].value
            }
            neuron.weights[neuron.weights.size - 1] =
                neuron.weights[neuron.weights.size - 1] + K * neuron.error * neuron.value * (1 - neuron.value)
        }

        if (printInfo) println("Rewriting layers' ws")
        for (layer in layersCount - 1 downTo 0) {
            if (layer != 0) {
                for (neuron in allLayers[layer]) {
                    for (w in 0 until neuronsPerLayer) {
                        neuron.weights[w] =
                            neuron.weights[w] + K * neuron.error * neuron.value * (1 - neuron.value) * allLayers[layer - 1][w].value
                    }
                    neuron.weights[neuron.weights.size - 1] =
                        neuron.weights[neuron.weights.size - 1] + K * neuron.error * neuron.value * (1 - neuron.value)
                }
            } else {
                for (neuron in allLayers[layer]) {
                    for (w in 0 until I_neuronsCount) {
                        neuron.weights[w] =
                            neuron.weights[w] + K * neuron.error * neuron.value * (1 - neuron.value) * inputTable[tableIndex][w]
                    }
                    neuron.weights[neuron.weights.size - 1] =
                        neuron.weights[neuron.weights.size - 1] + K * neuron.error * neuron.value * (1 - neuron.value)
                }
            }
        }

    }

    fun direct(tableIndex: Int) {
        for (layer in 0 until allLayers.size) {
            if (layer == 0) {
                if (printInfo) println("get input values for the first layer:")
                for (neuron in allLayers[layer]) {
                    for (w in 0 until I_neuronsCount) {
                        neuron.value += inputTable[tableIndex][w] * neuron.weights[w]
                    }
                    neuron.value += neuron.weights[neuron.weights.size - 1]
                    //print("unactivated: ${neuron.value} ")
                    neuron.value = f(neuron.value)
                    if (printInfo) println("Layer 0, neuron value: ${neuron.value}")
                }
            } else {
                if (printInfo) println("All rest layers:")
                for (neuron in allLayers[layer]) {
                    for (w in 0 until neuronsPerLayer) {
                        neuron.value += allLayers[layer - 1][w].value * neuron.weights[w]
                    }
                    neuron.value += neuron.weights[neuron.weights.size - 1]
                    //print("unactivated: ${neuron.value} ")
                    neuron.value = f(neuron.value)
                    if (printInfo) println("Layer $layer, neuron value: ${neuron.value}")
                }
            }
        }
        //count the output
        if (printInfo) println("Outputs:")
        for (neuron in outPutNeurons) {
            for (w in 0 until neuronsPerLayer) {
                neuron.value += allLayers[allLayers.size - 1][w].value * neuron.weights[w]
            }
            neuron.value += neuron.weights[neuron.weights.size - 1]
            neuron.value = f(neuron.value)
            if (printInfo) println("out: ${neuron.value}")
        }
    }

    fun analyze() = when {
        (outPutNeurons[0].value > 0.5) -> 1
        else -> 0
    }

    fun saveNeuronWeights(saveFileName: String) {
        //out
        val writer = File(saveFileName).writer()
        println("Saving rest layers ws to ${saveFileName}")
        for (neuron in allLayers[0]) {
            for (w in 0 until I_neuronsCount) {
                writer.write(neuron.weights[w].toString() + " ")
            }
            writer.write("[" + neuron.value.toString() + "]")
            writer.write("| ")
        }
        writer.write("\n")
        for (layer in 0 until layersCount) {
            for (neuron in allLayers[layer]) {
                for (w in 0 until neuronsPerLayer) {
                    writer.write(neuron.weights[w].toString() + " ")
                }
                writer.write("[" + neuron.value.toString() + "]")
                writer.write("| ")
            }
            writer.write("\n")
        }

        println("Saving output ws to ${saveFileName}")
        for (neuron in outPutNeurons) {
            for (w in 0 until neuronsPerLayer) {
                writer.write(neuron.weights[w].toString() + " ")
            }
            writer.write("[" + neuron.value.toString() + "]")
            writer.write("| ")
        }
        writer.write("\n")
        //writer.flush()
        writer.close()
    }

    fun getWeightFromLine(line: String, neuronsInLayer: Int, weightsForEachNeuron: Int): ArrayList<ArrayList<Double>> {
        var allWeights = ArrayList<ArrayList<Double>>()

        val builder = StringBuilder()
        var index = 0
        var value = line[index]
        //println(line)
        for (i in 0 until neuronsInLayer) {
            allWeights.add(ArrayList())
            for (j in 0 until weightsForEachNeuron + 1) {

                while (true) {

                    //println("VALUE IS " + value)

                    if (value == '[' || value == ']' || value == '|') {
                        index++
                        if (index >= line.length) {
                            index = 0
                            builder.clear()
                            break
                        }
                        value = line[index]
                        continue
                    }

                    if (value == ' ') {
                        //println("THE BUILDER IS " + builder.toString())
                        if (builder.toString().equals(""))
                            break
                        allWeights[i].add(builder.toString().toDouble())
                        builder.clear()
                        index++
                        if (index >= line.length) {
                            index = 0
                            break
                        }
                        value = line[index]
                        break
                    }

                    builder.append(value)
                    index++
                    if (index >= line.length) {
                        allWeights[i].add(builder.toString().toDouble())
                        builder.clear()
                        index = 0
                        break
                    }
                    value = line[index]
                }
            }
        }

        return allWeights
    }

    fun loadNeuronWeights(loadFileName: String) {
        //in
        val reader = File(loadFileName).bufferedReader()
        println("Loading rest layers ws from ${loadFileName}")
        var line = reader.readLine()
        var weightsForInputNeurons = getWeightFromLine(line, neuronsPerLayer, I_neuronsCount)
        for (neuronNumber in 0 until neuronsPerLayer) {
            for (w in 0 until I_neuronsCount) {
                allLayers[0][neuronNumber].weights[w] = weightsForInputNeurons[neuronNumber][w]
            }
            allLayers[0][neuronNumber].value = weightsForInputNeurons[neuronNumber][I_neuronsCount]
        }

        for (layer in 0 until layersCount) {
            line = reader.readLine()
            weightsForInputNeurons = getWeightFromLine(line, neuronsPerLayer, neuronsPerLayer)
            for (neuronNumber in 0 until neuronsPerLayer) {
                for (w in 0 until neuronsPerLayer) {
                    allLayers[layer][neuronNumber].weights[w] = weightsForInputNeurons[neuronNumber][w]
                }
                allLayers[layer][neuronNumber].value = weightsForInputNeurons[neuronNumber][neuronsPerLayer]
            }
        }

        line = reader.readLine()
        weightsForInputNeurons = getWeightFromLine(line, O_neuronsCount, neuronsPerLayer)
        println("Saving output ws to ${loadFileName}")
        for (neuronNumber in 0 until O_neuronsCount) {
            for (w in 0 until neuronsPerLayer) {
                outPutNeurons[neuronNumber].weights[w] = weightsForInputNeurons[neuronNumber][w]
            }
            outPutNeurons[neuronNumber].value = weightsForInputNeurons[neuronNumber][neuronsPerLayer]
        }
        //reader.flush()
        reader.close()
    }
}